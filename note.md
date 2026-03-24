有四个途径生成mapping进行迭代。
1、随机生成：实现方便，环境要求不高，但是生成合法mapping的概率较小，最终效果没有保证。
2、利用cosa：实现方便，环境要求高，效果有保证，但是可扩展性一般，如果cosa不支持，可能就不能用。
3、利用Soter:实现稍麻烦，环境要求不高，效果有保证，支持的操作类型比较多，但运行时间很长。可以不迭代，直接使用第一次迭代的结果
4、自己实现：实现最麻烦，环境要求不高，效果不确定，可扩展性最好。

针对每一个随机的维度组合，合法的mapping生成一个就可以了，可以把外围做完，每个方式都试试。

还有些细节，Soter分析的并行模式（两层并行）跟COSA分析的并行模式（三层并行）并不相同，所以对每个架构，可以直接尝试到底支持几重并行，最后针对模板进行专门设计。

可以先以cosa为基础，这个最方便，然后逐渐扩展。

要尽量少修改cosa和soter的代码，把他们当接口使用，实现自己的方法时，可能有很多代码跟他们重复，这是合理的

应该默认已经有了mapping，只是mapping是由不同工具产生的。

目前直接在nanuto上基于cosa实现了目标，但无法找到曾经直接用cosa跑扩展问题的最优结果，仍然比cosa和Soter好很多。

借鉴gamma_timeloop生成自己的自己实现的。
1、维度选择可以根据维度大小作为选择权重，维度小的没必要花费过多的随机次数
2、gamma_timeloop没有考虑约束，基本是随机生成。gamma_loop中“随机”选项表示

论文实验就是三部分：
1、效果，对比cosa、Soter针对Soter实验中的各种操作进行，可能可以加上时间对比
2、自身对比，不同扩展效果对比
3、扩展性，能扩展cosa和Soter的能力
4、消融实验，对比扩展和优化。

如果把mapping看成连续的，也就是说取值可以是分数，然后评估时向上取整，这些值构成扩展的最优问题维度，以及对应的划分，能不能考虑梯度下降的方法实现呢？目前使用的遗传算法其实就是某种意义的梯度下降但并不直接。模拟退火是不是更好一些？或者马尔科夫-蒙特卡洛？或者把cosa变成连续问题建模？相当于把ilp当成梯度下降。

smoothe中成本函数需要可微

export LD_LIBRARY_PATH=/home/mingchuan/Desktop/zousheng/accelergy-timeloop-infrastructure/src/timeloop/lib:/home/mingchuan/Desktop/zousheng/accelergy-timeloop-infrastructure/src/timeloop/build:$LD_LIBRARY_PATH


将Soter作为初始种群送进去，效果并没有提升，看来soter方法已经很好了，是上限不够，应该设计问题规模。


Soter是生成一组解，然后让transformer学习这组解好的为什么好，然后给出调整策略，不断迭代，但是限制非常多，只能做一或两级的并行，更高的并行度无法生成，并且修改了对应的并行度，效果并不好，值在给的架构上效果好。


```
    def _integerize_with_staged_optimization(self, solution):
        def calculate_remaining_capacity(data, fixed_rows):
            """
            计算在固定行确定后，各个存储层次的剩余容量
            
            参数:
            data: 当前的数据矩阵
            fixed_rows: 已固定的行索引列表
            
            返回:
            一个字典，包含每个存储层次的总容量、已占容量和剩余容量
            """
            result = {}
                     
            # 处理buffer约束
            for buffer_key in sorted(self.buffer_name_list.keys()):
                buffer_name = self.buffer_name_list[buffer_key]
                buffer_level = self.temporal_level[buffer_name]
                
                # 跳过DRAM
                if buffer_name == 'DRAM':
                    continue

                # 获取该buffer约束涉及的所有行
                all_involved_rows = list(range(buffer_level + 1))
                
                # 计算buffer约束的已占容量
                used_capacity = 0
                tensors_list = self.buffer_tensor_dict[buffer_name]
                
                for tensor in tensors_list:
                    tensor_capacity = 1
                    for l in all_involved_rows:
                        for dim in self.tensor_dimensions[tensor]:
                            # 如果行已固定，使用其值；否则假设为1
                            if l in fixed_rows:
                                tensor_capacity *= data[l][dim]
                            else:
                                tensor_capacity *= 1  # 假设值为1
                    
                    used_capacity += tensor_capacity
                
                result[buffer_name] = used_capacity            
            return result

        def remaining_capacity_constraint(data, fixed_rows):
            """
            检查在固定行确定后，指定存储层次的剩余容量约束是否满足
            
            参数:
            data: 当前的数据矩阵
            fixed_rows: 已固定的行索引列表
            storage_name: 存储层次名称
            
            返回:
            True如果约束满足，False否则
            """
            capacities = calculate_remaining_capacity(data, fixed_rows)
            
            # 检查指定存储层次的剩余容量是否非负
            for buffer_key in sorted(self.buffer_name_list.keys()):                
                buffer_capacity = self.buffer_size_list[buffer_key]
                buffer_name = self.buffer_name_list[buffer_key]
                if buffer_name == 'DRAM':
                    continue
                if capacities[buffer_name] > buffer_capacity:
                    return False
            
            return True 

               
        def maximize_row_with_constraint(data, row, constraint_func, target, objective_func, fixed_rows):
            """
            递归DFS版本：最大化指定行，同时满足特定约束，只处理原始值不为1的列
            """
            # 找出原始值为1的位置
            fixed_positions = np.where(data[row] == 1)[0]
            # 确定需要处理的列（原始值不为1的列）
            columns_to_process = [j for j in range(len(data[row])) if j not in fixed_positions]

            # 按原始值降序排序（关键修改点）
            columns_to_process.sort(key=lambda j: data[row][j], reverse=True)

            best_data = data.copy()
            best_objective = 0
            iterations = [0]

            def bfs(current_data, col_index):
                iterations[0] += 1
                nonlocal best_data, best_objective
                # 检查约束条件
                if constraint_func(current_data) <= target:
                    # 检查所有剩余容量约束
                    if remaining_capacity_constraint(current_data, fixed_rows + [row]):
                        current_objective = objective_func(current_data)
                        if current_objective > best_objective:
                            best_data = current_data.copy()
                            best_objective = current_objective
                            # print(f"找到更好的解，目标函数值: {best_objective}")
                        return True

                if col_index >= len(columns_to_process):
                    return False

                # 获取当前要处理的列索引
                actual_col = columns_to_process[col_index]
                original_value = data[row][actual_col]

                # 从大到小尝试值，找到第一个满足条件的值后停止
                for value in range(original_value, 0, -1):
                    new_data = current_data.copy()
                    new_data[row][actual_col] = value

                    # # 检查是否满足基本条件
                    # if np.all(new_data > 0):
                    #     # 检查约束条件
                    #     if constraint_func(new_data) > target:
                    #         continue
                    #     print("new_data2", new_data)
                    # 递归处理下一列
                    flag = bfs(new_data, col_index + 1)

                    if flag:
                        break

            # 从第一列开始DFS
            bfs(data.copy(), 0)

            # print(f"第{row}行最大化完成，目标函数值: {best_objective}，共执行 {iterations[0]} 次迭代")
            return best_data
        
        # 已固定的行列表
        fixed_rows = []

        data = np.ceil(solution).astype(int)
        for spatial_name in self.spatial_level:
            spatial_capacity = self.spatial_size_list[spatial_name]
            sp_level = self.spatial_level[spatial_name] 
            data = maximize_row_with_constraint(data, sp_level, lambda x: np.prod(x[sp_level]), spatial_capacity, lambda x: np.prod(x[sp_level]), fixed_rows)
            fixed_rows.append(sp_level)

        def create_buffer_constraint(buffer_name):
            """创建用于maximize_row_with_constraint的约束函数"""
            def constraint_func(data):
                buffer_level = self.temporal_level[buffer_name]
                tensors_list = self.buffer_tensor_dict[buffer_name]
                buffer_capacity = 0
                
                for tensor in tensors_list:
                    tmp = 1
                    for l in range(buffer_level+1):
                        for dim in self.tensor_dimensions[tensor]:
                            tmp *= data[l][dim]
                    buffer_capacity += tmp
                return buffer_capacity 

            return constraint_func
        
        
        for buffer_key in sorted(self.buffer_name_list.keys()):
            buffer_capacity = self.buffer_size_list[buffer_key]
            buffer_name = self.buffer_name_list[buffer_key]
            buffer_level = self.temporal_level[buffer_name]
            if buffer_name == 'DRAM':
                break
            constraint_func = create_buffer_constraint(buffer_name)  
            data = maximize_row_with_constraint(data, buffer_level, constraint_func, buffer_capacity, constraint_func, fixed_rows)
            fixed_rows.append(buffer_level)

        def compute_last_row(data):
            """计算最后一行的值，每一列等于对应的target除以这一列前几行的乘积向上取整"""
            # 计算前n-1行的乘积
            product = np.prod(data[:len(data)-2], axis=0)
            
            # 获取对应的target值
            targets = np.array(self.dimension)
            
            # 计算最后一行的值：target除以乘积，然后向上取整
            last_row = np.ceil(targets / product).astype(int)
            
            # 确保最后一行的每个元素至少为1
            last_row = np.maximum(last_row, 1)
            
            # 更新数据
            new_data = data.copy()
            new_data[len(data)-1] = last_row
            
            # print("最后一行计算完成")
            return new_data

        solution = compute_last_row(data)
        return solution
```


每一级存储器的带宽好像都有用，就是看看对面积的影响了。