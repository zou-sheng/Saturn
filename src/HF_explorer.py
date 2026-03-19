from datetime import datetime
from cost_model import Timeloop
import pathlib
from input_objs import Arch, Prob, Mapspace, Mapping
import utils
import random
from pulp import LpMaximize, LpProblem, LpVariable, PULP_CBC_CMD
import math
import copy
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import uuid
import os
import shutil
import numpy as np
from collections import deque
import time


class MappingExplorer:
    def __init__(self, operator_instance, accelerator_dir, accelerator, mapper, type, version, report_dir, optim_obj):
        if optim_obj == 'latency':
            self.fitness_obj = ['cycles']
        else:
            self.fitness_obj = [optim_obj]
        self.timeloop_out_config_path = f'./tmp/out_config_{datetime.now().strftime("%H:%M:%S")}'
        self.operator_instance = None
        self.problems_list = operator_instance
        self.tensor_dimensions_list = []
        for i in self.problems_list:
            dim = []
            for d in ['R', 'S', 'P', 'Q', 'C', 'K', 'H', 'N']:
                dim.append(i['problem']['instance'][d])
            self.tensor_dimensions_list.append(dim)
        
        self.report_dir = report_dir 
        self.expanded_scope = expanded_scope
        self.expanded_dict = expanded_dict
        self.mapper = mapper

        arch_path = accelerator_dir
        self.accelerator = Arch(arch_path, accelerator, version)
        mapspace_path = pathlib.Path('{}/{}/{}/{}.yaml'.format(accelerator_dir, mapper, accelerator, 'mapspace')).resolve()
        self.mapspace = Mapspace(mapspace_path)
        prob_path = pathlib.Path('{}/{}/{}/{}.yaml'.format(accelerator_dir, mapper, accelerator, 'problem')).resolve()
        self.problem = Prob(prob_path)
        self.buffer_tensor_dict = self.mapspace.buffer_tensor_dict
        self.bypass = self.mapspace.bypass
        self.tensor_dimensions = self.problem.get_tensor_dimensions()

        # self.cost_model = Timeloop(in_config_path='./SpatialAccelerators', out_config_path=self.timeloop_out_config_path,
        #                            accelerator=accelerator, opt_obj=self.opt_obj)

        buffer_name_list, buffer_size_list, buffer_spmap_cstr, num_buffer_levels = self.accelerator.get_arch_info()

        self.buffer_name_list = buffer_name_list
        self.buffer_size_list = buffer_size_list
        self.buffer_spmap_cstr = buffer_spmap_cstr
        self.buffers_with_spmap = set([key for key, value in self.buffer_spmap_cstr.items() if value > 1])
        self.num_buffer_level = num_buffer_levels

        self.buf_energy_cost = self.get_default_buffer_energy_cost()

        self.dimension, self.dimension_dict = self.problem.get_problem_info()
        self.expanded_dimension_dict = self.get_expanded_problem()

       
        # 用于存储temporal层级
        self.temporal_level = {}
        # 用于存储spatial层级
        self.spatial_level = {}
        self.spatial_size_list = {}

        self.obj_level = []

        index = 0

        self.targets = []
        self.type = []

        # 对 buffer_name_list 的键按一定顺序排序
        for key in sorted(self.buffer_name_list.keys()):
            buffer_name = self.buffer_name_list[key]
            if key in self.buffers_with_spmap:
                self.spatial_level[buffer_name] = index
                self.temporal_level[buffer_name] = index + 1
                self.obj_level.append(index)
                index += 2
                self.spatial_size_list[buffer_name] = self.buffer_spmap_cstr[key]
                self.targets.append(buffer_name)
                self.type.append('spatial')
            else:
                self.temporal_level[buffer_name] = index
                index += 1
            self.targets.append(buffer_name)
            self.type.append('temporal')

        for key in sorted(self.buffer_name_list.keys()):
            buffer_name = self.buffer_name_list[key]
            self.obj_level.append(self.temporal_level[buffer_name])

        self.para_dim = parameter_dimension

    def shuffle_factor_order(self, mapping, alpha=0.5):
        if random.random() < alpha:
            org_mapping = copy.deepcopy(mapping)
            # 随机选择一个列表的键
            random_key = random.choice(list(mapping.factor_dict.keys()))

            # 获取随机选择的列表
            list_to_shuffle = mapping.factor_dict[random_key]

            # 打乱列表
            random.shuffle(list_to_shuffle) 
            # if not self.is_mapping_valid(mapping):
            #     mapping = org_mapping



    def mutate_factor(self, mapping, alpha=0.5):
        if random.random() < alpha:
            org_mapping = copy.deepcopy(mapping)
            # 随机选择一个列表的键
            random_key1, random_key2 = random.sample(list(mapping.factor_dict.keys()), 2)

            # 获取随机选择的列表
            list_to_modify1 = mapping.factor_dict[random_key1]
            list_to_modify2 = mapping.factor_dict[random_key2]

            # 随机选择两个不同的索引
            index1, index2 = random.sample(range(len(list_to_modify1)), 2)

            # 选择这两个数字
            num1 = list_to_modify1[index1]
            num2 = list_to_modify1[index2]
            num3 = list_to_modify2[index1]
            num4 = list_to_modify2[index2]

            # 计算第一个数字的因子
            factors = [i for i in range(1, num1 + 1) if num1 % i == 0]

            # 随机选择一个因子
            random_factor = random.choice(factors)

            # 将因子乘到第二个数字上
            modified_num2 = num2 * random_factor
            modified_num1 = int(num1 / random_factor)
            list_to_modify1[index1] = modified_num1
            list_to_modify1[index2] = modified_num2
            if int(num4 / random_factor) == 0:
                pass
            else:
                list_to_modify2[index1] = num3 * random_factor
                list_to_modify2[index2] = int(num4 / random_factor)
            
            # if not self.is_mapping_valid(mapping):
            #     mapping = org_mapping



    def mutate_permutation(self, mapping, alpha=0.5):
        if random.random() < alpha:
            candidate_permutation = ['RSPQCKNH',    # 循环顺序是内->外
                                    'PQNRSCKH',
                                    'QPNRSCKH',
                                    'SRCPQKNH',
                                    'RSCPQKNH',
                                    'KQRSPCNH',
                                    'KPRSQCNH',
                                    'KSRPQCNH',
                                    'KRSPQCNH',
                                    'HRSPQCKN',
                                    'HPQNRSCK',
                                    'HQPNRSCK',
                                    'HSRCPQKN',
                                    'HRSCPQKN',
                                    'HKQRSPCN',
                                    'HKPRSQCN',
                                    'HKSRPQCN',
                                    'HKRSPQCN',
                                    ]
            # 随机选择一个当前列表中的索引
            index_to_replace = random.randint(0, len(mapping.permutation_list) - 1)

            # # 从候选列表中随机选择一个新的 permutation
            new_permutation = random.choice(candidate_permutation)
            # s = "RSPQCKNH"
            # s_list = list(s)
            # random.shuffle(s_list)
            # new_permutation = "".join(s_list) 
            # 替换选择的 permutation
            mapping.permutation_list[index_to_replace] = new_permutation

    # 随机选择一个维度的扩展
    def mutate_dimemsion(self, mapping, alpha=0.5):
        if random.random() < alpha:
            org_mapping = copy.deepcopy(mapping)
            for k in mapping.factor_dict.keys():
                if random.random() < alpha:
                    try:
                        new_dim = random.choice(self.expanded_dimension_dict[k])
                      
                        # 跟mapping中factor的个数有关
                        new_fator = utils.generate_similar_factor_list(mapping.factor_dict[k], new_dim)
                        product = 1
                        for num in new_fator:
                            product *= num
                        if product < self.problem[k]:
                            pass
                        else:
                            mapping.factor_dict[k] = new_fator
                    except:
                        pass   
            # if not self.is_mapping_valid(mapping):
            #     mapping = org_mapping

    def generate_prob_for_cosa(self, operator_instance):
        # 要保留的键
        desired_keys = [
            'C', 'Hdilation', 'Hstride', 'K', 'N', 'P', 'Q', 'R', 'S', 'H', 'Wdilation', 'Wstride'
        ]

        # 构建目标字典
        prob_dict = {
            'problem': {
                key: operator_instance[key] for key in desired_keys if key in operator_instance
            }
        }
        prob_dict['problem']['shape'] = 'cnn-layer'

        return prob_dict

    
    def generate_mapping(self, dimension, p):
        row = len(self.temporal_level) + len(self.spatial_level)
        sol = self.lpsolver(dimension, p[:row])
        mapping_dict, dimension_dict = utils.generate_mapping_for_lpsolver2(sol, self.targets, self.type, self.bypass, p[row:])
        
        prob = copy.deepcopy(self.problem.problem)       
        for key in dimension_dict.keys():
            prob['problem']['instance'][key] = dimension_dict[key]
        return mapping_dict, prob

    def lpsolver(self, dimension_list, p):
        rows = len(self.temporal_level) + len(self.spatial_level)
        cols = 8 # 问题维度R, S, P, Q, C, K, H, N

        # 创建一个最大化问题
        prob = LpProblem("Matrix_Mapping_Problem", LpMaximize)

        # 创建矩阵变量，每个元素是一个非负的连续变量
        matrix = [[LpVariable(f"x_{i}_{j}", lowBound=0) for j in range(cols)] for i in range(rows)]

        # 定义目标函数：矩阵元素的加权和
        objective = 0
        if self.para_dim == 1:
            idx = 0
            p_i = 0
            for spatial_name in self.spatial_level:
                obj_sum = 0
                sp_level = self.spatial_level[spatial_name] 
                for c in range(cols):
                    obj_sum += matrix[sp_level][c]
                obj_sum = obj_sum * p[p_i]
                objective += obj_sum
                idx += 1
                p_i += 1


            idx = 0
            # 除DRAM外每一个存储层次的存储容量约束
            for buffer_key in sorted(self.buffer_name_list.keys()):
                obj_sum = 0
                buffer_name = self.buffer_name_list[buffer_key]
                if buffer_name == 'DRAM':
                    break
                buffer_level = self.temporal_level[buffer_name]
                tensors_list = self.buffer_tensor_dict[buffer_name]
                for l in range(buffer_level+1):
                    for tensor in tensors_list:
                        for dim in self.tensor_dimensions[tensor]:
                            obj_sum += matrix[l][dim]
                obj_sum = obj_sum * p[p_i]
                objective += obj_sum
                idx += 1
                p_i += 1
            
        elif self.para_dim == 2:
            for r in range(rows):
                for c in range(cols):
                    objective += matrix[r][c] * p[r][c]
        
        prob += objective
        
        # 除DRAM外每一个存储层次的存储容量约束
        for buffer_key in sorted(self.buffer_name_list.keys()):
            buffer_capacity = 0
            buffer_name = self.buffer_name_list[buffer_key]
            if buffer_name == 'DRAM':
                break
            buffer_level = self.temporal_level[buffer_name]
            tensors_list = self.buffer_tensor_dict[buffer_name]
            for l in range(buffer_level+1):
                for tensor in tensors_list:
                    for dim in self.tensor_dimensions[tensor]:
                        buffer_capacity += matrix[l][dim]
            # print("buffer_capacity: ", buffer_capacity)
            if len(tensors_list) > 0:
                prob += buffer_capacity <= len(tensors_list)*math.log2(self.buffer_size_list[buffer_key]/len(tensors_list))
        
        # 并行容量约束
        for spatial_name in self.spatial_level:
            spatial_capacity = 0
            sp_level = self.spatial_level[spatial_name] 
            for c in range(cols):
                spatial_capacity += matrix[sp_level][c]
            # print("spatial_capacity: ", spatial_capacity)
            prob += spatial_capacity <= math.log2(self.spatial_size_list[spatial_name])

        # 维度约束
        for c in range(cols):
            col_sum = sum(matrix[r][c] for r in range(rows))
            prob += col_sum == math.log2(dimension_list[c])

        prob.solve(PULP_CBC_CMD(msg=0))
        # print(prob)
        
        # 记录开始时间
        # start_time = time.time()
        solution = [[2**matrix[i][j].value() for j in range(cols)] for i in range(rows)]
        # print(solution)
        # print("time1: ", time.time()-start_time)
        solution = self._integerize_with_staged_optimization(solution, p)
        # print(solution)
        # print("time2: ", time.time()-start_time)
        return solution
    
    def _integerize_with_staged_optimization1(self, solution):
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
                constraint['keep']
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
    
    def _integerize_with_staged_optimization2(self, solution):
        # 已固定的行列表
        fixed_rows = set()

        data = np.ceil(solution).astype(np.int64)
        
        # 处理空间层
        for spatial_name in self.spatial_level:
            spatial_capacity = self.spatial_size_list[spatial_name]
            sp_level = self.spatial_level[spatial_name]
            data = maximize_row_with_constraint(
                self.buffer_info,
                data,
                sp_level,
                lambda x: np.prod(x[sp_level]),
                spatial_capacity,
                lambda x: np.prod(x[sp_level]),
                fixed_rows
            )
            fixed_rows.add(sp_level)
        
        # 处理缓冲层
        for buffer_key in sorted(self.buffer_name_list.keys()):
            buffer_capacity = self.buffer_size_list[buffer_key]
            buffer_name = self.buffer_name_list[buffer_key]
            buffer_level = self.temporal_level[buffer_name]
            
            if buffer_name == 'DRAM':
                break
            
            def constraint_func(data):
                buffer_level = self.temporal_level[buffer_name]
                tensors_list = self.buffer_tensor_dict[buffer_name]
                buffer_capacity = 0
                
                for tensor in tensors_list:
                    tmp = 1
                    for l in range(buffer_level + 1):
                        for dim in self.tensor_dimensions[tensor]:
                            tmp *= data[l, dim]
                    buffer_capacity += tmp
                return buffer_capacity
            
            data = maximize_row_with_constraint(
                self.buffer_info,
                data,
                buffer_level,
                constraint_func,
                buffer_capacity,
                constraint_func,
                fixed_rows
            )
            fixed_rows.add(buffer_level)
        
        # 计算最后一行
        solution = compute_last_row(data, self.dimension)
        return solution

    def _integerize_with_staged_optimization3(self, solution):
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

        def compute_column_upper_bound(data, row, col, fixed_rows, constraint_func, target):
            """
            计算指定列在满足约束条件下的最大可能值
            """
            # 获取原始值作为初始上界
            original_value = data[row][col]
            
            # 尝试二分查找确定上界
            low, high = 1, original_value
            best_valid = 1
            
            while low <= high:
                mid = (low + high) // 2
                test_data = data.copy()
                test_data[row][col] = mid
                
                # 检查约束条件
                if constraint_func(test_data) <= target and remaining_capacity_constraint(test_data, fixed_rows + [row]):
                    best_valid = mid
                    low = mid + 1  # 尝试更大的值
                else:
                    high = mid - 1  # 尝试更小的值
            
            return best_valid

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

                # 计算当前列的上界
                upper_bound = compute_column_upper_bound(current_data, row, actual_col, fixed_rows, constraint_func, target)
                
                # 只尝试有效范围内的值
                for value in range(upper_bound, 0, -1):
                    new_data = current_data.copy()
                    new_data[row][actual_col] = value

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

    def _integerize_with_staged_optimization4(self, solution):
        def calculate_remaining_capacity(data, fixed_rows):
            """计算存储层次剩余容量"""
            result = {}
            for buffer_key in sorted(self.buffer_name_list.keys()):
                buffer_name = self.buffer_name_list[buffer_key]
                buffer_level = self.temporal_level[buffer_name]
                if buffer_name == 'DRAM':
                    continue
                all_involved_rows = list(range(buffer_level + 1))
                used_capacity = 0
                for tensor in self.buffer_tensor_dict[buffer_name]:
                    tensor_capacity = 1
                    for l in all_involved_rows:
                        for dim in self.tensor_dimensions[tensor]:
                            tensor_capacity *= data[l][dim] if l in fixed_rows else 1
                    used_capacity += tensor_capacity
                result[buffer_name] = used_capacity
            return result

        def remaining_capacity_constraint(data, fixed_rows):
            """检查容量约束"""
            capacities = calculate_remaining_capacity(data, fixed_rows)
            for buffer_key in sorted(self.buffer_name_list.keys()):
                buffer_name = self.buffer_name_list[buffer_key]
                if buffer_name == 'DRAM':
                    continue
                if capacities[buffer_name] > self.buffer_size_list[buffer_key]:
                    return False
            return True

        def compute_column_upper_bound(data, row, col, fixed_rows, columns_processed, constraint_func, target):
            """计算列的有效上界（不超过原始值，且未处理列设为1）"""
            original_value = data[row][col]
            test_data = data.copy()
            
            # 未处理的列设为1（包括当前列未处理时，但当前列会被单独设置）
            for c in range(len(data[row])):
                if c != col and c not in columns_processed:
                    test_data[row][c] = 1
            
            low, high = 1, original_value
            best_valid = 1
            
            while low <= high:
                mid = (low + high) // 2
                test_data[row][col] = mid  # 设置当前列的值
                
                if constraint_func(test_data) <= target and remaining_capacity_constraint(test_data, fixed_rows + [row]):
                    best_valid = mid
                    low = mid + 1  # 尝试更大的值，但不超过original_value
                else:
                    high = mid - 1
            
            # 确保上界不超过原始值
            return min(best_valid, original_value)

        def maximize_row_with_constraint(data, row, constraint_func, target, objective_func, fixed_rows):
            """带约束的行最大化（DFS+上界优化）"""
            fixed_positions = np.where(data[row] == 1)[0]
            columns_to_process = [j for j in range(len(data[row])) if j not in fixed_positions]
            columns_to_process.sort(key=lambda j: data[row][j], reverse=True)
            best_data = data.copy()
            best_objective = 0
            iterations = [0]

            def bfs(current_data, col_index, columns_processed):
                iterations[0] += 1
                nonlocal best_data, best_objective
                
                if constraint_func(current_data) <= target and remaining_capacity_constraint(current_data, fixed_rows + [row]):
                    current_objective = objective_func(current_data)
                    if current_objective > best_objective:
                        best_data = current_data.copy()
                        best_objective = current_objective
                    return True

                if col_index >= len(columns_to_process):
                    return False

                actual_col = columns_to_process[col_index]
                original_value = data[row][actual_col]
                columns_processed_current = columns_processed.union({actual_col})
                
                # 计算上界并限制不超过原始值
                upper_bound = compute_column_upper_bound(
                    current_data, row, actual_col, fixed_rows, columns_processed, constraint_func, target
                )
                
                for value in range(upper_bound, 0, -1):
                    new_data = current_data.copy()
                    new_data[row][actual_col] = value
                    if bfs(new_data, col_index + 1, columns_processed_current):
                        return True  # 找到最优解后提前终止
            
            bfs(data.copy(), 0, set())
            return best_data
        
        # 固定行优化流程
        fixed_rows = []
        data = np.ceil(solution).astype(int)
        
        # 空间层次优化
        for spatial_name in self.spatial_level:
            sp_level = self.spatial_level[spatial_name]
            spatial_capacity = self.spatial_size_list[spatial_name]
            data = maximize_row_with_constraint(
                data, sp_level, 
                lambda x: np.prod(x[sp_level]), spatial_capacity, 
                lambda x: np.prod(x[sp_level]), fixed_rows
            )
            fixed_rows.append(sp_level)
        
        # 存储层次优化
        def create_buffer_constraint(buffer_name):
            return lambda x: sum(
                np.prod(x[self.temporal_level[buffer_name]][self.tensor_dimensions[tensor]]) 
                for tensor in self.buffer_tensor_dict[buffer_name]
            )
                
        for buffer_key in sorted(self.buffer_name_list.keys()):
            buffer_name = self.buffer_name_list[buffer_key]
            if buffer_name == 'DRAM':
                continue
            buffer_level = self.temporal_level[buffer_name]
            buffer_capacity = self.buffer_size_list[buffer_key]
            constraint_func = create_buffer_constraint(buffer_name)
            data = maximize_row_with_constraint(
                data, buffer_level, constraint_func, buffer_capacity, constraint_func, fixed_rows
            )
            fixed_rows.append(buffer_level)
        
        # 计算最后一行
        def compute_last_row(data):
            product = np.prod(data[:-1], axis=0, dtype=np.float64)
            last_row = np.ceil(np.array(self.dimension) / product).astype(int)
            return np.vstack([data[:-1], np.maximum(last_row, 1)])
        
        return compute_last_row(data)

    def _integerize_with_staged_optimization(self, solution, p):
        def calculate_remaining_capacity(data, fixed_rows):
            """计算存储层次剩余容量"""
            result = {}
            for buffer_key in sorted(self.buffer_name_list.keys()):
                buffer_name = self.buffer_name_list[buffer_key]
                buffer_level = self.temporal_level[buffer_name]
                if buffer_name == 'DRAM':
                    continue
                all_involved_rows = list(range(buffer_level + 1))
                used_capacity = 0
                for tensor in self.buffer_tensor_dict[buffer_name]:
                    tensor_capacity = 1
                    for l in all_involved_rows:
                        for dim in self.tensor_dimensions[tensor]:
                            tensor_capacity *= data[l][dim] if l in fixed_rows else 1
                    used_capacity += tensor_capacity
                result[buffer_name] = used_capacity
            return result

        def remaining_capacity_constraint(data, fixed_rows):
            """检查容量约束"""
            capacities = calculate_remaining_capacity(data, fixed_rows)
            for buffer_key in sorted(self.buffer_name_list.keys()):
                buffer_name = self.buffer_name_list[buffer_key]
                if buffer_name == 'DRAM':
                    continue
                if capacities[buffer_name] > self.buffer_size_list[buffer_key]:
                    return False
            return True

        def compute_column_upper_bound(data, row, col, fixed_rows, columns_processed, constraint_func, target):
            """计算列的有效上界（不超过原始值，且未处理列设为1）"""
            original_value = data[row][col]
            test_data = data.copy()
            
            # 未处理的列设为1（包括当前列未处理时，但当前列会被单独设置）
            for c in range(len(data[row])):
                if c != col and c not in columns_processed:
                    test_data[row][c] = 1
            
            # low, high = 1, math.floor(original_value * 1.1)
            low, high = 1, original_value
            best_valid = 1
            
            while low <= high:
                mid = (low + high) // 2
                test_data[row][col] = mid  # 设置当前列的值
                
                if constraint_func(test_data) <= target and remaining_capacity_constraint(test_data, fixed_rows + [row]):
                    best_valid = mid
                    low = mid + 1  # 尝试更大的值，但不超过original_value
                else:
                    high = mid - 1
            
            # 确保上界不超过原始值
            return min(best_valid, original_value)

        def maximize_row_with_constraint(data, row, constraint_func, target, fixed_rows, p_row):
            """带约束的行最大化（DFS+上界优化）"""
            fixed_positions = np.where(data[row] == 1)[0]
            columns_to_process = [j for j in range(len(data[row])) if j not in fixed_positions]
            columns_to_process.sort(key=lambda j: data[row][j], reverse=True)
            best_data = data.copy()
            best_objective = 0
            iterations = [0]

            def bfs(current_data, col_index, columns_processed):
                iterations[0] += 1
                nonlocal best_data, best_objective
                
                if constraint_func(current_data) <= target and remaining_capacity_constraint(current_data, fixed_rows + [row]):
                    current_objective = np.sum(p_row * current_data[row])
                    if current_objective > best_objective:
                        best_data = current_data.copy()
                        best_objective = current_objective
                    return True

                if col_index >= len(columns_to_process):
                    return False

                actual_col = columns_to_process[col_index]
                original_value = data[row][actual_col]
                columns_processed_current = columns_processed.union({actual_col})
                
                # 计算上界并限制不超过原始值
                upper_bound = compute_column_upper_bound(
                    current_data, row, actual_col, fixed_rows, columns_processed, constraint_func, target
                )
                
                for value in range(upper_bound, 0, -1):
                    new_data = current_data.copy()
                    new_data[row][actual_col] = value
                    if bfs(new_data, col_index + 1, columns_processed_current):
                        return True  # 找到最优解后提前终止
            
            bfs(data.copy(), 0, set())
            return best_data
        
        # 固定行优化流程
        fixed_rows = []
        data = np.ceil(solution).astype(int)
        
        # 空间层次优化
        for spatial_name in self.spatial_level:
            sp_level = self.spatial_level[spatial_name]
            spatial_capacity = self.spatial_size_list[spatial_name]
            p_row = p[sp_level]  # 获取当前行的权重
            data = maximize_row_with_constraint(
                data, sp_level, 
                lambda x: np.prod(x[sp_level]), 
                spatial_capacity, 
                fixed_rows,
                p_row
            )
            fixed_rows.append(sp_level)
        
        # 存储层次优化
        def create_buffer_constraint(buffer_name):
            return lambda x: sum(
                np.prod(x[self.temporal_level[buffer_name]][self.tensor_dimensions[tensor]]) 
                for tensor in self.buffer_tensor_dict[buffer_name]
            )
                
        for buffer_key in sorted(self.buffer_name_list.keys()):
            buffer_name = self.buffer_name_list[buffer_key]
            if buffer_name == 'DRAM':
                continue
            buffer_level = self.temporal_level[buffer_name]
            buffer_capacity = self.buffer_size_list[buffer_key]
            p_row = p[buffer_level]  # 获取当前行的权重
            constraint_func = create_buffer_constraint(buffer_name)
            data = maximize_row_with_constraint(
                data, buffer_level, 
                constraint_func, 
                buffer_capacity, 
                fixed_rows,
                p_row
            )
            fixed_rows.append(buffer_level)
        
        # 计算最后一行
        def compute_last_row(data):
            product = np.prod(data[:-1], axis=0, dtype=np.float64)
            last_row = np.ceil(np.array(self.dimension) / product).astype(int)
            return np.vstack([data[:-1], np.maximum(last_row, 1)])
        
        return compute_last_row(data)

    def create_genome_for_parameters(self, num_population):
        para_list = []
        for i in range(num_population):
            if self.para_dim == 1:
                # p = []
                # for i in range(len(self.temporal_level) + len(self.spatial_level)):
                #     p.append(random.randint(0, 10))
                # para_list.append(p)
                row = len(self.temporal_level) + len(self.spatial_level) - 1
                para_list.append(np.random.randint(1, 11, size=row).astype(float))
            elif self.para_dim == 2:
                row = len(self.temporal_level) + len(self.spatial_level)
                col = 8
                # para_list.append(np.random.randint(1, 11, size=(row, col)).astype(float))
                weight_mat = np.random.uniform(low=1.0, high=100.0, size=(row, col))
                order_mat = np.stack([np.random.permutation(8) for _ in range(row)])
                para_list.append(np.vstack([weight_mat, order_mat]))
            else:
                print("无效的参数维度，请选择 1 或 2。")

        return para_list

    def judge(self):
        try:
            values = []
            for term in self.fitness_obj:
                if term == "cycles":
                    reward = -self.observation['cycles']
                elif term == "energy":
                    reward = -self.observation['energy']
                elif term == "utilization":
                    reward = self.observation['utilization']
                elif term == "EDP":
                    reward = -self.observation['EDP']
                elif term == "GFLOPs":
                    reward = self.observation['GFLOPs']
                else:
                    raise NameError('Undefined fitness type')
                values.append(reward)
            return values
        except:
            return None
    
    
    def thread_fun_p(self, p):
        original_dir = os.getcwd()
        # 创建一个唯一的临时文件夹
        temp_dir = f'{original_dir}/mars_tmp/temp_{uuid.uuid4()}'
        os.makedirs(temp_dir)
        try:
            mapping, _ = self.generate_mapping(self.dimension, p)
            mapping = Mapping(mapping)
        # try:
            remainders = {}
            outermost_idx = {}
            for d in mapping.factor_dict.keys():
                T = self.dimension_dict[d]
                F = mapping.factor_dict[d]
                remainders[d], outermost_idx[d] = utils.find_remainders(F[::-1], T)
                
            
            temp_mapping = utils.generate_mapping(mapping.factor_dict, mapping.permutation_list, mapping.target_list, mapping.type_list, mapping.bypass_list, remainders, outermost_idx)
            
            output_stats_path = os.path.join(temp_dir, 'timeloop-model.stats.txt')
            utils.store_yaml(f'{temp_dir}/temp.yaml', temp_mapping)
            utils.store_yaml(f'{temp_dir}/temp_prob.yaml', self.problem.problem)
            utils.store_yaml(f'{temp_dir}/temp_arch.yaml', self.accelerator.arch_dict)
            utils.run_timeloop(f'{temp_dir}/temp_arch.yaml', f'{temp_dir}/temp_prob.yaml', f'{temp_dir}/temp.yaml', cwd=temp_dir)
            try:
                self.observation = utils.parse_timeloop_output(output_stats_path)
            except:
                pass
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        return self.judge()

    def run_parameters(self, stage_idx=0, prev_stage_value=0, num_population=10, num_generations=100, elite_ratio=0.05,
                       parents_ratio=0.15, ratio_decay=1, num_finetune=1):
        row = len(self.temporal_level) + len(self.spatial_level)
        col = 8

        def crossover_tile(parents, pop, row, alpha=0.5):
            if len(parents) ==1:
                for idx in range(len(pop)):
                    pop[idx] = copy.deepcopy(parents[0])
            else:
                for idx in range(0, len(pop), 2):
                    dad = copy.deepcopy(parents[random.randint(0, len(parents)-1)])
                    mom = copy.deepcopy(parents[random.randint(0, len(parents)-1)])
                    
                    # 随机选择交叉策略
                    crossover_type = random.choice(['row_column', 'multi_point', 'arithmetic'])
                    
                    if crossover_type == 'row_column':
                        # 行/列交叉（原始实现）
                        if random.random() < alpha:
                            if dad.ndim == 2 and random.random() < 0.5:
                                # 行交叉
                                row_cutoff = random.randint(1, min(row, dad.shape[0])-1)
                                dad[row_cutoff:row], mom[row_cutoff:row] = mom[row_cutoff:row], dad[row_cutoff:row]
                            else:
                                # 列交叉
                                if dad.ndim == 1:
                                    col_cutoff = random.randint(1, len(dad)-1)
                                else:
                                    col_cutoff = random.randint(1, dad.shape[1]-1)
                                dad[:row, col_cutoff:], mom[:row, col_cutoff:] = mom[:row, col_cutoff:], dad[:row, col_cutoff:]
                    
                    elif crossover_type == 'multi_point':
                        # 多点交叉
                        if dad.ndim == 2:
                            cols = dad.shape[1]
                            for i in range(min(row, dad.shape[0])):
                                if random.random() < alpha:
                                    # 随机选择多个交叉点
                                    num_points = random.randint(1, cols//2)
                                    points = sorted(random.sample(range(1, cols), num_points))
                                    for j in range(len(points)):
                                        if j % 2 == 0:  # 交替交换片段
                                            start = points[j]
                                            end = points[j+1] if j+1 < len(points) else cols
                                            dad[i, start:end], mom[i, start:end] = mom[i, start:end], dad[i, start:end]
                        else:
                            if random.random() < alpha:
                                num_points = random.randint(1, len(dad)//2)
                                points = sorted(random.sample(range(1, len(dad)), num_points))
                                for j in range(len(points)):
                                    if j % 2 == 0:
                                        start = points[j]
                                        end = points[j+1] if j+1 < len(points) else len(dad)
                                        dad[start:end], mom[start:end] = mom[start:end], dad[start:end]
                    
                    elif crossover_type == 'arithmetic':
                        # 算术交叉：生成介于父代之间的子代
                        if dad.ndim == 2:
                            cols = dad.shape[1]
                            for i in range(min(row, dad.shape[0])):
                                for j in range(cols):
                                    if random.random() < alpha:
                                        beta = random.uniform(0.3, 0.7)  # 控制混合比例
                                        tmp = beta * dad[i, j] + (1-beta) * mom[i, j]
                                        mom[i, j] = beta * mom[i, j] + (1-beta) * dad[i, j]
                                        dad[i, j] = tmp
                        else:
                            for i in range(len(dad)):
                                if random.random() < alpha:
                                    beta = random.uniform(0.3, 0.7)
                                    tmp = beta * dad[i] + (1-beta) * mom[i]
                                    mom[i] = beta * mom[i] + (1-beta) * dad[i]
                                    dad[i] = tmp
                    
                    # 后row行交叉（仅行交叉）
                    if random.random() < alpha:
                        if dad.ndim == 2 and row < dad.shape[0]:  # 确保有后row行
                            # 行交叉（后row行）
                            start_row = row
                            end_row = min(2*row, dad.shape[0])
                            if end_row > start_row + 1:  # 确保有足够的行进行交叉
                                row_cutoff = random.randint(start_row + 1, end_row - 1)
                                dad[row_cutoff:end_row], mom[row_cutoff:end_row] = mom[row_cutoff:end_row], dad[row_cutoff:end_row]
            

                    pop[idx] = dad
                    if idx + 1 < len(pop):
                        pop[idx+1] = mom

        
        def shuffle_order(arr, alpha=0.5):
            if arr.ndim == 1:
                # 一维数组处理
                if random.random() < alpha:
                    np.random.shuffle(arr)

            elif arr.ndim == 2:
                # 二维数组处理
                for row in arr:
                    if random.random() < alpha:
                        np.random.shuffle(row)

            else:
                raise ValueError("输入数组必须是一维或二维")
            
        def mutate_factor(arr, row, alpha=0.5, generation=0, max_generations=100):
            """带自适应幅度的因子变异"""
            # 变异幅度随代数衰减（早期探索，后期开发）
            decay_factor = 1.0 - (generation / max_generations) * 0.7
            
            if arr.ndim == 2:
                rows, cols = arr.shape
                # 仅处理前row行
                for i in range(min(row, rows)):
                    for j in range(cols):
                        if random.random() < alpha:
                            # 随机选择变异方向（增大或减小）
                            if random.random() < 0.5:
                                # 增大因子（但幅度随迭代减小）
                                arr[i, j] *= (1 + 0.5 * decay_factor)
                            else:
                                # 减小因子
                                arr[i, j] *= (1 - 0.3 * decay_factor)
            else:
                for i in range(len(arr)):
                    if random.random() < alpha:
                        if random.random() < 0.5:
                            arr[i] *= (1 + 0.5 * decay_factor)
                        else:
                            arr[i] *= (1 - 0.3 * decay_factor)
            
            return arr
 
        def gaussian_mutation(arr, row, alpha=0.5, generation=0, max_generations=100, sigma_base=0.2):
            """
            带自适应步长的高斯变异算子
            
            参数:
            - arr: 待变异的参数矩阵 (numpy array)
            - alpha: 变异概率 (0.0-1.0)
            - generation: 当前迭代代数
            - max_generations: 总迭代代数
            - sigma_base: 基础标准差，控制变异步长
            
            返回:
            - 变异后的参数矩阵
            """
            # 自适应标准差：随迭代进行减小步长（早期探索，后期开发）
            decay_factor = 1.0 - (generation / max_generations) * 0.7
            sigma = sigma_base * decay_factor
            
            if arr.ndim == 2:
                # 只遍历前row行（使用min确保不越界）
                for i in range(min(row, arr.shape[0])):
                    for j in range(arr.shape[1]):
                        if random.random() < alpha:
                            arr[i, j] += np.random.normal(0, sigma)
                            # 可选：限制参数范围
                            # arr[i, j] = np.clip(arr[i, j], 0, 1)
            else:  # 一维数组处理
                for i in range(len(arr)):
                    if random.random() < alpha:
                        arr[i] += np.random.normal(0, sigma)
                        # arr[i] = np.clip(arr[i], 0, 1)
            
            return arr

        def reverse_sign(arr, alpha=0.5):
            if arr.ndim == 1:
                # 一维数组处理
                for i in range(len(arr)):
                    if random.random() < alpha:  # 以alpha的概率反转符号
                        arr[i] *= -1  # 乘以-1实现符号反转
                return arr
            elif arr.ndim == 2:
                # 二维数组处理
                rows, cols = arr.shape
                for i in range(rows):
                    for j in range(cols):
                        if random.random() < alpha:  # 以alpha的概率反转符号
                            arr[i, j] *= -1  # 乘以-1实现符号反转
                return arr
            else:
                raise ValueError("输入数组必须是一维或二维")
        
        def shuffle_row(arr, row, alpha=0.5):
            total_rows = arr.shape[0]
        
            # 打乱前row行（如果存在）
            if row > 1 and row <= total_rows:
                front_indices = list(range(row))
                random.shuffle(front_indices)
                arr[:row] = arr[front_indices]
            
            # 打乱后面的行（如果存在且行数>1）
            remaining_rows = total_rows - row
            if remaining_rows > 1 and row < total_rows:
                back_start = row
                back_indices = list(range(back_start, total_rows))
                random.shuffle(back_indices)
                arr[back_start:total_rows] = arr[back_indices]
        
        def shuffle_col(arr, alpha=0.5):
            if random.random() < alpha:
                # print(arr.shape)
                cols = list(range(arr.shape[1]))
                # print(cols)
                random.shuffle(cols)
                arr[:] = arr[:, cols]

        def adaptive_crossover_rate(generation, max_generations, initial_rate=0.8):
            """自适应交叉率：随迭代减少探索"""
            progress = generation / max_generations
            return initial_rate * (1 - progress * 0.4)  # 从0.8降至0.4

        def adaptive_mutation_rate(generation, max_generations, initial_rate=0.5):
            """自适应变异率：随迭代增加开发"""
            progress = generation / max_generations
            return initial_rate * (0.5 + progress * 0.5)  # 从0.5降至0.25


        def adaptive_parameters(g, max_generations, initial_alpha=0.5):
            """根据迭代代数自适应调整参数"""
            # 在进化后期减少探索，增加开发
            progress = g / max_generations
            alpha = initial_alpha * (1 - progress * 0.5)  # 从0.5逐渐降低到0.25
            return alpha

        def check_convergence1(best_reward_list, window_size=10, threshold=1e-3):
            """检查是否收敛"""
            if len(best_reward_list) < window_size * 2:
                return False
            
            recent = np.array(best_reward_list[-window_size:])
            older = np.array(best_reward_list[-window_size*2:-window_size])
            
            # 计算相对变化
            relative_change = np.abs((np.mean(recent) - np.mean(older)) / np.mean(older))
            
            return relative_change < threshold

        def calculate_population_diversity(pop):
            """计算种群多样性（基于欧氏距离）"""
            if len(pop) <= 1:
                return 0
            diversity = 0
            for i in range(len(pop)):
                for j in range(i+1, len(pop)):
                    diversity += np.linalg.norm(pop[i].flatten() - pop[j].flatten())
            return diversity / (len(pop) * (len(pop) - 1) / 2)
        
        def check_convergence(best_reward_list, window_size=10, threshold=1e-3):
            """改进的收敛检测"""
            if len(best_reward_list) < window_size * 2:
                return False
            recent = np.array([r[0] for r in best_reward_list[-window_size:]])
            older = np.array([r[0] for r in best_reward_list[-window_size*2:-window_size]])
            improvement = (np.mean(recent) - np.mean(older)) / (abs(np.mean(older)) + 1e-10)
            return improvement < threshold
        
        def simulated_annealing(solution, evaluate_func, para_dim, max_iter=100, initial_temp=100.0, cooling_rate=0.95):
            """
            模拟退火局部搜索函数
            
            参数:
            - solution: 当前解（参数矩阵）
            - evaluate_func: 评估函数，返回适应度值（单值，如stage_idx对应的奖励）
            - para_dim: 参数矩阵维度（1或2）
            - max_iter: 最大迭代次数
            - initial_temp: 初始温度
            - cooling_rate: 降温速率
            
            返回:
            - 优化后的解
            """
            current_sol = solution.copy()
            current_fitness = evaluate_func(current_sol)
            best_sol = current_sol.copy()
            best_fitness = current_fitness
            
            temp = initial_temp
            
            for t in range(max_iter):
                # 生成邻域解（根据维度选择变异方式）
                if para_dim == 1:
                    neighbor_sol = current_sol.copy()
                    idx = np.random.randint(len(neighbor_sol))
                    neighbor_sol[idx] += np.random.normal(0, 0.1)  # 高斯变异生成邻域解
                    neighbor_sol[idx] = np.clip(neighbor_sol[idx], 0.0, 1.0)  # 限制参数范围
                else:
                    neighbor_sol = current_sol.copy()
                    i, j = np.random.randint(0, neighbor_sol.shape[0]), np.random.randint(0, neighbor_sol.shape[1])
                    neighbor_sol[i, j] += np.random.normal(0, 0.1)
                    neighbor_sol[i, j] = np.clip(neighbor_sol[i, j], 0.0, 1.0)
                
                # 评估邻域解
                neighbor_fitness = evaluate_func(neighbor_sol)
                
                # 计算接受概率
                delta = neighbor_fitness - current_fitness
                if delta > 0 or np.random.rand() < math.exp(delta / temp):
                    current_sol = neighbor_sol
                    current_fitness = neighbor_fitness
                    
                    # 更新最优解
                    if current_fitness > best_fitness:
                        best_sol = current_sol.copy()
                        best_fitness = current_fitness
                
                # 降温
                temp *= cooling_rate
            
            return best_sol

        
        def evaluate_solution(solution):
            """辅助评估函数：返回当前解的适应度（单值）"""
            reward = self.thread_fun_p(solution)
            if reward is None or any(np.array(reward) >= 0):
                return -float("Inf")
            if stage_idx > 0 and any([reward[kk] < prev_stage_value[kk] for kk in range(len(prev_stage_value))]):
                return -float("Inf")
            return reward[stage_idx]

        num_generations = num_generations
        num_population = num_population
        num_elite = int(num_population * elite_ratio)
        pool = Pool(min(num_population + num_elite, cpu_count()))
        best_reward_list = []
        best_reward = [-float("Inf") for _ in range( len(self.fitness_obj))]
        best_sol = None

        population = self.create_genome_for_parameters(num_population)
        # mapping, _ = self.generate_mapping(self.dimension, population[0])
        # exit()
        fitness = np.ones((num_population, len(self.fitness_obj)), float)
        num_parents = num_population
        for g in range(num_generations):
            # alpha = adaptive_parameters(g, num_generations)
            
            # 计算自适应参数
            diversity = calculate_population_diversity(population)
            crossover_alpha = adaptive_crossover_rate(g, num_generations)
            mutation_alpha = adaptive_mutation_rate(g, num_generations)

            finetine_iter = 1 if g < num_generations // 2 else num_finetune
            for f in range(finetine_iter):
                gen_best = -float("Inf")
                gen_best_idx = 0
                count_non_valid = 0
                if num_parents < 1:  # restart
                    print("restart!")
                    return 
                population, fitness, parents = self.select_parents(population, fitness, num_parents, num_population,
                                                                  stage_idx, first_stage_value=prev_stage_value)
                elite = copy.deepcopy(parents[:num_elite])
                elite_fitness = copy.deepcopy(fitness[:(len(elite))])

                crossover_tile(parents, population, row, alpha=crossover_alpha)
             
                # 变异
                for pop in population[num_elite:]:
                    # 选择一种主要变异策略
                    if random.random() < 0.7:  # 70%概率使用高斯变异
                        gaussian_mutation(pop, row, alpha=mutation_alpha, generation=g, max_generations=num_generations)
                    else:  # 30%概率使用因子变异
                        mutate_factor(pop, row, alpha=mutation_alpha, generation=g, max_generations=num_generations)
                    shuffle_order(pop, alpha=0.1)
                    # reverse_sign(pop, alpha=0.2)
                    if self.para_dim == 2:
                        shuffle_row(pop, row, alpha=0.1)
                        shuffle_col(pop, alpha=0.1)

                # 1. 更新种群（参数矩阵）
                population = elite + population[num_elite:]  # 合并精英和变异后的个体

                # 2. 更新适应度数组
                # 假设elite_fitness的长度是num_elite，population的长度是num_population
                # 需要确保新的fitness数组与population长度一致
                fitness = np.zeros((num_population, len(self.fitness_obj)))
                fitness[:num_elite] = elite_fitness  # 前num_elite个位置填充精英的适应度

                reward_list = pool.map(self.thread_fun_p, population)
                for i in range(len(population)):
                    reward =reward_list[i]
                    if reward is None or any(np.array(reward) >= 0):
                        reward = [float("-Inf") for _ in range(len(best_reward))]
                        count_non_valid += 1
                    elif stage_idx > 0:
                        if any([reward[kk] < prev_stage_value[kk] for kk in range(len(prev_stage_value))]):
                            reward = [float("-Inf") for _ in range(len(best_reward))]
                            count_non_valid += 1
                    judging_reward = reward[stage_idx]
                    fitness[i] = reward
                    if gen_best < judging_reward:
                        gen_best = judging_reward
                        gen_best_idx = i
                judging_best_reward = best_reward[stage_idx]
                if judging_best_reward < gen_best:
                    best_reward = copy.deepcopy(fitness[gen_best_idx])
                    best_sol = copy.deepcopy(population[gen_best_idx])

                num_parents = int(num_population * parents_ratio)
                num_parents = min(num_parents, len(population) - count_non_valid)
                parents_ratio *= ratio_decay
                best_reward_list.append(best_reward)
                chkpt = {
                    "best_reward": best_reward,
                    "best_reward_list": best_reward_list,
                    "best_sol": best_sol,
                    "num_population": num_population,
                    "num_generations": num_generations,
                    "fitness_use": self.fitness_obj
                }
                
                print( "[Stage {}]Gen {}:  1st stage Reward: {}, Best reward: {}".format(stage_idx + 1, (g + 1), np.abs(prev_stage_value), np.abs(best_reward)))
                # 检查收敛
                if check_convergence(best_reward_list):
                    print("Converged at generation", g)
                    break
            
            # # 在迭代后期（如后半程）应用模拟退火
            # if g >= num_generations // 2:
            #     # 对当前最优解进行模拟退火优化
            #     if best_sol is not None:
            #         print(f"Generation {g}: 应用模拟退火优化最优解")
            #         best_sol = simulated_annealing(
            #             best_sol,
            #             evaluate_solution,
            #             para_dim=self.para_dim,
            #             max_iter=50,
            #             initial_temp=50.0,
            #             cooling_rate=0.98
            #         )
            #         # 重新评估优化后的解
            #         best_reward = [evaluate_solution(best_sol)] + [-float("Inf")] * (len(self.fitness_obj)-1)
            #         best_reward[stage_idx] = evaluate_solution(best_sol)
            
            # 每5代检查并恢复多样性
            if g % 5 == 0 and diversity < 0.2:
                # print(f"Generation {g}: 种群多样性低，注入新个体")
                replace_count = max(5, int(num_population * 0.1))
                for i in range(replace_count):
                    worst_idx = np.argmin(fitness[:, stage_idx])
                    population[worst_idx] = self.create_genome_for_parameters(1)[0]
                    fitness[worst_idx] = [float("-Inf")] * len(best_reward)

        pool.close()

        remainders = {}
        outermost_idx = {}
        # print(best_sol)
        best_map, _ = self.generate_mapping(self.dimension, best_sol)
        best_map = Mapping(best_map)
        for d in best_map.factor_dict.keys():
            T = self.dimension_dict[d]
            F = best_map.factor_dict[d]
            remainders[d], outermost_idx[d] = utils.find_remainders(F[::-1], T)
        best_mapping = utils.generate_mapping(best_map.factor_dict, best_map.permutation_list, best_map.target_list, best_map.type_list, best_map.bypass_list, remainders, outermost_idx)
        map_path = f'{self.report_dir}/map.yaml'
        utils.store_yaml(map_path, best_mapping)
        prob_path = f'{self.report_dir}/problem.yaml'
        utils.store_yaml(prob_path, self.problem.problem)
        arch_path = f'{self.report_dir}/arch.yaml'
        utils.store_yaml(arch_path, self.accelerator.arch_dict)

        # 如果要使用cwd, 文件路径要么是绝对路径要么是cwd的相对路径
        utils.run_timeloop('arch.yaml', 'problem.yaml', 'map.yaml', cwd=self.report_dir)

        return best_map        

