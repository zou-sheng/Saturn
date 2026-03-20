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
    def __init__(self, operator_instance, accelerator_dir, accelerator, mapper, version, report_dir):
        self.accl_name = accelerator
        self.fitness_obj = ['EDP']
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
        self.mapper = mapper

        arch_path = accelerator_dir
        self.accelerator = Arch(arch_path, accelerator, version)
        mapspace_path = pathlib.Path('{}/{}/{}/{}.yaml'.format(accelerator_dir, mapper, accelerator, 'mapspace')).resolve()
        
        print("mapspace_path:", mapspace_path)  # 打印路径
        print("路径是否存在：", os.path.exists(mapspace_path))  # 检查路径是否存在
        print("路径是否是文件：", os.path.isfile(mapspace_path))  # 检查是否是文件
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



        self.dimension, self.dimension_dict = self.problem.get_problem_info()


       
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

        if self.accl_name == 'Simba':
            self.buffer_hierarchy = ['Registers', 'AccumulationBuffer', 'WeightBuffer', 'InputBuffer', 'GlobalBuffer', 'DRAM']
            self.tensor_in_buffer = {'Registers': ['Weights'], 'AccumulationBuffer': ['Outputs'], 'WeightBuffer': ['Weights'], 'InputBuffer': ['Inputs'], 'GlobalBuffer': ['Inputs','Outputs'], 'DRAM': ['Weights','Inputs','Outputs']}
            # self.buffer_capacity = buffer_capacity
            # self.spatial_capacity = spatial_capacity
        else:
            print("未定义")
            exit()
  
    def generate_mapping(self, dimension, p):
        sol = self.lpsolver_all_DNN(dimension, p)
        mapping_dict, dimension_dict = utils.generate_mapping_for_lpsolver2(sol, self.targets, self.type, self.bypass)
        
        prob = copy.deepcopy(self.problem.problem)       
        for key in dimension_dict.keys():
            prob['problem']['instance'][key] = dimension_dict[key]
        return mapping_dict, prob

    def lpsolver_all_DNN(self, dimension_list, p):
        if self.accl_name == "Simba":
            rows = 6
        else:
            print("未定义")
            exit()
        cols = 8 # 问题维度R, S, P, Q, C, K, H, N

        # 创建一个最大化问题
        prob = LpProblem("Matrix_Mapping_Problem", LpMaximize)

        # 创建矩阵变量，每个元素是一个非负的连续变量
        spatial_tiles = [[[LpVariable(f"st_{i}_{j}_{k}", lowBound=0) for j in range(cols)] for i in range(rows)] for k in range(len(dimension_list))]
        temporal_tiles = [[[LpVariable(f"tt_{i}_{j}_{k}", lowBound=0) for j in range(cols)] for i in range(rows)] for k in range(len(dimension_list))]

        temporal_tiles_weight = p[:rows]
        spatial_tiles_weight = p[rows:]
        
        # 定义目标函数：矩阵元素的加权和
        objective = 0
        for k in range(len(dimension_list)):
            for r in range(rows):
                for c in range(cols):
                    objective += spatial_tiles[k][r][c] * spatial_tiles_weight[k][r][c]

        for k in range(len(dimension_list)):
            for r in range(rows):
                for c in range(cols):
                    objective += temporal_tiles[k][r][c] * temporal_tiles_weight[k][r][c]

        prob += objective
        
        # 除DRAM外每一个存储层次的存储容量约束
        for k in range(len(dimension_list)):
            # 去掉DRAM
            for r in range(len(self.buffer_hierarchy)-1):
                buf_capacity = 0
                tensors_list = self.tensor_in_buffer[self.buffer_hierarchy[r]]
                for i in range(r+1):
                    for tensor in tensors_list:
                        for dim in self.tensor_dimensions[tensor]:
                            buf_capacity += temporal_tiles[k][i][dim]
                            buf_capacity += spatial_tiles[k][i][dim]
                
                prob += buf_capacity <= len(tensors_list)*(self.buffer_capacity[r] - math.log2(len(tensors_list)))

            
        # 并行容量约束
        for k in range(len(dimension_list)):
            for r in range(rows):
                sp_capacity = 0
                for c in range(cols):
                    sp_capacity += spatial_tiles[k][r][c]
                prob += sp_capacity <= self.spatial_capacity[r]

        # 维度约束
        for k in range(len(dimension_list)):
            for c in range(cols):
                temporal_sum = sum(temporal_tiles[k][r][c] for r in range(rows))
                spatial_sum = sum(spatial_tiles[k][r][c] for r in range(rows))
                col_sum = temporal_sum + spatial_sum
                for r in range(rows):
                    prob += temporal_tiles[k][r][c] <= math.log2(dimension_list[k][c])
                    prob += spatial_tiles[k][r][c] <= math.log2(dimension_list[k][c])
                prob += col_sum == math.log2(dimension_list[k][c])

        prob.solve(PULP_CBC_CMD(msg=0))
        # print(prob)
        
        spatial_tiles_solution = [[[2**spatial_tiles[k][i][j].value() for j in range(cols)] for i in range(rows)] for k in range(len(dimension_list))]
        temporal_tiles_solution = [[[2**temporal_tiles[k][i][j].value() for j in range(cols)] for i in range(rows)] for k in range(len(dimension_list))]
        
        solutions = [spatial_tiles_solution, temporal_tiles_solution]
        # print(solution)
        # print("time1: ", time.time()-start_time)
        # 映射优先的调整
        solutions, config = self._integerize_with_staged_optimization_with_config_all_DNN(solutions, p)
        # print("time2: ", time.time()-start_time)
        
        return solutions, config
    
    def _integerize_with_staged_optimization_with_config_all_DNN(self, solutions, p):
        
        pass


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
            row = len(self.temporal_level) + len(self.spatial_level)
            col = 8
            para_list.append(np.random.uniform(low=1.0, high=100.0, size=(row, col)))


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

    def select_parents(self, pop, fitness, num_parents, num_population, stage_idx=0, first_stage_value=None):
        # 选择在当前代中表现最好的个体作为下一代后代的父母
        if stage_idx==0:
            idx = np.argsort(fitness[:,0])[::-1]
            new_pop = [pop[i] for i in idx][:num_population]
            new_fitness = fitness[idx][:num_population]
            parents = copy.deepcopy(new_pop[:num_parents])
        else:
            elite_pop = [pop[i] for i in range(len(pop)) if all([fitness[i][kk]>=first_stage_value[kk] for kk in range(len(first_stage_value))])]
            elite_fitness = fitness
            for kk in range(len(first_stage_value)):
                elite_fitness = elite_fitness[(elite_fitness[:, kk] >= first_stage_value[kk])]
            idx = np.argsort(elite_fitness[:, stage_idx])[::-1]
            elite_pop = [elite_pop[i] for i in idx][:num_population]
            parents = copy.deepcopy(elite_pop[:num_parents])  if len(elite_pop)>0 else copy.deepcopy(pop[:num_parents])
            new_pop = pop[:num_population]
            new_fitness = fitness[:num_population]

        return new_pop, new_fitness, parents


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

