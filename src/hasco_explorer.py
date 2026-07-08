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

class HardwareConfig:
    def __init__(self, name, buffer_hierarchy, NoC, tensor_in_buffer, temporal_tile_weights, spatial_tile_weights, buffer_temporal_weights=None, buffer_spatial_weights=None, buffer_temporal_list=None, buffer_spatial_list=None):
        self.name = name
        self.buffer_hierarchy = buffer_hierarchy
        self.NoC = NoC
        self.tensor_in_buffer = tensor_in_buffer
        self.temporal_tile_weights = temporal_tile_weights
        self.spatial_tile_weights = spatial_tile_weights
        self.buffer_temporal_weights = buffer_temporal_weights
        self.buffer_spatial_weights = buffer_spatial_weights
        # 有n个
        self.buffer_temporal_orders = None
        self.buffer_spatial_orders = None
        self.buffer_capacity = buffer_temporal_list
        self.spatial_capacity = buffer_spatial_list

        
    def mutation(self, best_spatial_tile_weights, best_temporal_tile_weights, best_buffer_temporal_weights, best_buffer_spatial_weights, no_mutation, alpha=0.5, generation=0, max_generations=100):
        # self.mutate_noc()
        # self.mutate_tensor_in_buffer()
        self.mutate_temporal_tile_weights(best_temporal_tile_weights, no_mutation, alpha=alpha, generation=generation, max_generations=max_generations)
        self.mutate_spatial_tile_weights(best_spatial_tile_weights, no_mutation, alpha=alpha, generation=generation, max_generations=max_generations)
        self.mutate_buffer_temporal_weights(best_buffer_temporal_weights, alpha=alpha, generation=generation, max_generations=max_generations)
        self.mutate_buffer_spatial_weights(best_buffer_spatial_weights, alpha=alpha, generation=generation, max_generations=max_generations)
        # self.mutate_buffer_temporal_order()
        # self.mutate_buffer_spatial_order()

    def mutation_for_DNN(self, best_spatial_tile_weights, best_temporal_tile_weights, no_mutation, alpha=0.5, generation=0, max_generations=100):
        # self.mutate_noc()
        # self.mutate_tensor_in_buffer()
        self.mutate_temporal_tile_weights(best_temporal_tile_weights, no_mutation, alpha=alpha, generation=generation, max_generations=max_generations)
        self.mutate_spatial_tile_weights(best_spatial_tile_weights, no_mutation, alpha=alpha, generation=generation, max_generations=max_generations)
        # self.mutate_buffer_temporal_weights(best_buffer_temporal_weights, alpha=alpha, generation=generation, max_generations=max_generations)
        # self.mutate_buffer_spatial_weights(best_buffer_spatial_weights, alpha=alpha, generation=generation, max_generations=max_generations)
        # self.mutate_buffer_temporal_order()
        # self.mutate_buffer_spatial_order()

    def mutate_noc(self):
        pass

    def mutate_tensor_in_buffer(self):
        pass

    def mutate_temporal_tile_weights(self, best_temporal_tile_weights, no_mutation, alpha=0.5, generation=0, max_generations=100):
        layer_num = len(self.temporal_tile_weights)
        for i in range(layer_num):
            if no_mutation[i] == True:
                self.temporal_tile_weights[i] = self.temporal_tile_weights[i]
                continue
            if random.random() < 0.7:
                self.temporal_tile_weights[i] = best_temporal_tile_weights[i]
            else:
                if random.random() < 0.7:
                    self.gaussian_mutation(self.temporal_tile_weights[i], alpha=alpha, generation=generation, max_generations=max_generations)
                else:
                    self.mutate_factor(self.temporal_tile_weights[i], alpha=alpha, generation=generation, max_generations=max_generations)
                self.shuffle_order(self.temporal_tile_weights[i], alpha=alpha)

    def mutate_spatial_tile_weights(self, best_spatial_tile_weights, no_mutation, alpha=0.5, generation=0, max_generations=100):
        layer_num = len(self.spatial_tile_weights)
        for i in range(layer_num):
            if no_mutation[i] == True:
                self.spatial_tile_weights[i] = self.spatial_tile_weights[i]
                continue
            if random.random() < 0.7:
                self.spatial_tile_weights[i] = best_spatial_tile_weights[i]
            else:
                if random.random() < 0.7:
                    self.gaussian_mutation(self.spatial_tile_weights[i], alpha=alpha, generation=generation, max_generations=max_generations)
                else:
                    self.mutate_factor(self.spatial_tile_weights[i], alpha=alpha, generation=generation, max_generations=max_generations)
                self.shuffle_order(self.spatial_tile_weights[i], alpha=alpha)

    def mutate_buffer_temporal_weights(self, best_buffer_temporal_weights, alpha=0.5, generation=0, max_generations=100):
        if random.random() < 0.7:
            self.buffer_temporal_weights = random.choice(best_buffer_temporal_weights)
        else:
            self.mutate_factor(self.buffer_temporal_weights, alpha=alpha, generation=generation, max_generations=max_generations)
            self.shuffle_order(self.buffer_temporal_weights)

    def mutate_buffer_spatial_weights(self, best_buffer_spatial_weights, alpha=0.5, generation=0, max_generations=100):
        if random.random() < 0.7:
            self.buffer_spatial_weights = random.choice(best_buffer_spatial_weights)
        else:
            self.mutate_factor(self.buffer_spatial_weights, alpha=alpha, generation=generation, max_generations=max_generations)
            self.shuffle_order(self.buffer_spatial_weights)

    def mutate_buffer_temporal_order(self):
        pass

    def mutate_buffer_spatial_order(self):
        pass

    def gaussian_mutation(self, arr, alpha=0.5, generation=0, max_generations=100, sigma_base=0.2):
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
            rows, cols = arr.shape
            for i in range(rows):
                for j in range(cols):
                    if random.random() < alpha:  # 以alpha概率进行变异
                        # 添加高斯噪声
                        arr[i, j] += np.random.normal(0, sigma)
                        
                        # 可选：限制参数范围（根据实际需求调整）
                        # arr[i, j] = np.clip(arr[i, j], 0, 1)  # 例如限制在[0,1]
        else:  # 一维数组处理
            for i in range(len(arr)):
                if random.random() < alpha:
                    arr[i] += np.random.normal(0, sigma)
                    # arr[i] = np.clip(arr[i], 0, 1)
        
        return arr
    
    def mutate_factor(self, arr, alpha=0.5, generation=0, max_generations=100):
        """带自适应幅度的因子变异"""
        # 变异幅度随代数衰减（早期探索，后期开发）
        decay_factor = 1.0 - (generation / max_generations) * 0.7
        
        if arr.ndim == 2:
            rows, cols = arr.shape
            for i in range(rows):
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

    def shuffle_order(self, arr, alpha=0.5):
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
        
    def crossover_2D(self, dad, mom, best_matrix, no_mutation, alpha=0.5):
        if no_mutation:
            dad = dad
            mom = mom
        elif random.random() < 0.7:
            dad = best_matrix
            mom = best_matrix
        else:
            # 随机选择交叉策略
            crossover_type = random.choice(['row_column', 'multi_point', 'arithmetic'])
            
            if crossover_type == 'row_column':
                # 行/列交叉（原始实现）
                if random.random() < alpha:
                    if dad.ndim == 2 and random.random() < 0.5:
                        # 行交叉
                        row_cutoff = random.randint(1, dad.shape[0]-1)
                        dad[row_cutoff:], mom[row_cutoff:] = mom[row_cutoff:], dad[row_cutoff:]
                    else:
                        # 列交叉
                        if dad.ndim == 1:
                            col_cutoff = random.randint(1, len(dad)-1)
                        else:
                            col_cutoff = random.randint(1, dad.shape[1]-1)
                        dad[:, col_cutoff:], mom[:, col_cutoff:] = mom[:, col_cutoff:], dad[:, col_cutoff:]
            
            elif crossover_type == 'multi_point':
                # 多点交叉
                if dad.ndim == 2:
                    rows, cols = dad.shape
                    for i in range(rows):
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
                    rows, cols = dad.shape
                    for i in range(rows):
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

    def crossover_1D(self, dad, mom, best_array, alpha=0.5):
        if random.random() < 0.3:
            dad = random.choice(best_array)
            mom = random.choice(best_array)
        else:
            if random.random() < alpha:
                col_cutoff = random.randint(1, len(dad)-1)
                dad[col_cutoff:], mom[col_cutoff:] = mom[col_cutoff:], dad[col_cutoff:]

    def crossover(self, other, best_spatial_tile_weights, best_temporal_tile_weights, best_buffer_temporal_weights, best_buffer_spatial_weights, no_mutation, alpha=0.5):
        # NoC

        # tensor_in_buffer

        # temporal_tile_weights
        layer_num = len(self.temporal_tile_weights)
        for i in range(layer_num):
            self.crossover_2D(self.temporal_tile_weights[i], other.temporal_tile_weights[i], best_temporal_tile_weights, no_mutation[i])

        # spatial_tile_weights
        for i in range(layer_num):
            self.crossover_2D(self.spatial_tile_weights[i], other.spatial_tile_weights[i], best_spatial_tile_weights, no_mutation[i])

        # buffer_temporal_weights
        self.crossover_1D(self.buffer_temporal_weights, other.buffer_temporal_weights, best_buffer_temporal_weights)

        # buffer_spatial_weights
        self.crossover_1D(self.buffer_spatial_weights, other.buffer_spatial_weights, best_buffer_spatial_weights)

        # buffer_temporal_order

        # buffer_spatial_order

    def crossover_for_DNN(self, other, best_spatial_tile_weights, best_temporal_tile_weights, no_mutation, alpha=0.5):
        # NoC

        # tensor_in_buffer

        # temporal_tile_weights
        layer_num = len(self.temporal_tile_weights)
        for i in range(layer_num):
            self.crossover_2D(self.temporal_tile_weights[i], other.temporal_tile_weights[i], best_temporal_tile_weights, no_mutation)

        # spatial_tile_weights
        for i in range(layer_num):
            self.crossover_2D(self.spatial_tile_weights[i], other.spatial_tile_weights[i], best_spatial_tile_weights, no_mutation)

        # # buffer_temporal_weights
        # self.crossover_1D(self.buffer_temporal_weights, other.buffer_temporal_weights, best_buffer_temporal_weights)

        # # buffer_spatial_weights
        # self.crossover_1D(self.buffer_spatial_weights, other.buffer_spatial_weights, best_buffer_spatial_weights)

        # buffer_temporal_order

        # buffer_spatial_order

    def get_target(self):
        pass

    def get_tiles_type(self):
        pass

    def get_bypass(self):
        pass




class MappingExplorer:
    def __init__(self, operator_instance, buffer_capacity, spatial_capacity, accelerator, mapper, version, report_dir):
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

        arch_path = pathlib.Path('{}/{}.yaml'.format(report_dir, 'arch')).resolve()
        self.accelerator = Arch(arch_path, accelerator, version)
        mapspace_path = pathlib.Path('{}/{}.yaml'.format(report_dir, 'mapspace')).resolve()
        self.mapspace = Mapspace(mapspace_path)
        self.buffer_tensor_dict = self.mapspace.buffer_tensor_dict
        self.bypass = self.mapspace.bypass
        self.tensor_dimensions = {'Weights': [0, 1, 4, 5, 6], 'Inputs': [0, 1, 2, 3, 4, 6, 7], 'Outputs': [2, 3, 5, 6, 7]}

        # self.cost_model = Timeloop(in_config_path='./SpatialAccelerators', out_config_path=self.timeloop_out_config_path,
        #                            accelerator=accelerator, opt_obj=self.opt_obj)

        buffer_name_list, buffer_size_list, buffer_spmap_cstr, num_buffer_levels = self.accelerator.get_arch_info()

        self.buffer_name_list = buffer_name_list
        self.buffer_size_list = buffer_size_list
        self.buffer_spmap_cstr = buffer_spmap_cstr
        self.buffers_with_spmap = set([key for key, value in self.buffer_spmap_cstr.items() if value > 1])
        self.num_buffer_level = num_buffer_levels


      
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

        if self.accl_name == "Simba":
            self.buffer_hierarchy = ['Registers', 'AccumulationBuffer', 'WeightBuffer', 'InputBuffer', 'GlobalBuffer', 'DRAM']
            self.tensor_in_buffer = {'Registers': ['Weights'], 'AccumulationBuffer': ['Outputs'], 'WeightBuffer': ['Weights'], 'InputBuffer': ['Inputs'], 'GlobalBuffer': ['Inputs','Outputs'], 'DRAM': ['Weights','Inputs','Outputs']}
            self.buffer_capacity = buffer_capacity
            self.spatial_capacity = spatial_capacity
        elif self.accl_name == "Gemmini":
            self.buffer_hierarchy = ['Registers', 'Accumulator', 'Scratchpad', 'DRAM']
            self.tensor_in_buffer = {'Registers': ['Weights'], 'Accumulator': ['Outputs'], 'Scratchpad': ['Inputs', 'Weights'], 'DRAM': ['Weights','Inputs','Outputs']}
            self.buffer_capacity = buffer_capacity
            self.spatial_capacity = spatial_capacity
        else:
            print("未定义1", self.accl_name == "Simba")
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
        elif self.accl_name == "Gemmini":
            rows = 4
        else:
            print("未定义", self.accl_name)
            exit()
        cols = 8 # 问题维度R, S, P, Q, C, K, H, N

        # 创建一个最大化问题
        prob = LpProblem("Matrix_Mapping_Problem", LpMaximize)

        # 创建矩阵变量，每个元素是一个非负的连续变量
        spatial_tiles = [[[LpVariable(f"st_{i}_{j}_{k}", lowBound=0) for j in range(cols)] for i in range(rows)] for k in range(len(dimension_list))]
        temporal_tiles = [[[LpVariable(f"tt_{i}_{j}_{k}", lowBound=0) for j in range(cols)] for i in range(rows)] for k in range(len(dimension_list))]

        temporal_tiles_weight = p.spatial_tile_weights
        spatial_tiles_weight = p.temporal_tile_weights
        # 定义目标函数：矩阵元素的加权和
        objective = 0
        for k in range(len(dimension_list)):
            # prob += spatial_tiles[k][1][0] == 0
            # prob += spatial_tiles[k][1][1] == 0
            # prob += spatial_tiles[k][1][2] == 0
            # prob += spatial_tiles[k][1][3] == 0
            # prob += spatial_tiles[k][1][6] == 0
            # prob += spatial_tiles[k][1][7] == 0
            # prob += spatial_tiles[k][2][0] == 0
            # prob += spatial_tiles[k][2][1] == 0
            # prob += spatial_tiles[k][2][2] == 0
            # prob += spatial_tiles[k][2][3] == 0
            # prob += spatial_tiles[k][2][6] == 0
            # prob += spatial_tiles[k][2][7] == 0
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
                
                prob += buf_capacity <= len(tensors_list)*(math.log2(self.buffer_capacity[r]) - math.log2(len(tensors_list)))


        # 并行容量约束
        for k in range(len(dimension_list)):
            for r in range(rows):
                sp_capacity = 0
                for c in range(cols):
                    sp_capacity += spatial_tiles[k][r][c]
                prob += sp_capacity <= math.log2(self.spatial_capacity[r])

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
        
        # 计算最后一行
        def compute_last_row(dimension, spatial_tiles, temporal_tiles):

            spatial_prod = np.prod(spatial_tiles, axis=0, dtype=np.float64)
            temporal_prod = np.prod(temporal_tiles[:-1], axis=0, dtype=np.float64)
            last_row = np.ceil(np.array(dimension) / (spatial_prod*temporal_prod)).astype(int)

            return last_row
        
        # 计算最小硬件开销
        def min_hardware_config(spatial_tiles_list, temporal_tiles_list):
            """
            计算多个矩阵对应的最小硬件配置（每个约束维度取所有矩阵的最小值）
            
            参数:
                data_list: 包含多个矩阵的列表（每个元素是一个矩阵，对应原data）
            返回:
                最小硬件配置列表（每个元素对应一个约束的最小值）
            """
            # 先收集每个矩阵的硬件配置
            all_configs = []
            for k in range(len(spatial_tiles_list)):

                spatial_config = []
                buffer_config = []
                # 1. 并行容量约束
                for r in range(len(p.buffer_hierarchy)):
                    spatial_capacity = np.prod(spatial_tiles_list[k][r])
                    spatial_config.append(spatial_capacity)
                # 2. 除DRAM外的存储层次容量约束
                # 去掉DRAM
                for r in range(len(p.buffer_hierarchy)-1):
                    buffer_capacity = 0
                    tensors_list = p.tensor_in_buffer[p.buffer_hierarchy[r]]
                    for tensor in tensors_list:
                        tensor_capacity = 1
                        if tensor == 'Inputs':
                            R, S, P, Q, C, K, H, N = 1, 1, 1, 1, 1, 1, 1, 1
                            for l in range(r+1):
                                R *= spatial_tiles_list[k][l][0]
                                S *= spatial_tiles_list[k][l][1]
                                P *= spatial_tiles_list[k][l][2]
                                Q *= spatial_tiles_list[k][l][3]
                                C *= spatial_tiles_list[k][l][4]
                                K *= spatial_tiles_list[k][l][5]
                                H *= spatial_tiles_list[k][l][6]
                                N *= spatial_tiles_list[k][l][7]
                                R *= temporal_tiles_list[k][l][0]
                                S *= temporal_tiles_list[k][l][1]
                                P *= temporal_tiles_list[k][l][2]
                                Q *= temporal_tiles_list[k][l][3]
                                C *= temporal_tiles_list[k][l][4]
                                K *= temporal_tiles_list[k][l][5]
                                H *= temporal_tiles_list[k][l][6]
                                N *= temporal_tiles_list[k][l][7]
                            tensor_capacity = N*((P-1)*self.problems_list[k]['problem']['instance']['Wstride']+1+self.problems_list[k]['problem']['instance']['Wdilation']*(R-1))*((Q-1)*self.problems_list[k]['problem']['instance']['Hstride']+1+self.problems_list[k]['problem']['instance']['Hdilation']*(S-1))*C*H
                          
                        else:
                            for l in range(r+1):
                                for dim in self.tensor_dimensions[tensor]:
                                    tensor_capacity *= spatial_tiles_list[k][l][dim]
                                    tensor_capacity *= temporal_tiles_list[k][l][dim]
                        buffer_capacity += tensor_capacity
                    buffer_config.append(buffer_capacity)
                all_configs.append([spatial_config, buffer_config])
            # 对每个约束维度取所有矩阵的最大值（即最小硬件配置）
            max_spatial = [max(col) for col in zip(*[cfg[0] for cfg in all_configs])]
            max_buffer = [max(col) for col in zip(*[cfg[1] for cfg in all_configs])]
            min_config = [max_spatial, max_buffer]
            return min_config
        
        def safe_ceil(x, eps=1e-6):
            """安全向上取整，避免因浮点误差导致1.0000001被取为2"""
            # 先四舍五入到eps精度（如1e-6），消除微小误差
            x_rounded = np.around(x, decimals=3)  # 保留6位小数，可根据需要调整
            # 若修正后的值与某个整数的差小于eps，则视为该整数
            integer_part = np.floor(x_rounded)
            if x_rounded - integer_part < eps:
                return int(integer_part)
            else:
                # 确实需要向上取整的情况（如1.1 → 2）
                return int(np.ceil(x_rounded))

        
        spatial_tiles_solution = solutions[0]
        temporal_tiles_solution = solutions[1]
        
        
        spatial_tiles_list = []
        temporal_tiles_list = []
        datas = []
        for i in range(len(solutions[0])):
            tile_matrix = np.array(temporal_tiles_solution[i])
            # 用vectorize实现批量处理（效率较高）
            safe_ceil_vec = np.vectorize(safe_ceil)
            tile_matrix_int = safe_ceil_vec(tile_matrix).astype(int)
            temporal_tiles_list.append(tile_matrix_int)
            tile_matrix = np.array(spatial_tiles_solution[i])
            # 用vectorize实现批量处理（效率较高）
            safe_ceil_vec = np.vectorize(safe_ceil)
            tile_matrix_int = safe_ceil_vec(tile_matrix).astype(int)
            spatial_tiles_list.append(tile_matrix_int)
            
            # spatial_tiles_list.append(np.ceil(spatial_tiles_solution[i]).astype(int))
            # temporal_tiles_list.append(np.ceil(temporal_tiles_solution[i]).astype(int))
        
        config = min_hardware_config(spatial_tiles_list, temporal_tiles_list)
        
        new_data = []
        for i in range(len(solutions[0])):
            last_row = compute_last_row(self.tensor_dimensions_list[i], spatial_tiles_list[i], temporal_tiles_list[i])
            new_data.append((spatial_tiles_list[i], np.vstack((temporal_tiles_list[i][:-1], last_row))))
        return new_data, config

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
    
    def generate_mapping_and_config_for_all_DNN(self, dimensions_list, p):
        sols, cfg = self.lpsolver_all_DNN(dimensions_list, p)
        mapping_list = []
        for i in range(len(sols)):
            # 如果要探索更多的配置，这部分需要修改
            mapping_dict = utils.generate_mapping_for_lpsolver3(sols[i], p)
            mapping_list.append(mapping_dict)
        arch_dict = utils.generate_accelerator(cfg, p)
        return mapping_list, None, arch_dict
    
    def thread_fun_p(self, p):
        original_dir = os.getcwd()
        # 创建一个唯一的临时文件夹
        temp_dir = f'{original_dir}/mars_tmp/temp_{uuid.uuid4()}'
        os.makedirs(temp_dir)
        rewards = []
        # try:
        mapping_list, _, arch = self.generate_mapping_and_config_for_all_DNN(self.tensor_dimensions_list, p)
        
        
        # 初始化存储列表（记录每层原始值，而非奖励值）
        values_list = []
        # 原始指标存储（用于计算聚合值）
        raw_metrics = {
            "cycles": [], "energy": [], "utilization": [], 
            "GFLOPs": [], "EDP": []  # 保留各层EDP仅作备份
        }
        utils.store_yaml(f'{temp_dir}/temp_arch.yaml', arch)
        for i in range(len(mapping_list)):
            mapping = Mapping(mapping_list[i])
            remainders = {}
            outermost_idx = {}
            problem = self.problems_list[i]
            
            # 计算remainders和outermost_idx（原有逻辑不变）
            for d in mapping.factor_dict.keys():
                T = problem['problem']['instance'][d]
                F = mapping.factor_dict[d]
                remainders[d], outermost_idx[d] = utils.find_remainders2(F[::-1], T)
            
            # 生成临时映射并运行timeloop（原有逻辑不变）
            temp_mapping = utils.generate_mapping(
                mapping.factor_dict, mapping.permutation_list, 
                mapping.target_list, mapping.type_list, 
                mapping.bypass_list, remainders, outermost_idx
            )
            output_stats_path = os.path.join(temp_dir, 'timeloop-model.stats.txt')
            utils.store_yaml(f'{temp_dir}/{i}_temp.yaml', temp_mapping)
            utils.store_yaml(f'{temp_dir}/{i}_temp_prob.yaml', problem)
            utils.run_timeloop(
                f'{temp_dir}/temp_arch.yaml', 
                f'{temp_dir}/{i}_temp_prob.yaml', 
                f'{temp_dir}/{i}_temp.yaml', 
                cwd=temp_dir
            )
            
            # 解析输出并记录原始指标（核心修改：先存原始值，再算奖励）
            self.observation = utils.parse_timeloop_output(output_stats_path)
            if self.observation is not None:
                # 记录当前层的原始指标（未取负/未加工）
                for metric in raw_metrics.keys():
                    raw_metrics[metric].append(self.observation.get(metric, 0.0))
                # 生成当前层的奖励值（仅用于记录，最终奖励从原始值聚合计算）
                values = self.judge()
                values_list.append(values)
            else:
                values_list.append(None)

        # -------------- 核心：按合理逻辑计算聚合指标 --------------
        rewards = []
        # 1. 先计算总cycles和总energy（基础指标）
        total_cycles = sum(raw_metrics["cycles"])
        total_energy = sum(raw_metrics["energy"])
        # 2. 遍历fitness_obj，按指标类型计算聚合值
        fitness_obj = self.fitness_obj
        for term in fitness_obj:
            if term == "cycles":
                # 总周期：直接累加
                rewards.append(-total_cycles)  # 负号：周期越少奖励越高
            elif term == "energy":
                # 总能耗：直接累加
                rewards.append(-total_energy)  # 负号：能耗越少奖励越高
            elif term == "EDP":
                # 总EDP：总周期 × 总能耗
                total_edp = total_cycles * total_energy * 0.000001
                rewards.append(-total_edp)  # 负号：EDP越小奖励越高
            elif term == "utilization":
                # 利用率：有效算力加权平均（权重=各层cycles占比）
                if total_cycles == 0:
                    avg_util = 0.0
                else:
                    # 加权平均：(util1×cycles1 + util2×cycles2 + ...) / 总cycles
                    weighted_util = sum(u * c for u, c in zip(raw_metrics["utilization"], raw_metrics["cycles"]))
                    avg_util = weighted_util / total_cycles
                rewards.append(avg_util)  # 正号：利用率越高奖励越高
            elif term == "GFLOPs":
                # GFLOPs：总算力 / 总时间（这里用总cycles近似时间）
                if total_cycles == 0:
                    total_gflops = 0.0
                else:
                    # 总算力=各层GFLOPs×各层cycles（因为GFLOPs是“每秒算力”，乘时间=总算力）
                    total_compute = sum(g * c for g, c in zip(raw_metrics["GFLOPs"], raw_metrics["cycles"]))
                    total_gflops = total_compute / total_cycles
                rewards.append(total_gflops)  # 正号：算力越高奖励越高
            else:
                raise NameError(f'Undefined fitness type: {term}')
        # except Exception as e:
        #     print(f"发生错误: {e}")
        #     rewards = None
        a = (raw_metrics["cycles"], raw_metrics["energy"])
        # finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return rewards, values_list, mapping_list, a

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

    def create_genome_for_parameters3(self, num_population):
        pop = []
        buffer_hierarchy = ['Registers', 'AccumulationBuffer', 'WeightBuffer', 'InputBuffer', 'GlobalBuffer', 'DRAM']
        tensor_in_buffer = {'Registers': ['Weights'], 'AccumulationBuffer': ['Outputs'], 'WeightBuffer': ['Weights'], 'InputBuffer': ['Inputs'], 'GlobalBuffer': ['Inputs','Outputs'], 'DRAM': ['Weights','Inputs','Outputs']}
        layer_num = len(self.problems_list)
        row = len(buffer_hierarchy)
        col = 8
        for i in range(num_population):
            # 只生成一份矩阵
            temp_tw = np.random.uniform(low=1.0, high=100.0, size=(row, col))
            temp_sw = np.random.uniform(low=1.0, high=100.0, size=(row, col))
            # 每层都引用同一份矩阵（通过列表重复）
            temporal_tile_weights = [temp_tw] * layer_num
            spatial_tile_weights = [temp_sw] * layer_num
            buffer_temporal_weights = np.random.randint(10, 100, size=row).astype(float)
            buffer_spatial_weights = np.random.randint(10, 100, size=row).astype(float)
            p = HardwareConfig("Simba", buffer_hierarchy, 0, tensor_in_buffer,
                            temporal_tile_weights, spatial_tile_weights,
                            buffer_temporal_weights, buffer_spatial_weights)
            pop.append(p)
        return pop
    
    def create_genome_for_gemmini(self, num_population):
        pop = []
        buffer_hierarchy = ['Registers', 'Accumulator', 'Scratchpad', 'DRAM']
        tensor_in_buffer = {'Registers': ['Weights'], 'Accumulator': ['Outputs'], 'Scratchpad': ['Inputs', 'Weights'], 'DRAM': ['Weights','Inputs','Outputs']}
        layer_num = len(self.problems_list)
        row = len(buffer_hierarchy)
        col = 8
        for i in range(num_population):
            temp_tw = np.random.uniform(low=1.0, high=100.0, size=(row, col))
            temp_sw = np.random.uniform(low=1.0, high=100.0, size=(row, col))
            temporal_tile_weights = [temp_tw] * layer_num
            spatial_tile_weights = [temp_sw] * layer_num
            buffer_temporal_weights = np.random.randint(10, 100, size=row).astype(float)
            buffer_spatial_weights = np.random.randint(10, 100, size=row).astype(float)
            p = HardwareConfig("Gemmini", buffer_hierarchy, 0, tensor_in_buffer,
                            temporal_tile_weights, spatial_tile_weights,
                            buffer_temporal_weights, buffer_spatial_weights)
            pop.append(p)
        return pop
    
    def run(self, stage_idx=0, prev_stage_value=0, num_population=10, num_generations=100, elite_ratio=0.05,
                       parents_ratio=0.15, ratio_decay=1, num_finetune=1):
        def crossover(parents, pop, best_spatial_tile_weights, best_temporal_tile_weights, best_buffer_temporal_weights, best_buffer_spatial_weights, no_mutation, alpha=0.5):
            if len(parents) == 1:
                for idx in range(len(pop)):
                    pop[idx] = copy.deepcopy(parents[0])
            else:
                for idx in range(0, len(pop), 2):
                    dad = copy.deepcopy(parents[random.randint(0, len(parents)-1)])
                    mom = copy.deepcopy(parents[random.randint(0, len(parents)-1)])

                    dad.crossover(mom, best_spatial_tile_weights, best_temporal_tile_weights, best_buffer_temporal_weights, best_buffer_spatial_weights, no_mutation, alpha)
                    pop[idx] = dad
                    if idx + 1 < len(pop):
                        pop[idx+1] = mom
        
        num_generations = num_generations
        num_population = num_population
        num_elite = int(num_population * elite_ratio)
        pool = Pool(min(num_population + num_elite, cpu_count()))

        best_reward = [-float("Inf") for _ in range( len(self.fitness_obj))]
        best_sol = None
        best_values = []

      
        if self.accl_name == 'Simba':
            population = self.create_genome_for_parameters3(num_population)
        elif self.accl_name == 'Gemmini':
            population = self.create_genome_for_gemmini(num_population)
        elif self.accl_name == 'NEW':
            population = self.create_genome_for_NEW(num_population)
        # mapping, _ = self.generate_mapping(self.dimension, population[0])
        # exit()
        fitness = np.ones((num_population, len(self.fitness_obj)), float)
        num_parents = num_population
        row = len(population[0].buffer_hierarchy)
        col = 8
        best_temporal_tile_weights = [np.random.uniform(low=1.0, high=100.0, size=(row, col)) for _ in range(len(self.problems_list))]
        best_spatial_tile_weights = [np.random.uniform(low=1.0, high=100.0, size=(row, col)) for _ in range(len(self.problems_list))]
        best_tile_weights_scores = [-float('inf') for _ in range(len(self.problems_list))]
        best_buffer_temporal_weights = [np.random.randint(10, 100, size=row).astype(float) for _ in range(len(self.problems_list))]
        best_buffer_spatial_weights = [np.random.randint(10, 100, size=row).astype(float) for _ in range(len(self.problems_list))]
        no_mutation = [False for _ in range(len(self.problems_list))]
        for g in range(num_generations):
            start_time = time.time()
            # alpha = adaptive_parameters(g, num_generations)
            
        
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

                
                crossover(parents, population, best_spatial_tile_weights, best_temporal_tile_weights, best_buffer_temporal_weights, best_buffer_spatial_weights, no_mutation, alpha=0.5)

                # 变异
                for pop in population[num_elite:]:
                    # 选择一种主要变异策略
                    pop.mutation(best_spatial_tile_weights, best_temporal_tile_weights, best_buffer_temporal_weights, best_buffer_spatial_weights, no_mutation, alpha=0.5, generation=g, max_generations=num_generations)
                

                # 1. 更新种群（参数矩阵）thread_fun_pation长度一致
                fitness = np.zeros((num_population, len(self.fitness_obj)))
                fitness[:num_elite] = elite_fitness  # 前num_elite个位置填充精英的适应度

                results = pool.map(self.thread_fun_p, population)
                reward_list = [res[0] for res in results]
                tile_value_list = [res[1] for res in results]
                mapping_list = [res[2] for res in results]
                a = [res[3] for res in results]
                for idx_prob in range(len(self.problems_list)):
                    current_tile_weights_scores = -float('inf')
                    current_temporal_tile_weights = None
                    current_spatial_tile_weights = None
                    current_buffer_temporal_weights = None
                    current_buffer_spatial_weights = None
                    for idx_pop in range(len(population)):
                        if tile_value_list[idx_pop][idx_prob][0] > current_tile_weights_scores:
                            current_tile_weights_scores = tile_value_list[idx_pop][idx_prob][0]
                            current_temporal_tile_weights = population[idx_pop].temporal_tile_weights[idx_prob]
                            current_spatial_tile_weights = population[idx_pop].spatial_tile_weights[idx_prob]
                            current_buffer_temporal_weights = population[idx_pop].buffer_temporal_weights
                            current_buffer_spatial_weights = population[idx_pop].buffer_spatial_weights

                    if current_tile_weights_scores > best_tile_weights_scores[idx_prob]:
                        best_tile_weights_scores[idx_prob] = current_tile_weights_scores
                        best_temporal_tile_weights[idx_prob] = current_temporal_tile_weights
                        best_spatial_tile_weights[idx_prob] = current_spatial_tile_weights
                        best_buffer_temporal_weights[idx_prob] = current_buffer_temporal_weights
                        best_buffer_spatial_weights[idx_prob] = current_buffer_spatial_weights

                
                print(best_tile_weights_scores)
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
                    best_values = copy.deepcopy(tile_value_list[gen_best_idx])
                    best_a = copy.deepcopy(a[gen_best_idx])

                print(best_values)
                print(no_mutation)
                print("total_cycles, total_energy", best_a)
                num_parents = int(num_population * parents_ratio)
                num_parents = min(num_parents, len(population) - count_non_valid)
                parents_ratio *= ratio_decay

                # if g > 20:
                #     average = np.mean(best_values)
                #     for i in range(len(self.problems_list)):
                #         relative_error = abs(best_values[i][0] - best_tile_weights_scores[i]) / max(abs(best_tile_weights_scores[i]), 1e-8)
                #         if relative_error < 0.2 and average > best_values[i][0]:
                #             no_mutation[i] = True
                
                print( "[Stage {}]Gen {}:  1st stage Reward: {}, Best reward: {}".format(stage_idx + 1, (g + 1), np.abs(prev_stage_value), np.abs(best_reward)))
       
   
            elapsed_time = time.time() - start_time
            print("Generation {} 耗时: {:.3f}秒".format(g, elapsed_time))
        pool.close()

        mapping_list, _, arch = self.generate_mapping_and_config_for_all_DNN(self.tensor_dimensions_list, best_sol)
        
        # 原始指标存储（用于计算聚合值）
        raw_metrics = {
            "cycles": [], "energy": [], "utilization": [], 
            "GFLOPs": [], "EDP": []  # 保留各层EDP仅作备份
        }
        for i in range(len(mapping_list)):
            mapping = Mapping(mapping_list[i])
            remainders = {}
            outermost_idx = {}
            problem = self.problems_list[i]
            for d in mapping.factor_dict.keys():
                T = problem['problem']['instance'][d]
                F = mapping.factor_dict[d]
                remainders[d], outermost_idx[d] = utils.find_remainders2(F[::-1], T)
            # 需要根据探索的结果修改permutation_list，type_list， target_list和bypass_list
            temp_mapping = utils.generate_mapping(mapping.factor_dict, mapping.permutation_list, mapping.target_list, mapping.type_list, mapping.bypass_list, remainders, outermost_idx)
            report_dir = f'{self.report_dir}/layer-{i}'
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            utils.store_yaml(f'{report_dir}/map.yaml', temp_mapping)
            utils.store_yaml(f'{report_dir}/problem.yaml', problem)
            utils.store_yaml(f'{report_dir}/arch.yaml', arch)
            utils.run_timeloop('arch.yaml', 'problem.yaml', 'map.yaml', cwd=report_dir)
            output_stats_path = os.path.join(report_dir, 'timeloop-model.stats.txt')
            # 解析输出并记录原始指标（核心修改：先存原始值，再算奖励）
            self.observation = utils.parse_timeloop_output(output_stats_path)
            if self.observation is not None:
                # 记录当前层的原始指标（未取负/未加工）
                for metric in raw_metrics.keys():
                    raw_metrics[metric].append(self.observation.get(metric, 0.0))
                # 生成当前层的奖励值（仅用于记录，最终奖励从原始值聚合计算）
                values = self.judge()
        # -------------- 核心：按合理逻辑计算聚合指标 --------------
        rewards = []
        # 1. 先计算总cycles和总energy（基础指标）
        total_cycles = sum(raw_metrics["cycles"])
        total_energy = sum(raw_metrics["energy"])
        # 2. 遍历fitness_obj，按指标类型计算聚合值
        fitness_obj = self.fitness_obj
        for term in fitness_obj:
            if term == "cycles":
                # 总周期：直接累加
                rewards.append(-total_cycles)  # 负号：周期越少奖励越高
            elif term == "energy":
                # 总能耗：直接累加
                rewards.append(-total_energy)  # 负号：能耗越少奖励越高
            elif term == "EDP":
                # 总EDP：总周期 × 总能耗
                total_edp = total_cycles * total_energy * 0.000001
                rewards.append(-total_edp)  # 负号：EDP越小奖励越高
            elif term == "utilization":
                # 利用率：有效算力加权平均（权重=各层cycles占比）
                if total_cycles == 0:
                    avg_util = 0.0
                else:
                    # 加权平均：(util1×cycles1 + util2×cycles2 + ...) / 总cycles
                    weighted_util = sum(u * c for u, c in zip(raw_metrics["utilization"], raw_metrics["cycles"]))
                    avg_util = weighted_util / total_cycles
                rewards.append(avg_util)  # 正号：利用率越高奖励越高
            elif term == "GFLOPs":
                # GFLOPs：总算力 / 总时间（这里用总cycles近似时间）
                if total_cycles == 0:
                    total_gflops = 0.0
                else:
                    # 总算力=各层GFLOPs×各层cycles（因为GFLOPs是“每秒算力”，乘时间=总算力）
                    total_compute = sum(g * c for g, c in zip(raw_metrics["GFLOPs"], raw_metrics["cycles"]))
                    total_gflops = total_compute / total_cycles
                rewards.append(total_gflops)  # 正号：算力越高奖励越高
            else:
                raise NameError(f'Undefined fitness type: {term}')
        print('聚合的结果', total_cycles, total_energy,total_edp)
        print(raw_metrics)
        return total_cycles, total_energy         


import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import numpy as np
import copy
import os
import shutil
import uuid
from multiprocessing import cpu_count

class HardwareOptimEnv(gym.Env):
    def __init__(self, explorer_creator, accl_name, fitness_index=0):
        super().__init__()
        self.explorer_creator = explorer_creator
        self.accl_name = accl_name
        temp_explorer = self.explorer_creator()
        temp_config = self._create_config(temp_explorer)
        self.vec_dim = len(temp_config.get_param_vector())
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.vec_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.fitness_index = fitness_index
        self.best_reward = -float('inf')
        self.best_config = None
        self.explorer = None

    def _create_config(self, explorer):
        if explorer.accl_name == 'Simba':
            return explorer.create_genome_for_parameters3(1)[0]
        elif explorer.accl_name == 'Gemmini':
            return explorer.create_genome_for_gemmini(1)[0]
        else:
            raise ValueError(f"Unsupported accelerator: {explorer.accl_name}")

    def _get_explorer(self):
        if self.explorer is None:
            self.explorer = self.explorer_creator()
        return self.explorer

    def reset(self, seed=None, options=None):
        self.explorer = self.explorer_creator()
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        explorer = self._get_explorer()
        config = self._create_config(explorer)
        weight = np.clip(np.exp(action), 0.5, 200.0)
        config.set_param_vector(weight)

        try:
            result = explorer.thread_fun_p(config)
            if result[0] is not None:
                reward = result[0][self.fitness_index]
            else:
                reward = -1e6
        except Exception as e:
            print(f"评估异常: {e}")
            reward = -1e6

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_config = copy.deepcopy(config)

        # 新版 Gym API 要求 5 个返回值：obs, reward, terminated, truncated, info
        return np.zeros(1, dtype=np.float32), reward, True, False, {}
# 在 HardwareConfig 中添加向量化辅助方法（直接添加即可，无需修改原有代码）
def _get_param_vector(self):
    """将所有优化权重展平为一维向量（所有层共用同一组权重）"""
    vectors = []
    # 只取第一个矩阵即可，因为所有层引用同一对象
    vectors.append(self.temporal_tile_weights[0].flatten())
    vectors.append(self.spatial_tile_weights[0].flatten())
    return np.concatenate(vectors).astype(np.float32)

def _set_param_vector(self, vector):
    """从一维向量恢复权重矩阵，并让所有层共享"""
    offset = 0
    # 恢复 temporal 权重（只需修改第一个矩阵）
    size = self.temporal_tile_weights[0].size
    self.temporal_tile_weights[0].flat = vector[offset:offset+size]
    offset += size
    # 恢复 spatial 权重
    size = self.spatial_tile_weights[0].size
    self.spatial_tile_weights[0].flat = vector[offset:offset+size]
    # 强制所有层指向同一个矩阵对象（确保共享）
    layer_num = len(self.temporal_tile_weights)
    tw0 = self.temporal_tile_weights[0]
    sw0 = self.spatial_tile_weights[0]
    self.temporal_tile_weights = [tw0] * layer_num
    self.spatial_tile_weights = [sw0] * layer_num
    
# 将方法绑定到 HardwareConfig 类（如果尚未存在）
if not hasattr(HardwareConfig, 'get_param_vector'):
    HardwareConfig.get_param_vector = _get_param_vector
if not hasattr(HardwareConfig, 'set_param_vector'):
    HardwareConfig.set_param_vector = _set_param_vector


# MappingExplorer 新增方法
def run_rl(self, total_timesteps=400, n_envs=4, learning_rate=1e-3, fitness_obj='EDP'):
    """
    用强化学习（PPO）替代遗传算法进行映射探索。
    
    参数:
        total_timesteps: 总训练步数（每一次 step 运行一次完整的 DNN 评估）
        n_envs: 并行环境数（建议不超过 CPU 核心数）
        learning_rate: PPO 学习率
        fitness_obj: 优化目标，需与 self.fitness_obj 中的某项匹配（默认 'EDP'）
    """
    # 确定奖励索引
    if fitness_obj not in self.fitness_obj:
        raise ValueError(f"fitness_obj 必须是 {self.fitness_obj} 之一")
    fitness_idx = self.fitness_obj.index(fitness_obj)

    # 保存构造 explorer 所需的参数
    explorer_args = {
        'operator_instance': self.problems_list,
        'buffer_capacity': self.buffer_capacity,
        'spatial_capacity': self.spatial_capacity,
        'accelerator': self.accl_name,
        'mapper': self.mapper,
        'version': self.accelerator.version,
        'report_dir': self.report_dir
    }

    def make_env(rank):
        def _init():
            # 每个子进程创建一个全新的 MappingExplorer
            exp = MappingExplorer(**explorer_args)
            env = HardwareOptimEnv(
                explorer_creator=lambda: exp,  # 每次 reset 时使用同一个 exp
                accl_name=exp.accl_name,
                fitness_index=fitness_idx
            )
            return env
        return _init

    # 使用 SubprocVecEnv 并行运行多个环境
    # env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    env = DummyVecEnv([make_env(0)]) 

    # 创建 PPO 模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=min(2048, total_timesteps) // n_envs,  # 每次 rollout 的步数
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        # tensorboard_log=os.path.join(self.report_dir, "ppo_logs"),
        device='cpu'
    )

    print("开始强化学习训练...")
    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(self.report_dir, "ppo_hardware_opt"))

    # 从环境中提取最佳配置（所有环境中最佳的那个）
    envs = env.get_attr('best_config')
    best_rewards = env.get_attr('best_reward')
    best_idx = np.argmax(best_rewards)
    best_config = envs[best_idx]
    best_reward = best_rewards[best_idx]

    print(f"训练完成，最佳奖励: {best_reward}")

    # 使用最佳配置重新运行一次完整的最终评估（生成报告）
    print("生成最终映射与报告...")
    explorer_final = MappingExplorer(**explorer_args)
    # 注意：best_config 是在子进程 explorer 中生成的，其内部参数可直接使用
    # 为安全起见，我们直接调用 explorer_final 的 thread_fun_p 以得到最终统计
    mapping_list, _, arch = explorer_final.generate_mapping_and_config_for_all_DNN(
    explorer_final.tensor_dimensions_list, best_config)

    out_mapping = []
    # 按层生成映射文件
    raw_metrics = {"cycles": [], "energy": [], "utilization": [], "GFLOPs": [], "EDP": []}
    for i in range(len(mapping_list)):
        mapping = Mapping(mapping_list[i])
        remainders = {}
        outermost_idx = {}
        problem = self.problems_list[i]
        for d in mapping.factor_dict.keys():
            T = problem['problem']['instance'][d]
            F = mapping.factor_dict[d]
            remainders[d], outermost_idx[d] = utils.find_remainders2(F[::-1], T)
        temp_mapping = utils.generate_mapping(
            mapping.factor_dict, mapping.permutation_list,
            mapping.target_list, mapping.type_list, mapping.bypass_list,
            remainders, outermost_idx
        )
        out_mapping.append(temp_mapping)
        report_dir = os.path.join(self.report_dir, f'layer-{i}')
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        utils.store_yaml(os.path.join(report_dir, 'map.yaml'), temp_mapping)
        utils.store_yaml(os.path.join(report_dir, 'problem.yaml'), problem)
        utils.store_yaml(os.path.join(report_dir, 'arch.yaml'), arch)
        utils.run_timeloop('arch.yaml', 'problem.yaml', 'map.yaml', cwd=report_dir)
        output_stats_path = os.path.join(report_dir, 'timeloop-model.stats.txt')
        obs = utils.parse_timeloop_output(output_stats_path)
        if obs is not None:
            for metric in raw_metrics.keys():
                raw_metrics[metric].append(obs.get(metric, 0.0))

    # 计算聚合指标
    total_cycles = sum(raw_metrics["cycles"])
    total_energy = sum(raw_metrics["energy"])
    print("各层 cycles:", raw_metrics["cycles"])
    print("各层 energy:", raw_metrics["energy"])
    print("各层的mapping")
    for i in out_mapping:
        print(i)
    total_edp = total_cycles * total_energy * 1e-6
    print(f"最终结果: 总周期 {total_cycles:.2f}, 总能量 {total_energy:.2f}, EDP {total_edp:.2f}")
    return total_cycles, total_energy


# 将新方法绑定到 MappingExplorer 类
MappingExplorer.run_rl = run_rl