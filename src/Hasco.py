from input_objs import Arch, Prob, Mapspace, Mapping
import pathlib
from datetime import datetime
import time
from algorithms.simple_mobo.mobo import MOBO
import numpy as np
from typing import Dict, List, Tuple
import os
import yaml
import shutil
import uuid
from hasco_explorer import MappingExplorer
from pathlib import Path

class HardwareFirstSearch:
    def __init__(self, operator_instance, accelerator_dir, accelerator, mapper, type, version, report_dir, optim_obj, all_DNN=False):
        self.accl_name = accelerator

        if optim_obj == 'latency':
            self.fitness_obj = ['cycles']
        elif optim_obj == 'all':
            self.fitness_obj = ['EDP', 'energy', 'cycles']
        else:
            self.fitness_obj = [optim_obj]
        self.timeloop_out_config_path = f'./tmp/out_config_{datetime.now().strftime("%H:%M:%S")}'
        
        self.all_DNN = all_DNN
        if self.all_DNN:
            self.problems_list = operator_instance
            self.operator_instance = None  # 明确初始化
            self.tensor_dimensions_list = []
            for i in self.problems_list:
                dim = []
                for d in ['R', 'S', 'P', 'Q', 'C', 'K', 'H', 'N']:
                    dim.append(i['problem']['instance'][d])
                self.tensor_dimensions_list.append(dim)
        else:
            self.operator_instance = operator_instance
            self.problems_list = None  # 明确初始化

        
        self.report_dir = report_dir 
        self.mapper = mapper

        arch_path = pathlib.Path('{}/{}/{}/{}.yaml'.format(accelerator_dir, mapper, accelerator, type)).resolve()
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



        # 针对多层DNN的中间变量
        self.buffer_temporal_list = None
        self.buffer_spatial_list = None

        self.init()
        self.original_dir = os.getcwd()
        self.temp_dir = f'{self.original_dir}/HF_tmp/{self.accl_name}/temp_{uuid.uuid4()}'
        self.clean_temp = True

    # def __del__(self):
    #     """类实例销毁时自动清理临时目录（析构函数）"""
    #     if self.clean_temp and os.path.exists(self.temp_dir):
    #         try:
    #             shutil.rmtree(self.temp_dir)
    #             print(f"✅ 自动清理临时目录：{self.temp_dir}")
    #         except Exception as e:
    #             print(f"⚠️ 清理临时目录失败：{e}")

    def init(self):
        # --------------------------
        # 统一使用新格式的维度定义
        # --------------------------
        if self.accl_name == 'Simba':
            self.optim_dims = {
                0: {"name": "GlobalBuffer", "attr": "entries", "min": 512, "max": 3584, "slice_num": 24, "default": 2048},
                1: {"name": "InputBuffer", "attr": "entries", "min": 128, "max": 1920, "slice_num": 28, "default": 1024},
                2: {"name": "WeightBuffer", "attr": "entries", "min": 32,  "max": 992,  "slice_num": 30, "default": 512},
                3: {"name": "AccumulationBuffer", "attr": "entries", "min": 32,  "max": 224,  "slice_num": 12, "default": 128},
                4: {"name": "Registers", "attr": "instances", "min": 512, "max": 2048, "slice_num": 12, "default": 1024}
            }
            
            # --------------------------
            # 同步/固定规则（适配新命名）
            # --------------------------
            self.syn_instances = {"LMAC": "Registers"}  # 原PEWeightRegs→Registers
            self.keep_instances = {
                "GlobalBuffer": 1,
                "InputBuffer": 16,
                "WeightBuffer": 128,
                "AccumulationBuffer": 128
            }
        elif self.accl_name == 'Gemmini':
            self.optim_dims = {
                0: {"name": "Scratchpad", "attr": "entries", "min": 512, "max": 3584, "slice_num": 24, "default": 2048},
                1: {"name": "Accumulator", "attr": "entries", "min": 128, "max": 1920, "slice_num": 28, "default": 1024},
                # 2: {"name": "Registers", "attr": "instances", "min": 512, "max": 1024, "slice_num": 16, "default": 1024}
                # 2: {"name": "Registers", "attr": "instances", "min": 512, "max": 1024, "slice_num": 14, "default": 1024}
            }
            
            # --------------------------
            # 同步/固定规则（适配新命名）
            # --------------------------
            self.syn_instances = {"LMAC": "Registers"}  # 原PEWeightRegs→Registers
            self.keep_instances = {
                "Scratchpad": 1,
                "Accumulator": 32,
                "Registers": 1024
            }
        # --------------------------
        # 可优化维度总数
        # --------------------------
        self.dims = len(self.optim_dims)

    def get_space_list(self, dim_id: int) -> List[int]:
        if dim_id not in self.optim_dims:
            raise ValueError(f"维度ID {dim_id} 未在optim_dims中定义")
        
        dim = self.optim_dims[dim_id]
        min_val = dim["min"]
        max_val = dim["max"]
        slice_num = dim["slice_num"]
        step = (max_val - min_val) / slice_num
        
        # 生成离散的可选值列表
        space_list = [int(min_val + step * (i + 1)) for i in range(slice_num)]
        return space_list

    def get_ub(self) -> np.ndarray:
        ub = np.zeros(self.dims)
        for dim_id in range(self.dims):
            ub[dim_id] = len(self.get_space_list(dim_id))
        return ub

    def gen_depth_updates(self, candidate: List[int]) -> Dict[str, int]:
        updates = {}
        for dim_id in range(len(candidate)):
            names = self.dim2names[dim_id]
            if "depth" in names:
                level_name = names[0]
                space_list = self.get_space_list(dim_id)
                level_depth = space_list[candidate[dim_id]]
                updates[level_name] = level_depth
        
        for key, value in self.syn_depth.items():
            updates[key] = updates[value]
        for key, value in self.keep_depth.items():
            updates[key] = value
        
        return updates

    # def calc_area(self, cand: List[int]) -> float:
    #     # get updates
    #     depth_updates = self.gen_depth_updates(cand)
    #     inst_updates = self.gen_instances_updates(cand)
    #     # call accelergy handler
    #     self.accelergy_handler.update_accelergy(depth_updates, inst_updates)
    #     ert, art = self.accelergy_handler.get_ert_art()
    #     # calculate area
    #     area = self.gen_area_from_art(art, inst_updates)
    #     return area

    def gen_instances_updates(self, candidate: List[int]) -> Dict[str, int]:
        updates = {}
        for key, value in self.keep_instances.items():
            updates[key] = value
        for dim_id in range(len(candidate)):
            names = self.dim2names[dim_id]
            if "instances" in names:
                name = names[0]
                space_list = self.get_space_list(dim_id)
                instances_num = space_list[candidate[dim_id]]
                updates[name] = instances_num
        
        for key, value in self.syn_instances.items():
            updates[key] = updates[value]
        return updates

    def _get_value_by_idx(self, dim_id: int, idx: int) -> int:
        """
        根据维度ID和索引，计算对应的实际参数值（方案B适配版）
        :param dim_id: 维度ID（对应optim_dims的key）
        :param idx: 维度内的索引（0 ~ slice_num-1）
        :return: 实际参数值（如entries/instances的具体数值）
        """
        # 校验维度ID合法性
        if dim_id not in self.optim_dims:
            raise ValueError(f"无效的维度ID：{dim_id}，仅支持{list(self.optim_dims.keys())}")
        
        # 从optim_dims中读取维度配置
        dim_config = self.optim_dims[dim_id]
        min_val = dim_config["min"]
        max_val = dim_config["max"]
        slice_num = dim_config["slice_num"]
        
        # 校验索引合法性
        if idx < 0 or idx >= slice_num:
            raise ValueError(f"维度{dim_id}的索引{idx}越界，范围：0 ~ {slice_num-1}")
        
        # 计算步长并生成实际值
        step = (max_val - min_val) / slice_num
        target_value = int(min_val + step * (idx + 1))
        return target_value

    def __call__(self, x: np.ndarray) -> List[float]:
        energy: float = 2**63
        cycles: float = 2**63
        area: float = 2**63
        cand = list(x.astype(int))
        #先不处理area
        area = 0       
        # ========== 2. 硬编码基础架构模板（核心：替代原Accelergy加载的架构） ==========
        if self.accl_name == 'Simba':
            new_arch = {
                "arch": {
                    "arithmetic": {
                        "instances": 1024,  # 默认值，会被候选参数更新
                        "word-bits": 8
                    },
                    "storage": [
                        {
                            "cluster-size": 64,
                            "entries": 1,
                            "instances": 1024,  # 默认值
                            "name": "Registers",
                            "num-banks": 8,
                            "num-ports": 2,
                            "word-bits": 8
                        },
                        {
                            "cluster-size": 1,
                            "entries": 3072,  # 默认值
                            "instances": 16,
                            "name": "AccumulationBuffer",
                            "network-word-bits": 16,
                            "num-banks": 2,
                            "num-ports": 2,
                            "word-bits": 24
                        },
                        {
                            "block-size": 4,
                            "entries": 32768,  # 默认值
                            "instances": 16,
                            "name": "WeightBuffer",
                            "num-banks": 8,
                            "num-ports": 1,
                            "word-bits": 8
                        },
                        {
                            "block-size": 4,
                            "entries": 8192,  # 默认值
                            "instances": 16,
                            "name": "InputBuffer",
                            "num-banks": 1,
                            "num-ports": 2,
                            "word-bits": 8
                        },
                        {
                            "block-size": 8,
                            "entries": 65536,  # 默认值
                            "instances": 1,
                            "name": "GlobalBuffer",
                            "num-banks": 256,
                            "num-ports": 2,
                            "word-bits": 8
                        },
                        {
                            "bandwidth": 1,
                            "block-size": 64,
                            "instances": 1,
                            "name": "DRAM",
                            "technology": "DRAM",
                            "word-bits": 8
                        }
                    ]
                }
            }
            
            mapspace_config = {
                "mapspace": {
                    "constraints": [
                        {
                            "target": "Registers",
                            "type": "datatype",
                            "keep": [
                                "Weights"
                            ],
                            "bypass": [
                                "Inputs",
                                "Outputs"
                            ]
                        },
                        {
                            "target": "AccumulationBuffer",
                            "type": "datatype",
                            "keep": [
                                "Outputs"
                            ],
                            "bypass": [
                                "Weights",
                                "Inputs"
                            ]
                        },
                        {
                            "target": "WeightBuffer",
                            "type": "datatype",
                            "keep": [
                                "Weights"
                            ],
                            "bypass": [
                                "Inputs",
                                "Outputs"
                            ]
                        },
                        {
                            "target": "InputBuffer",
                            "type": "datatype",
                            "keep": [
                                "Inputs"
                            ],
                            "bypass": [
                                "Weights",
                                "Outputs"
                            ]
                        },
                        {
                            "target": "GlobalBuffer",
                            "type": "datatype",
                            "keep": [
                                "Inputs",
                                "Outputs"
                            ],
                            "bypass": [
                                "Weights"
                            ]
                        },
                        {
                            "target": "DRAM",
                            "type": "datatype",
                            "keep": [
                                "Inputs",
                                "Outputs",
                                "Weights"
                            ]
                        }
                    ]
                }
            }
        if self.accl_name == 'Gemmini':
            new_arch = {
                "arch": {
                    "arithmetic": {
                        "instances": 1024,  # 默认值，会被候选参数更新
                        "word-bits": 8
                    },
                    "storage": [
                        {
                            "cluster-size": 64,
                            "entries": 1,
                            "instances": 1024,  # 默认值
                            "name": "Registers",
                            "num-banks": 8,
                            "num-ports": 2,
                            "word-bits": 8
                        },
                        {
                            "cluster-size": 1,
                            "entries": 3072,  # 默认值
                            "instances": 64,
                            "name": "Accumulator",
                            "network-word-bits": 16,
                            "num-banks": 2,
                            "num-ports": 2,
                            "word-bits": 24
                        },
                        {
                            "block-size": 4,
                            "entries": 32768,  # 默认值
                            "instances": 1,
                            "name": "Scratchpad",
                            "num-banks": 8,
                            "num-ports": 1,
                            "word-bits": 8
                        },
                        {
                            "bandwidth": 1,
                            "block-size": 64,
                            "instances": 1,
                            "name": "DRAM",
                            "technology": "DRAM",
                            "word-bits": 8,
                            "shared_bandwidth": 16
                        }
                    ]
                }
            }
            
            mapspace_config = {
                "mapspace": {
                    "constraints": [
                        {
                            "target": "Registers",
                            "type": "datatype",
                            "keep": [
                                "Outputs",
                                "Inputs"
                            ],
                            "bypass": [
                                "Weights"
                            ]
                        },
                        {
                            "target": "Accumulator",
                            "type": "datatype",
                            "keep": [
                                "Weights",
                                "Inputs"
                            ],
                            "bypass": [
                                "Outputs"
                            ]
                        },
                        {
                            "target": "Scratchpad",
                            "type": "datatype",
                            "keep": [
                                "Inputs",
                                "Weights"
                            ],
                            "bypass": [
                                "Outputs"
                            ]
                        },
                        {
                            "target": "DRAM",
                            "type": "datatype",
                            "keep": [
                                "Inputs",
                                "Outputs",
                                "Weights"
                            ]
                        }
                    ]
                }
            }
        
        # cand_area = self.calc_area(cand) # update accelergy in the mean time
        # if(cand_area > self.default_area):
        #     return [energy, cycles, area]
        
        # generate tmp arch file
        # ========== 3. 根据候选参数更新架构（核心逻辑） ==========
        for dim_id, idx in enumerate(cand):
            try:
                # 调用修复后的_get_value_by_idx方法获取实际值
                target_value = self._get_value_by_idx(dim_id, idx)
                # 获取维度对应的组件名和属性名
                dim_config = self.optim_dims[dim_id]
                comp_name = dim_config["name"]
                attr_name = dim_config["attr"]

                # 更新架构字典
                if self.accl_name == 'Simba':
                    if dim_id == 4: # PE寄存器特殊处理
                        new_arch["arch"]["arithmetic"][attr_name] = target_value
                elif self.accl_name == 'Gemmini':
                    if dim_id == 2: # PE寄存器特殊处理
                        new_arch["arch"]["arithmetic"][attr_name] = target_value
                
                for storage_comp in new_arch["arch"]["storage"]:
                    if storage_comp["name"] == comp_name:
                        storage_comp[attr_name] = target_value
                        break

            except ValueError as e:
                print(f"❌ 维度{dim_id}参数更新失败：{e}")
                return [energy, cycles, area]
            
        instance = [comp["instances"] for comp in new_arch["arch"]["storage"]]
        spatial_capacity = [1] + [int(instance[i] / instance[i+1]) for i in range(len(instance)-1)]
        buffer_capacity = [comp.get('entries', 1) for comp in new_arch["arch"]["storage"]]
        # ========== 4. 生成临时架构文件 ==========
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
            temp_arch_path = os.path.join(self.temp_dir, "arch.yaml")
            with open(temp_arch_path, "w", encoding='utf-8') as f:
                yaml.dump(
                    new_arch,
                    f,
                    default_flow_style=False,  # 块状格式，非行内
                    sort_keys=False,           # 保留键顺序
                    indent=2                   # 缩进2个空格
                )
            temp_mapspace_path = os.path.join(self.temp_dir, "mapspace.yaml")
            with open(temp_mapspace_path, "w", encoding='utf-8') as f:
                yaml.dump(
                    mapspace_config,
                    f,
                    default_flow_style=False,  # 块状格式，非行内
                    sort_keys=False,           # 保留键顺序
                    indent=2                   # 缩进2个空格
                )
        except Exception as e:
            print(f"❌ 生成架构文件失败：{e}")
            return [energy, cycles, area]
        
        # run saturn4HF
        ME = MappingExplorer(self.problems_list, buffer_capacity, spatial_capacity, self.accl_name, 'MARS', 'v1', self.temp_dir)
        
        cycles, energy = ME.run_rl(total_timesteps=400, n_envs=1)
        area = 0 # 先不处理area
        return [energy, cycles, area]

    def run(self):
        lb = np.zeros(self.dims)
        ub = self.get_ub()
        # neval=60 意味着总共会评估 60 个点 (10个初始 + 50个迭代)
        bo = MOBO(nobjs=3, ninit=5, lb=lb, ub=ub, dims=self.dims, func=self, neval=25, 
            seed=35)
        begin = time.time()
        bo.start()
        end = time.time()
        print(f"MOBO takes {end - begin} seconds")

        # ===================== 1. 提取帕累托最优解 =====================
        pareto_X = bo.curt_pareto_samples[0]
        pareto_objs = bo.curt_pareto_samples[1]
        
        print("\n" + "="*50)
        print("🚀 MOBO 优化完成 - 帕累托最优解分析：")
        print(f"帕累托解总数：{len(pareto_X)}")
        print(f"原始总采样点数：{len(bo.samples[0])}") # 应该是 60
        print("="*50)
        
        if len(pareto_X) > 0:
            pareto_X_np = np.array(pareto_X)
            pareto_objs_np = np.array(pareto_objs)
            
            # --- 步骤 A: 在帕累托集中找到乘积最小的解 ---
            if self.dims < 2:
                print("⚠️ 警告：维度小于2，无法计算前两个值的乘积。")
                return

            products = pareto_X_np[:, 0] * pareto_X_np[:, 1]
            best_pareto_idx = np.argmin(products)
            
            best_x_pareto = pareto_X_np[best_pareto_idx]
            best_obj_pareto = pareto_objs_np[best_pareto_idx]
            min_product_val = products[best_pareto_idx]

            # ===================== 🔍 核心新增：回溯原始索引 =====================
            # 获取所有历史采样点 (形状: [60, dims])
            all_history_X = np.array(bo.samples[0])
            
            # 计算最佳解与所有历史点的欧氏距离
            # axis=1 表示对每一行（每个点）计算距离
            distances = np.linalg.norm(all_history_X - best_x_pareto, axis=1)
            
            # 找到距离最小的索引
            original_best_idx = np.argmin(distances)
            min_distance = distances[original_best_idx]
            
            # 验证匹配精度 (浮点数误差通常极小，如 1e-15)
            if min_distance > 1e-5:
                print(f"⚠️ 警告：匹配到的最近点距离为 {min_distance:.6f}，可能存在精度问题或非唯一解。")
            else:
                print(f"✅ 成功定位：该最优点是原始 60 个采样点中的第 [{original_best_idx}] 个 (0-based)。")
                print(f"   即：这是 BO 启动后第 {original_best_idx + 1} 次调用 __call__ 产生的结果。")
            # ===================================================================

            # --- 步骤 B: 输出详细信息 ---
            print(f"\n🏆 【特定目标最优点】(前两个参数乘积最小)")
            print(f"   🆔 原始采样点索引：No. {original_best_idx} (共 60 个点)")
            print(f"   📉 最小乘积值 (x[0]*x[1]): {min_product_val:.6f}")
            print(f"   🔢 完整输入参数 X: {np.round(best_x_pareto, 6)}")
            print(f"   🎯 对应目标值 Objs:  {np.round(best_obj_pareto, 4)}")
            
            # --- 步骤 C: 高精度列表打印 (带原始索引标记) ---
            print(f"\n--- 所有 {len(pareto_X)} 个帕累托解详情 (含原始索引映射) ---")
            print(f"{'帕累托ID':<8} | {'原始ID':<8} | {'Prod':<12} | {'X[:3] (高精度)':<35} | {'Objs':<25} | 标记")
            print("-" * 100)
            
            for idx, (x, objs) in enumerate(zip(pareto_X, pareto_objs)):
                is_best = (idx == best_pareto_idx)
                marker = "🏆 BEST" if is_best else ""
                
                # 计算当前帕累托点在历史数据中的原始索引
                dists = np.linalg.norm(all_history_X - np.array(x), axis=1)
                orig_idx = np.argmin(dists)
                
                p_val = x[0] * x[1]
                # 格式化输出
                x_str = np.array2string(np.array(x)[:3], precision=6, suppress_small=True, floatmode='fixed')
                objs_str = np.array2string(np.round(objs, 4), precision=4, suppress_small=True)
                
                print(f"[{idx:<6}] | [{orig_idx:<6}] | {p_val:<12.6f} | {x_str:<35} | {objs_str:<25} | {marker}")

            # --- 步骤 D: 统计信息 ---
            print("\n📊 全体帕累托解目标值统计：")
            for obj_idx in range(bo.nobjs):
                obj_vals = pareto_objs_np[:, obj_idx]
                # 注意：samples[1] 存的是取反后的值（如果是 minimization），但 curt_pareto_samples 里已经变回来了
                # 这里直接统计 pareto_objs_np 即可
                print(f"目标 {obj_idx+1} - 最小值：{np.min(obj_vals):.4f}，平均值：{np.mean(obj_vals):.4f}")
                
        else:
            print("⚠️ 未找到任何帕累托最优解！")
        print("="*50 + "\n")

if __name__ == "__main__":
    pass