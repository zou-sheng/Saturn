from input_objs import Arch, Prob, Mapspace, Mapping
import pathlib
from datetime import datetime
import time
from algorithms.simple_mobo.mobo import MOBO

class HardwareFisrt:
    def __int__(self, operator_instance, accelerator_dir, accelerator, mapper, type, version, report_dir, optim_obj, all_DNN=False):
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



        # 针对多层DNN的中间变量
        self.buffer_temporal_list = None
        self.buffer_spatial_list = None

def run(operator_instance, accelerator_dir, accelerator, mapper, type, version, report_dir, optim_obj, all_DNN=False):
    HF = HardwareFisrt(operator_instance, accelerator_dir, accelerator, mapper, type, version, report_dir, optim_obj, all_DNN=False)
    bo = MOBO(nobjs=3, ninit=10, lb=ds.lb, ub=ds.ub, dims=ds.dims, func=ds, neval=60, 
        seed=35)
    begin = time.time()
    bo.start()
    end = time.time()
    print(f"MOBO takes {end - begin} seconds")
    # HF.dump_pareto(bo.curt_pareto_samples, output_dir)