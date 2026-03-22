import pathlib
import utils
import copy
import re
import numpy as np

class Arch:
    def __init__(self, arch_path, arch_name, version):
        self.path      = arch_path.resolve()
        self.arch_dict = utils.parse_yaml(self.path)
        self.arch_name = arch_name
        self.version   = version

    def config_str(self):
        """Return the filename for the input yaml with postfix."""
        return self.path.stem

    def print(self):
        print(self.__dict__)

    def get_arch_info(self):
        if self.version == 'v1':
            return self.get_arch_info_v1()
        elif self.version == 'v3':
            return self.get_arch_info_v3()
        else:
            raise Exception("不支持该版本！")

    def get_arch_info_v1(self):
        arch = copy.deepcopy(self.arch_dict)
        buffer_name_list = []
        buffer_size_list = []
        num_instances = []
        num_buffer_levels = 0
        arch = arch['arch']

        num_arithmetic = arch['arithmetic']['instances']

        storage = arch['storage']
        num_buffer_levels = len(storage)


        for buffer in storage:
            buffer_name_list.append(buffer['name'])
            num_instances.append(buffer['instances'])
            if buffer['name'] == 'DRAM':
                buffer_size_list.append(float('Inf'))
            else:
                buffer_size_list.append(buffer['entries'])

        
        buffer_name_list.reverse()
        buffer_size_list.reverse()
        num_instances.reverse()
        num_instances.append(num_arithmetic)

        print(buffer_name_list, num_instances, buffer_size_list)

        # num_pes = int(num_arithmetic / reg_cluster_size)
        # assert(num_arithmetic % reg_cluster_size == 0)
        sp_cstr = []
        for i in range(len(num_instances) - 1):
            allowed_sp_size = num_instances[i + 1] // num_instances[i]
            sp_cstr.append(allowed_sp_size)
            if num_instances[i + 1] % num_instances[i] != 0:
                print(num_instances[i + 1], num_instances[i])
                raise ValueError('Invalid Architecture File. '
                                 'Buffer hierarchy not perfectly divisible.')

        print(sp_cstr)
       
        return {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_name_list)}, \
               {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_size_list)}, \
               {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), sp_cstr)}, \
               num_buffer_levels
        
    def get_arch_info_v3(self):
        arch = copy.deepcopy(self.arch_dict)
        buffer_name_list = []
        buffer_size_list = []
        num_instances = []
        num_buffer_levels = 0
        arch = arch['architecture']

        if self.arch_name == 'Simba':
            main_memory = arch['subtree'][0]['local'][0]
            buffer_name = main_memory['name']
            attributes = main_memory['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            global_buffer = arch['subtree'][0]['subtree'][0]['local'][0]
            buffer_name = global_buffer['name']
            attributes = global_buffer['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            pe = arch['subtree'][0]['subtree'][0]['subtree'][0]
            num_pes = int(pe['name'].split('..')[1].split(']')[0]) + 1

            for buf in pe['local'][:-1]:
                buffer_name = buf['name']
                attributes = buf['attributes']
                depth = attributes['depth'] if 'depth' in attributes else float('Inf')
                word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
                width = attributes['width'] if 'width' in attributes else 8
                block_size = attributes['block-size'] if 'block-size' in attributes else 1
                buffer_size = depth * block_size
                instances = 1
                re_ret = re.search(r'.*\[', buffer_name)
                if re_ret:
                    instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                    buffer_name = re_ret.group(0)[:-1]
                instances *= num_pes
                buffer_name_list.append(buffer_name)
                buffer_size_list.append(buffer_size)
                num_instances.append(instances)
                num_buffer_levels += 1

            macc = pe['local'][-1]['name']
            instances = int(macc.split('..')[1].split(']')[0]) + 1
            instances *= num_pes
            num_instances.append(instances)
        elif 'Eyeriss' in self.arch_name:
            main_memory = arch['subtree'][0]['local'][0]
            buffer_name = main_memory['name']
            attributes = main_memory['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            global_buffer = arch['subtree'][0]['subtree'][0]['local'][0]
            buffer_name = global_buffer['name']
            attributes = global_buffer['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            dummy_buffer = arch['subtree'][0]['subtree'][0]['local'][1]
            buffer_name = dummy_buffer['name']
            attributes = dummy_buffer['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            pe = arch['subtree'][0]['subtree'][0]['subtree'][0]
            num_pes = int(pe['name'].split('..')[1].split(']')[0]) + 1

            for buf in pe['local'][:-1]:
                buffer_name = buf['name']
                attributes = buf['attributes']
                depth = attributes['depth'] if 'depth' in attributes else float('Inf')
                word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
                width = attributes['width'] if 'width' in attributes else 8
                block_size = attributes['block-size'] if 'block-size' in attributes else 1
                buffer_size = depth * block_size
                instances = 1
                re_ret = re.search(r'.*\[', buffer_name)
                if re_ret:
                    instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                    buffer_name = re_ret.group(0)[:-1]
                instances *= num_pes
                buffer_name_list.append(buffer_name)
                buffer_size_list.append(buffer_size)
                num_instances.append(instances)
                num_buffer_levels += 1

            macc = pe['local'][-1]['name']
            re_ret = re.search(r'.*\[', macc)
            if re_ret:
                instances = (int(macc.split('..')[1].split(']')[0]) + 1) * num_pes
            else:
                instances = num_pes
            num_instances.append(instances)
        elif 'TensorCore' in self.arch_name:
            main_memory = arch['subtree'][0]
            buffer_name = main_memory['name']
            attributes = main_memory['local'][0]['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
            buffer_name = main_memory['local'][0]['name']
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            global_buffer = arch['subtree'][0]['subtree'][0]
            buffer_name = global_buffer['name']
            attributes = global_buffer['local'][0]['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances *= int(buffer_name.split('..')[1].split(']')[0]) + 1
            buffer_name = global_buffer['local'][0]['name']
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            local_buffer = arch['subtree'][0]['subtree'][0]['subtree'][0]
            buffer_name = local_buffer['name']
            attributes = local_buffer['local'][0]['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances *= int(buffer_name.split('..')[1].split(']')[0]) + 1
            buffer_name = local_buffer['local'][0]['name']
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            pe_buffer = arch['subtree'][0]['subtree'][0]['subtree'][0]['subtree'][0]
            buffer_name = pe_buffer['name']
            attributes = pe_buffer['local'][0]['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            re_ret = re.search(r'.*\[', buffer_name)
            if re_ret:
                instances *= int(buffer_name.split('..')[1].split(']')[0]) + 1
            buffer_name = pe_buffer['local'][0]['name']
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1
            num_pes = instances

            macc = arch['subtree'][0]['subtree'][0]['subtree'][0]['subtree'][0]['local'][1]['name']
            re_ret = re.search(r'.*\[', macc)
            if re_ret:
                instances *= (int(macc.split('..')[1].split(']')[0]) + 1)
            num_instances.append(instances)

        print(buffer_name_list, num_instances, buffer_size_list)

        sp_cstr = []
        for i in range(len(num_instances) - 1):
            allowed_sp_size = num_instances[i + 1] // num_instances[i]
            sp_cstr.append(allowed_sp_size)
            if num_instances[i + 1] % num_instances[i] != 0:
                raise ValueError('Invalid Architecture File. '
                                 'Buffer hierarchy not perfectly divisible.')

        print(sp_cstr)

        return {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_name_list)}, \
               {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_size_list)}, \
               {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), sp_cstr)}, \
               num_buffer_levels


class Prob:
    def __init__(self, prob_path):
        self.path = prob_path.resolve()
        self.problem = utils.parse_yaml(self.path)
        self.density = {'Inputs': 0.5, 'Weights': 1, 'Outputs': 1}
        self.dim2note = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'H', 7: 'N'}
        self.note2dim = {'R': 0, 'S': 1, 'P': 2, 'Q': 3, 'C': 4, 'K': 5, 'H': 6, 'N': 7}

    def get_tensor_dimensions(self):
        tensor_dimensions = {}

        # 遍历数据空间
        for data_space in self.problem['problem']['shape']['data_spaces']:
            tensor_name = data_space['name']
            projection = data_space['projection']
            involved_dimensions = []

            # 递归函数用于提取维度
            def extract_dimensions(proj):
                if isinstance(proj, list):
                    for item in proj:
                        extract_dimensions(item)
                elif isinstance(proj, str) and proj in self.dim2note.values():
                    involved_dimensions.append(self.note2dim[proj])

            extract_dimensions(projection)
            # 去除重复维度
            unique_dimensions = list(set(involved_dimensions))
            tensor_dimensions[tensor_name] = unique_dimensions

        return tensor_dimensions
    
    def get_problem_info(self):
        problem = copy.deepcopy(self.problem)
        dimension = []
        dimension_dicts = {}
        for key in self.dim2note.values():
            value = problem['problem']['instance'][key]
            dimension.append(value)
            dimension_dicts[key] = value
        return dimension, dimension_dicts

    def config_str(self):
        """Returnsthe key str name for representing a unique layer."""
        val_arr = []
        for value in self.prob_bound:
            val_arr.append(str(value))
        keys = ['Wstride', 'Hstride', 'Wdilation', 'Hdilation']
        val_arr.extend([str(self.prob[key]) for key in keys])
        val_str = "_".join(val_arr)
        return val_str

    def print(self):
        print(self.__dict__)

class Mapping:
    def __init__(self, mapping_data):
        if isinstance(mapping_data, str):
            # 如果输入是字符串，认为是文件路径，解析 YAML 文件
            mapping_dict = utils.parse_yaml(mapping_data)
        elif isinstance(mapping_data, dict):
            # 如果输入是字典，直接使用
            mapping_dict = mapping_data
        else:
            raise ValueError("输入必须是文件路径（字符串）或字典。")
        self.mapping = mapping_dict['mapping']
        self.items   = [
                            item for item in self.mapping
                            if all(key in item for key in ['factors', 'permutation', 'target', 'type'])
                        ]
        
        # 按target分组
        self.target_groups = {}
        for item in self.items:
            target = item['target']
            if target not in self.target_groups:
                self.target_groups[target] = []
            self.target_groups[target].append(item)
        # 对每个target组内的元素按照type排序，spatial在前，temporal在后
        for target in self.target_groups:
            self.target_groups[target].sort(key=lambda x: 0 if x['type'] == 'spatial' else 1)

        self.factor_dict = self.get_factors()
        self.permutation_list = self.get_permutation()
        self.target_list = self.get_target()
        self.type_list = self.get_type()
        self.bypass_list = [
                            item for item in self.mapping
                            if all(key in item for key in ['bypass', 'keep', 'target', 'type'])
                        ]

    def get_factors(self):
        factors_dict = {'R': [], 'S': [], 'P': [], 'Q': [], 'C': [], 'K': [], 'N': [], 'H': []}
        
        for group in self.target_groups.values():
            for item in group:
                factors_str = item['factors']
                # 分割字符串并提取 R 的值
                for factor in factors_str.split():
                    if factor.startswith('R='):
                        r_value = factor.split('=')[1]  # 获取 R 的值
                        factors_dict['R'].append(int(r_value))  # 转换为整数并添加到列表中
                    elif factor.startswith('S='):
                        s_value = factor.split('=')[1]  
                        factors_dict['S'].append(int(s_value)) 
                    elif factor.startswith('P='):
                        p_value = factor.split('=')[1]  
                        factors_dict['P'].append(int(p_value)) 
                    elif factor.startswith('Q='):
                        q_value = factor.split('=')[1]  
                        factors_dict['Q'].append(int(q_value)) 
                    elif factor.startswith('C='):
                        c_value = factor.split('=')[1]  
                        factors_dict['C'].append(int(c_value)) 
                    elif factor.startswith('K='):
                        k_value = factor.split('=')[1]  
                        factors_dict['K'].append(int(k_value))
                    elif factor.startswith('N='):
                        n_value = factor.split('=')[1]  
                        factors_dict['N'].append(int(n_value))  
                    elif factor.startswith('H='):
                        n_value = factor.split('=')[1]  
                        factors_dict['H'].append(int(n_value))  
            
        return factors_dict

    def get_permutation(self):
        permutation_list = [item['permutation'] for group in self.target_groups.values() for item in group]
        return permutation_list
    
    def get_target(self):
        target_list = [item['target'] for group in self.target_groups.values() for item in group]
        return target_list

    def get_type(self):
        type_list = [item['type'] for group in self.target_groups.values() for item in group]
        return type_list

class Mapspace:
    def __init__(self, mapspace_path):
        self.mapspace_dict = utils.parse_yaml(mapspace_path)

        buffer_tensor_dict = {}
        bypass_data = []
        # 分析每个存储层次保存的张量
        for constraint in self.mapspace_dict['mapspace']['constraints']:
            target = constraint['target']
            keep_tensors = constraint.get('keep', [])
            if keep_tensors is None:  # 处理 keep: - 解析为 None 的情况
                keep_tensors = []
            buffer_tensor_dict[target] = keep_tensors
            # 过滤掉目标为 'DRAM' 的约束
            if constraint['target'] != 'DRAM':
                item = {
                    'bypass': constraint['bypass'],
                    'keep': keep_tensors,
                    'target': constraint['target'],
                    'type': constraint['type']
                }
                bypass_data.append(item)
            
        self.buffer_tensor_dict = buffer_tensor_dict
        self.bypass = bypass_data


if __name__ == "__main__":
    arch_path = pathlib.Path('../SpatialAccelerators/Simba/arch_v1.yaml').resolve()
    simba = Arch(arch_path)