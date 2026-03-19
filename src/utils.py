import os
import logging
import yaml
import math
from functools import reduce
from collections import defaultdict, OrderedDict
import subprocess
import random
import numpy as np

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
                level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

def run_timeloop(arch, prob, mapp, cwd=os.getcwd(), stdout=None, stderr=None):
    try:
        # p = subprocess.check_call(['/home/mingchuan/Desktop/zousheng/accelergy-timeloop-infrastructure/src/timeloop/build/timeloop-model', str(arch), str(prob), str(mapp)], \
        #                             cwd=cwd, stdout=stdout, stderr=stderr)
        p = subprocess.check_call(['timeloop-model', str(arch), str(prob), str(mapp)], \
                                    cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info('run_timeloop> timeloop-model {} {} {} in {}'.format(arch, prob, mapp, cwd))
        return True
    except:
        return False

def run_cosa(output_path, arch_path, map_path, prob_path, cwd=os.getcwd(), stdout=None, stderr=None):
    try:
        p = subprocess.check_call(['cosa', '-o', str(output_path), '-ap', str(arch_path), '-mp', str(map_path), '-pp', str(prob_path)], \
                                    cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info('run_cosa> cosa -o {} -ap {} -mp {} -pp {}'.format(output_path, arch_path, map_path, prob_path))
        return True
    except:
        return False

def parse_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.full_load(f)
    return data

def store_yaml(yaml_path, data):
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)

def get_factors(n):
    return list(reduce(list.__add__,
                        ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

def get_prime_factors(num):
    i = 2
    factors = defaultdict(int)
    while i * i <= num:
        while num % i == 0:
            factors[i] += 1
            num //= i
        i += 1
    if num > 1:
        factors[num] += 1
    return factors

def generate_factors_list(m, n):
    if n < 1:
        print("n 必须大于等于1")
        return []
    
    if m == 0:
        print("m 为0时，所有因数必须为0")
        return [0] * n
    
    if m == 1:
        return [1] * n
    
    factors = get_prime_factors(m)

    result = [1] * n  # 初始化结果列表为 n 个1
    random.shuffle(factors)  # 随机打乱因数
    
    for prime, count in factors.items():
        for _ in range(count):
            index = random.randint(0, n-1)
            result[index] *= prime
    
    return result

def generate_similar_factor_list(pattern, target):
    # 计算模式中所有元素的乘积
    pattern_product = 1
    for num in pattern:
        pattern_product *= num
    # 计算目标值与模式乘积的比值
    ratio = target / pattern_product
    # 找到模式中最大元素的索引
    max_index = pattern.index(max(pattern))
    # 复制模式列表
    result = pattern.copy()
    # 调整最大元素的值
    result[max_index] = int(result[max_index] * ratio)
    # 重新计算调整后列表的乘积
    current_product = 1
    for num in result:
        current_product *= num
    # 如果乘积不等于目标值，尝试微调
    if current_product != target:
        # 尝试逐个调整元素来达到目标值
        for i in range(len(result)):
            if i != max_index:
                new_ratio = target / (current_product / result[i])
                result[i] = int(new_ratio)
                new_product = 1
                for num in result:
                    new_product *= num
                if new_product == target:
                    break
    return result

def run_cosa(output_path='../cosa_tmp', arth_path='../SpatialAccelerators/Cosa/Simba/arch.yaml', map_path='../SpatialAccelerators/Cosa/Simba/mapspace.yaml', prob_path='../SpatialAccelerators/Cosa/Simba/problem.yaml', cwd=os.getcwd(), stdout=None, stderr=None):
    try:
        p = subprocess.check_call(['cosa', '-o', str(output_path), '-ap', str(arth_path), '-mp', str(map_path), '-pp', str(prob_path)], \
                                  cwd=cwd, stdout=stdout, stderr=stderr)
        logger.info('run_cosa> cosa -o {} -ap {} -mp {} -pp {}'.format(output_path, arth_path, map_path, prob_path))
        return True
    except:
        return False

def generate_problem_for_cosa(prob_path, dimension_dict):
    dir_path = os.path.dirname(prob_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(prob_path, 'w') as yaml_file:
        yaml.dump(dimension_dict, yaml_file, default_flow_style=False)

def find_remainders(A, T):
    A = A.copy()
    n = len(A) - 1  # n is the number of b's
    # Calculate constant part
    constant_part = (A[0] - 1) * np.prod(A[1:])  # 计算常量部分
    while  T - constant_part <= 0:
        A[0] = A[0] - 1
        constant_part = (A[0] - 1) * np.prod(A[1:])

    required_sum = T - constant_part  # 需要的和
    results = []
    outermost_idx = -1
    # 使用递归穷举 b1, b2, ..., bn
    def backtrack(index, current_b):
        if index == n:  # 已经填充完 b1 到 bn
            # 计算当前和
            current_sum = sum((current_b[i] - 1) * np.prod(A[i + 2:]) for i in range(n-1)) + current_b[n-1]

            if current_sum == required_sum:
                results.append(current_b.copy()[::-1] + [A[0]])  # 存储满足条件的 b
            return 
        
        # 迭代 b_index 的可能值
        for b in range(1, A[index + 1] + 1):  # b的范围是从1到a_{i+1}
            current_b[index] = b
            backtrack(index + 1, current_b)


    # 开始回溯
    backtrack(0, [0] * n)
    if not results:
        print(f"Warning: No valid combination found for A={A}, T={T}, required_sum={required_sum}")
        return [], -1
    lst = results[0]
    for i in range(len(lst)):
        if lst[i] != 1:
            # 检查当前索引之后的所有元素是否都为1
            if all(x == 1 for x in lst[i+1:]):
                outermost_idx = i
    return results, outermost_idx


def find_remainders2(F, T):
    # 保证不修改原始 F
    A = F.copy()
    n = len(A) - 1  # 对应 b 的个数

    # 计算常量部分： (A[0]-1)*np.prod(A[1:])
    constant_part = (A[0] - 1) * np.prod(A[1:])
    # 保证 T - constant_part >0，否则调整 A[0]
    while T - constant_part <= 0:
        A[0] -= 1
        constant_part = (A[0] - 1) * np.prod(A[1:])

    required_sum = T - constant_part  # 需要的额外和
    results = []
    outermost_idx = -1

    # 预计算乘积数组 prod，其中：
    # 对于 0 <= i < n-1: prod[i] = np.prod(A[i+2:]) ；对于 i == n-1，我们记作1（因为最后一项直接累加 b_n）
    prod = [1] * n
    if n >= 1:
        prod[-1] = 1  # 对应 b_n 的权重为1
    for i in range(n-2, -1, -1):
        prod[i] = A[i+2] * prod[i+1]  # A[i+2:] 的乘积

    # 预计算剩余上界数组 max_remain[i] 表示从第 i 个 b 开始，
    # 能够贡献的最大值之和（用于剪枝）
    max_remain = [0] * (n + 1)
    max_remain[n] = 0
    # 对于 0 <= i <= n-2: 最大贡献为 (A[i+1]-1)*prod[i]（因为 b_i 的范围 1~A[i+1]）；对于 i == n-1: 最大贡献为 A[n]
    for i in range(n-1, -1, -1):
        if i == n-1:
            max_contrib = A[n]  # 最后一个 b 的最大贡献
        else:
            max_contrib = (A[i+1] - 1) * prod[i]
        max_remain[i] = max_contrib + max_remain[i+1]

    # 使用递归遍历 b1, b2, ..., bn，每一步累加贡献，并进行剪枝
    def backtrack(index, current_b, current_sum):
        # 使用剪枝：如果当前和加上后续所有索引可能贡献不足 required_sum，则无需继续
        if current_sum + max_remain[index] < required_sum:
            return

        if index == n:
            # 当填满所有 b 时判断累计和是否恰好等于目标
            if current_sum == required_sum:
                # 按原代码要求，存储顺序为 [b_n, ..., b_1, A[0]]
                results.append(current_b.copy()[::-1] + [A[0]])
            return

        # b 的取值范围：1 到 A[index+1]
        for b in range(1, A[index + 1] + 1):
            # 根据位置计算这一位的贡献：
            # 当 index < n-1，贡献为 (b-1)*prod[index]；
            # 当 index == n-1，贡献直接为 b（因为最后一项不减1，加权为1）。
            if index < n - 1:
                contribution = (b - 1) * prod[index]
            else:
                contribution = b

            new_sum = current_sum + contribution

            # 如果 new_sum 超过 required_sum，因为 b 递增所以可以提前 break
            if new_sum > required_sum:
                break

            current_b[index] = b
            backtrack(index + 1, current_b, new_sum)

    # 开始回溯：起始当前和为 0，current_b 用长度为 n 的数组保存当前解
    backtrack(0, [0] * n, 0)

    if not results:
        print(f"Warning: No valid combination found for A={A}, T={T}, required_sum={required_sum}")
        return [], -1

    # 查找 outermost_idx: 从左到右第一个不为 1 且其后全为 1 的位置
    # 由外层向内层
    lst = F
    for i in range(len(lst)):
        if lst[i] != 1:
            outermost_idx = i
            break
    # for i in range(len(lst)):
    #     if lst[i] != 1:
    #         if all(x == 1 for x in lst[i + 1:]):
    #             outermost_idx = i
    #             break

    return results, outermost_idx


def expand_factors(factors, target):

    product = 1
    for factor in factors:
        product *= factor
    if product == target:
        return factors, target
    
    # 复制因数列表，避免修改原始列表
    current_factors = factors.copy()
    current_factors.reverse()

    n = len(current_factors)
    # 找到第一个不为1的因数的索引
    start_index = 0
    # for i, factor in enumerate(current_factors):
    #     if factor != 1:
    #         start_index = i
    #         break

    # 从第一个不为1的因数开始扩展，保证乘积不小于目标值
    product = 1
    for factor in current_factors:
        product *= factor
    while product < target:
        current_factors[start_index] += 1
        product = 1
        for factor in current_factors:
            product *= factor

    # 逐级往后缩减
    index = start_index
    while True:
        while product > target:
            current_factors[index] -= 1
            product = 1
            for factor in current_factors:
                product *= factor
            if product < target:
                current_factors[index] += 1
                product = 1
                for factor in current_factors:
                    product *= factor
                break

        index += 1
        if index == n:
            break

        # 重新扩展当前位置的因数，保证乘积不小于目标值
        while product < target:
            current_factors[index] += 1
            product = 1
            for factor in current_factors:
                product *= factor

    product = 1
    for factor in current_factors:
        product *= factor

    current_factors.reverse()
    return current_factors, product

def generate_mapping_for_lpsolver(solution, dimension, targets, types, bypass_data):
    permutations = ['RSPQCKHN'] * len(targets)
    
    rounded_array = [[math.floor(num) for num in row] for row in solution]
    
    # 按列提取出来
    extracted_columns = [list(col) for col in zip(*rounded_array)]

    factors_list = []
    product_list = []
    for i in range(len(dimension)):
        factors, product = expand_factors(extracted_columns[i], dimension[i])
        factors_list.append(factors)
        product_list.append(product)

    factors_dict = {}
    factors_dict['R'] = factors_list[0]
    factors_dict['S'] = factors_list[1]
    factors_dict['P'] = factors_list[2]
    factors_dict['Q'] = factors_list[3]
    factors_dict['C'] = factors_list[4]
    factors_dict['K'] = factors_list[5]
    factors_dict['H'] = factors_list[6]
    factors_dict['N'] = factors_list[7]
    dimension_dict = {}
    dimension_dict['R'] = product_list[0]
    dimension_dict['S'] = product_list[1]
    dimension_dict['P'] = product_list[2]
    dimension_dict['Q'] = product_list[3]
    dimension_dict['C'] = product_list[4]
    dimension_dict['K'] = product_list[5]
    dimension_dict['H'] = product_list[6]
    dimension_dict['N'] = product_list[7]

    mapping = generate_mapping(factors_dict, permutations, targets, types, bypass_data)
    

    return mapping, dimension_dict

def generate_mapping_for_lpsolver2(solution, targets, types, bypass_data):
    permutations = ['RSPQCKHN'] * len(targets)
    
    rounded_array = [[math.floor(num) for num in row] for row in solution]
    
    # 按列提取出来
    extracted_columns = [list(col) for col in zip(*rounded_array)]

    factors_dict = {}
    factors_dict['R'] = extracted_columns[0]
    factors_dict['S'] = extracted_columns[1]
    factors_dict['P'] = extracted_columns[2]
    factors_dict['Q'] = extracted_columns[3]
    factors_dict['C'] = extracted_columns[4]
    factors_dict['K'] = extracted_columns[5]
    factors_dict['H'] = extracted_columns[6]
    factors_dict['N'] = extracted_columns[7]
    dimension_dict = {}
    dimension_dict['R'] = np.prod(factors_dict['R'])
    dimension_dict['S'] = np.prod(factors_dict['S'])
    dimension_dict['P'] = np.prod(factors_dict['P'])
    dimension_dict['Q'] = np.prod(factors_dict['Q'])
    dimension_dict['C'] = np.prod(factors_dict['C'])
    dimension_dict['K'] = np.prod(factors_dict['K'])
    dimension_dict['H'] = np.prod(factors_dict['H'])
    dimension_dict['N'] = np.prod(factors_dict['N'])

    mapping = generate_mapping(factors_dict, permutations, targets, types, bypass_data)
    

    return mapping, dimension_dict

def generate_mapping(factors, permutations, targets, types, bypass_data, remainders={}, outermost_idx={}):
    # 构建结果列表
    mapping = {'mapping': []}
    n = len(permutations)
    
    raw_factors = {'R': [], 'S': [], 'P': [], 'Q': [], 'C': [], 'K': [], 'H': [], 'N': [], }
    # 遍历将因数、排列、目标和类型组合成字典
    for i in range(len(permutations)-1, -1, -1):
        # 获取当前的因数组合
        current_permutation = permutations[i]
        current_target = targets[i]
        current_type = types[i]
        
        # 获取因数的值
        factors_values = {}
        for char in current_permutation:
            factors_values[char] = factors[char][i] if len(factors[char]) > 0 else 0
        # 组合成字符串形式
        factors_str_parts = []  # 使用列表来暂存每个部分
        for key in factors_values:
            all_ones = False
            if len(raw_factors[key]) == n-i-1 and all(v == 1 for v in raw_factors[key]):
                all_ones = True
            if key in remainders  and remainders[key] != [] and remainders[key][0][i] != factors_values[key]:
                if i == (len(permutations) - 1 - outermost_idx[key]):
                    factors_str_parts.append(f'{key}={remainders[key][0][i]}')
                    raw_factors[key].append(remainders[key][0][i])
                elif all_ones:
                    factors_str_parts.append(f'{key}={remainders[key][0][i]}')
                    raw_factors[key].append(remainders[key][0][i])
                else:
                    # 添加格式化字符串到列表
                    factors_str_parts.append(f'{key}={factors_values[key]},{remainders[key][0][i]}')
            else:
                factors_str_parts.append(f'{key}={factors_values[key]}')
                raw_factors[key].append(factors_values[key])

        # 使用 join 方法将所有部分合并并去除末尾的空格
        factors_str = ' '.join(factors_str_parts)
        # 添加到结果中
        mapping['mapping'].append({
            'factors': factors_str,
            'permutation': current_permutation,
            'target': current_target,
            'type': current_type
        })

    mapping['mapping'].reverse()

    # 添加旁路信息
    mapping['mapping'].extend(bypass_data)

    # 返回结果
    return mapping

def parse_timeloop_output(file_path):
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    in_summary_section = False
    summary = {}
    for line in lines:
        line = line.strip()
        
        # 检测是否进入 Summary Stats 部分
        if line.startswith("Summary Stats"):
            in_summary_section = True
        
        # 若在 Summary Stats 中，查找 Cycles
        if in_summary_section:
            if line.startswith("Cycles:"):
                cycles = float(line.split(":")[1].strip())  # 提取 cycles 值
                summary['cycles'] = cycles
            elif line.startswith("Energy:"):
                energy_str = line.split(':')[1].strip().replace('uJ', '')  # 去掉 uJ
                energy = float(energy_str)  # 转换为浮点数
                summary['energy'] = energy
            elif line.startswith("Utilization:"):
                utilization_str = line.split(':')[1].strip().replace('%', '')  # 去掉百分号
                utilization = float(utilization_str) / 100  # 转换为小数
                summary['utilization'] = utilization
            elif line.startswith("EDP(J*cycle):"):
                EDP = float(line.split(":")[1].strip())  
                summary['EDP'] = EDP
            elif line.startswith("GFLOPs (@1GHz)"):
                GFLOPs = float(line.split(":")[1].strip())  
                summary['GFLOPs'] = GFLOPs
    
    return summary

def generate_arch(accelerator, config):
    # 目前仅处理simba
    arch_dict = {'arch': 
     {'arithmetic': {'instances': 1024, 'word-bits': 8}, 
      'storage': [{'name': 'Registers', 'entries': 1, 'instances': 1024, 'word-bits': 8, 'cluster-size': 1, 'num-ports': 2, 'num-banks': 8}, 
                  {'name': 'AccumulationBuffer', 'entries': 3072, 'instances': 16, 'word-bits': 24, 'cluster-size': 1, 'network-word-bits': 16, 'num-ports': 2, 'num-banks': 2}, 
                  {'name': 'WeightBuffer', 'entries': 32768, 'instances': 16, 'word-bits': 8, 'block-size': 4, 'num-ports': 1, 'num-banks': 8}, 
                  {'name': 'InputBuffer', 'entries': 8192, 'instances': 16, 'word-bits': 8, 'block-size': 4, 'num-ports': 2, 'num-banks': 1}, 
                  {'name': 'GlobalBuffer', 'entries': 65536, 'instances': 1, 'word-bits': 8, 'block-size': 8, 'num-ports': 2, 'num-banks': 256}, 
                  {'name': 'DRAM', 'technology': 'DRAM', 'instances': 1, 'word-bits': 8, 'block-size': 64, 'bandwidth': 1}]}}

    # 并行容量：通过 .item() 将 NumPy 标量转为 Python 原生类型（如 int）
    arch_dict['arch']['arithmetic']['instances'] = (config[0] * config[1]).item()
    arch_dict['arch']['storage'][0]['instances'] = (config[0] * config[1]).item()
    arch_dict['arch']['storage'][1]['instances'] = config[1].item()
    arch_dict['arch']['storage'][2]['instances'] = config[1].item()
    arch_dict['arch']['storage'][3]['instances'] = config[1].item()
    arch_dict['arch']['storage'][4]['instances'] = 1  # 原生int，无需转换
    
    # 存储容量：同样用 .item() 转换 NumPy 标量
    arch_dict['arch']['storage'][0]['entries'] = config[2].item()
    arch_dict['arch']['storage'][1]['entries'] = config[3].item()
    arch_dict['arch']['storage'][2]['entries'] = config[4].item()
    arch_dict['arch']['storage'][3]['entries'] = config[5].item()
    arch_dict['arch']['storage'][4]['entries'] = config[6].item()
    
    return arch_dict

def get_tensor_dimensions(problem):
    tensor_dimensions = {}
    dim2note = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'H', 7: 'N'}
    note2dim = {'R': 0, 'S': 1, 'P': 2, 'Q': 3, 'C': 4, 'K': 5, 'H': 6, 'N': 7}
    
    # 遍历数据空间
    for data_space in problem['problem']['shape']['data_spaces']:
        tensor_name = data_space['name']
        projection = data_space['projection']
        involved_dimensions = []

        # 递归函数用于提取维度
        def extract_dimensions(proj):
            if isinstance(proj, list):
                for item in proj:
                    extract_dimensions(item)
            elif isinstance(proj, str) and proj in dim2note.values():
                involved_dimensions.append(note2dim[proj])

        extract_dimensions(projection)
        # 去除重复维度
        unique_dimensions = list(set(involved_dimensions))
        tensor_dimensions[tensor_name] = unique_dimensions

    return tensor_dimensions

def generate_accelerator(config, hardware):
    arch_dict = {}
    spatial_capacity = config[0]
    buffer_capacity = config[1]
    if hardware.name == "Simba":
        arch_dict = {'arch': 
            {'arithmetic': {'instances': 1024, 'word-bits': 8}, 
            'storage': [{'name': 'Registers', 'entries': 1, 'instances': 1024, 'word-bits': 8, 'cluster-size': 1, 'num-ports': 2, 'num-banks': 8}, 
                        {'name': 'AccumulationBuffer', 'entries': 3072, 'instances': 16, 'word-bits': 24, 'cluster-size': 1, 'network-word-bits': 16, 'num-ports': 2, 'num-banks': 2}, 
                        {'name': 'WeightBuffer', 'entries': 32768, 'instances': 16, 'word-bits': 8, 'block-size': 4, 'num-ports': 1, 'num-banks': 8}, 
                        {'name': 'InputBuffer', 'entries': 8192, 'instances': 16, 'word-bits': 8, 'block-size': 4, 'num-ports': 2, 'num-banks': 1}, 
                        {'name': 'GlobalBuffer', 'entries': 65536, 'instances': 1, 'word-bits': 8, 'block-size': 8, 'num-ports': 2, 'num-banks': 256}, 
                        {'name': 'DRAM', 'technology': 'DRAM', 'instances': 1, 'word-bits': 8, 'block-size': 64, 'bandwidth': 1}]}}
        
        # 并行容量：通过 .item() 将 NumPy 标量转为 Python 原生类型（如 int）
        arch_dict['arch']['arithmetic']['instances'] = (spatial_capacity[1] * spatial_capacity[2] * spatial_capacity[3] * spatial_capacity[4] * spatial_capacity[5]).item()
        arch_dict['arch']['storage'][0]['instances'] = (spatial_capacity[1] * spatial_capacity[2] * spatial_capacity[3] * spatial_capacity[4] * spatial_capacity[5]).item()
        arch_dict['arch']['storage'][1]['instances'] = (spatial_capacity[2] * spatial_capacity[3] * spatial_capacity[4] * spatial_capacity[5]).item()
        arch_dict['arch']['storage'][2]['instances'] = (spatial_capacity[3] * spatial_capacity[4] * spatial_capacity[5]).item()
        arch_dict['arch']['storage'][3]['instances'] = (spatial_capacity[4] * spatial_capacity[5]).item()
        arch_dict['arch']['storage'][4]['instances'] = (spatial_capacity[5]).item()
        arch_dict['arch']['storage'][5]['instances'] = 1  

        assert spatial_capacity[0] == 1
        assert arch_dict['arch']['storage'][4]['instances'] == 1
        
        # 存储容量：同样用 .item() 转换 NumPy 标量
        arch_dict['arch']['storage'][0]['entries'] = buffer_capacity[0].item()
        arch_dict['arch']['storage'][1]['entries'] = buffer_capacity[1].item()
        arch_dict['arch']['storage'][2]['entries'] = buffer_capacity[2].item()
        arch_dict['arch']['storage'][3]['entries'] = buffer_capacity[3].item()
        arch_dict['arch']['storage'][4]['entries'] = buffer_capacity[4].item()
    elif hardware.name == "Gemmini":
        arch_dict = {'arch':
                     {'arithmetic': {'instances': 128, 'word-bits': 8}, 
                    'storage': [{'name': 'Registers', 'entries': 1, 'instances': 128, 'word-bits': 8, 'n_rdwr_ports': 2, 'n_banks': 1}, 
                                {'name': 'Accumulator', 'entries': 4096, 'depth': 1024, 'width': 512, 'instances': 16, 'word-bits': 32, 'network-word-bits': 16, 'n_rdwr_ports': 2, 'n_banks': 2}, 
                                {'name': 'Scratchpad', 'class': 'SRAM', 'entries': 8196, 'depth': 16384, 'width': 128, 'instances': 1, 'word-bits': 8, 'n_rdwr_ports': 1, 'n_banks': 4}, 
                                {'name': 'DRAM', 'technology': 'DRAM', 'instances': 1, 'word-bits': 8, 'block_size': 64, 'bandwidth': 5.75}]}}

        arch_dict['arch']['arithmetic']['instances'] = (spatial_capacity[1] * spatial_capacity[2]).item()
        arch_dict['arch']['storage'][0]['instances'] = (spatial_capacity[1] * spatial_capacity[2]).item()
        arch_dict['arch']['storage'][1]['instances'] = (spatial_capacity[2]).item()
        arch_dict['arch']['storage'][2]['instances'] = 1
        arch_dict['arch']['storage'][3]['instances'] = 1


        assert spatial_capacity[0] == 1
        assert arch_dict['arch']['storage'][2]['instances'] == 1
        
        # 存储容量：同样用 .item() 转换 NumPy 标量
        arch_dict['arch']['storage'][0]['entries'] = buffer_capacity[0].item()
        arch_dict['arch']['storage'][1]['entries'] = buffer_capacity[1].item()
        arch_dict['arch']['storage'][2]['entries'] = buffer_capacity[2].item()
    elif hardware.name == "NEW":
        arch_dict = {'arch':
                     {'arithmetic': {'instances': 128, 'word-bits': 8}, 
                    'storage': [{'name': 'Registers', 'entries': 1, 'instances': 128, 'word-bits': 8, 'n_rdwr_ports': 2, 'n_banks': 1}, 
                                {'name': 'InputBuffer', 'entries': 4096, 'depth': 1024, 'width': 512, 'instances': 16, 'word-bits': 32, 'network-word-bits': 16, 'n_rdwr_ports': 2, 'n_banks': 2},
                                {'name': 'Accumulator', 'entries': 4096, 'depth': 1024, 'width': 512, 'instances': 16, 'word-bits': 32, 'network-word-bits': 16, 'n_rdwr_ports': 2, 'n_banks': 2},  
                                {'name': 'Scratchpad', 'class': 'SRAM', 'entries': 8196, 'depth': 16384, 'width': 128, 'instances': 1, 'word-bits': 8, 'n_rdwr_ports': 1, 'n_banks': 4}, 
                                {'name': 'DRAM', 'technology': 'DRAM', 'instances': 1, 'word-bits': 8, 'block_size': 64, 'bandwidth': 5.75}]}}

        arch_dict['arch']['arithmetic']['instances'] = (spatial_capacity[1] * spatial_capacity[2]*spatial_capacity[3]).item()
        arch_dict['arch']['storage'][0]['instances'] = (spatial_capacity[1] * spatial_capacity[2]*spatial_capacity[3]).item()
        arch_dict['arch']['storage'][1]['instances'] = (spatial_capacity[2] * spatial_capacity[3]).item()
        arch_dict['arch']['storage'][2]['instances'] = (spatial_capacity[3]).item()
        arch_dict['arch']['storage'][3]['instances'] = 1
        arch_dict['arch']['storage'][4]['instances'] = 1


        assert spatial_capacity[0] == 1
        # assert arch_dict['arch']['storage'][2]['instances'] == 1
        
        # 存储容量：同样用 .item() 转换 NumPy 标量
        arch_dict['arch']['storage'][0]['entries'] = buffer_capacity[0].item()
        arch_dict['arch']['storage'][1]['entries'] = buffer_capacity[1].item()
        arch_dict['arch']['storage'][2]['entries'] = buffer_capacity[2].item()
        arch_dict['arch']['storage'][3]['entries'] = buffer_capacity[3].item()

    else:
        print("目前不支持", hardware.name)

    return arch_dict

def generate_mapping_for_lpsolver3(solution, p):
    non_all_ones = []
    spatial_tile = solution[0]
    temporal_tile = solution[1]
    for idx, row in enumerate(spatial_tile):
        # 检查行中是否存在非1的元素
        if isinstance(row, np.ndarray):
            # 对 numpy 数组使用 .all() 方法
            if not (row == 1).all():
                non_all_ones.append(idx)
        else:
            # 对普通列表使用内置 all()
            if not all(element == 1 for element in row):
                non_all_ones.append(idx)
    buffer_hierarchy = p.buffer_hierarchy
    targets = []
    tile_type = []
    tiles = []
    for i in range(len(buffer_hierarchy)):
        if i in non_all_ones:
            targets.append(buffer_hierarchy[i])
            tile_type.append('spatial')
            tiles.append(spatial_tile[i])
        targets.append(buffer_hierarchy[i])
        tile_type.append('temporal')
        tiles.append(temporal_tile[i])
    
    permutations = ['RSPQCKHN'] * len(targets)
    tensor_in_buffer = p.tensor_in_buffer
    bypass = []
    # 去掉DRAM
    for i in range(len(buffer_hierarchy)-1):
        item = {}
        item['keep'] = tensor_in_buffer[buffer_hierarchy[i]]
        item['bypass'] = [x for x in ['Weights','Inputs','Outputs'] if x not in item['keep']]
        item['target'] = buffer_hierarchy[i]
        item['type'] = 'datatype'
        bypass.append(item)

    rounded_array = [[math.floor(num) for num in row] for row in tiles]
  
    # 按列提取出来
    extracted_columns = [list(col) for col in zip(*rounded_array)]
    
    factors_dict = {}
    factors_dict['R'] = extracted_columns[0]
    factors_dict['S'] = extracted_columns[1]
    factors_dict['P'] = extracted_columns[2]
    factors_dict['Q'] = extracted_columns[3]
    factors_dict['C'] = extracted_columns[4]
    factors_dict['K'] = extracted_columns[5]
    factors_dict['H'] = extracted_columns[6]
    factors_dict['N'] = extracted_columns[7]

    mapping = generate_mapping(factors_dict, permutations, targets, tile_type, bypass)

    return mapping
    

if __name__ == "__main__":
    print(find_remainders([7, 2, 2, 6, 5], 699))
    print(find_remainders2([1, 1, 1, 128, 2, 1], 192))
    factor = {'P': [2,2,2,1,1,1]}
    permutations = ['P', 'P', 'P', 'P', 'P', 'P']
    targets = ['temporal', 'temporal', 'temporal', 'temporal', 'temporal', 'temporal']
    types = ['temporal', 'temporal', 'temporal', 'temporal', 'temporal', 'temporal']
    bypass = []
    remainders ={'P': [[1, 1, 1, 1, 1, 1]]}
    outermost_idx = {'P': 3}

    mapping = generate_mapping(factor, permutations, targets, types, bypass, remainders, outermost_idx)
    print(mapping)