import argparse
import yaml
import os
import pickle

from explorer2 import MappingExplorer
from datetime import datetime
import json
from hardware_first import HardwareFirstSearch
from mapping_first import MappingFirstSearch

def main():
    # current_time = datetime.now()
    # print("当前时间：", current_time)  # 输出示例：2023-06-15 14:30:25.123456
    opt                        = parser.parse_args()
    benchmark_dir              = '../Benchmarks'
    accelerator_dir            = '../SpatialAccelerators'
    config_dir                 = '../config'
    accelerator                = opt.accelerator
    mapper                     = opt.mapper
    type                       = opt.type
    version                    = opt.version
    workload                   = opt.workload
    layer_id                   = opt.layer_id
    expanded_scope             = opt.expanded_scope
    expanded_config            = opt.expanded_config
    batch_size                 = opt.batch_size
    parameter_dimension        = opt.parameter_dimension
    weight_matrix_config       = opt.weight_matrix_config
    solver                     = opt.solver
    all_DNN                    = opt.all_DNN
    weight_num                 = opt.weight_num
    all_workload               = opt.all_workload
    search_mode                = opt.search_mode
    
    if all_DNN:
        if all_workload:
            layer_list = []
            layer_name = []
            with open(os.path.join(benchmark_dir, 'bert_workload/layers.yaml'.format(workload)), 'r') as fd:
                layers1 = yaml.load(fd, Loader=yaml.SafeLoader)
                layer_list.append(layers1)
                layer_name.append('bert')
            
            with open(os.path.join(benchmark_dir, 'unet_workload/layers.yaml'.format(workload)), 'r') as fd:
                layers2 = yaml.load(fd, Loader=yaml.SafeLoader)
                layer_list.append(layers2)
                layer_name.append('unet')

            with open(os.path.join(benchmark_dir, 'resnet50_workload/layers.yaml'.format(workload)), 'r') as fd:
                layers3 = yaml.load(fd, Loader=yaml.SafeLoader)
                layer_list.append(layers3)
                layer_name.append('resnet50')

            with open(os.path.join(benchmark_dir, 'retinanet_workload/layers.yaml'.format(workload)), 'r') as fd:
                layers4 = yaml.load(fd, Loader=yaml.SafeLoader)
                layer_list.append(layers4)
                layer_name.append('retinanet')

            with open(os.path.join(benchmark_dir, 'alexnet_workload/layers.yaml'.format(workload)), 'r') as fd:
                layers5 = yaml.load(fd, Loader=yaml.SafeLoader)
                layer_list.append(layers5)
                layer_name.append('alexnet')

            with open(os.path.join(benchmark_dir, 'vgg16_workload/layers.yaml'.format(workload)), 'r') as fd:
                layers6 = yaml.load(fd, Loader=yaml.SafeLoader)
                layer_list.append(layers6)
                layer_name.append('vgg16')

            with open(os.path.join(benchmark_dir, 'llama2_7B_workload/layers.yaml'.format(workload)), 'r') as fd:
                layers7 = yaml.load(fd, Loader=yaml.SafeLoader)
                layer_list.append(layers7)
                layer_name.append('llama2_7B')

            with open(os.path.join(benchmark_dir, 'llama2_13B_workload/layers.yaml'.format(workload)), 'r') as fd:
                layers8 = yaml.load(fd, Loader=yaml.SafeLoader)
                layer_list.append(layers8)
                layer_name.append('llama2_13B')

            with open(os.path.join(benchmark_dir, 'llama2_34B_workload/layers.yaml'.format(workload)), 'r') as fd:
                layers9 = yaml.load(fd, Loader=yaml.SafeLoader)
                layer_list.append(layers9)
                layer_name.append('llama2_34B')
        
        
            report_dir = os.path.join(opt.report_dir, '{}'.format(mapper), '{}'.format(accelerator),'obj_{}'.format(opt.optim_obj),
                                    'all_input{}_all_DNN'.format(workload, batch_size))
            os.makedirs(report_dir, exist_ok=True)
            problems_list = []
            for i in range(len(layer_list)):
                for layer_id in range(len(layer_list[i])):
                    layer = layer_list[i][layer_id]
                    with open(os.path.join(benchmark_dir, '{}_workload/{}.yaml'.format(layer_name[i], layer)), 'r') as fd:
                        layer_problem = yaml.load(fd, Loader=yaml.SafeLoader)
                        problem = {'problem': {
                            'shape': {'name': 'CNN_Layer', 'dimensions': ['H', 'C', 'K', 'R', 'S', 'N', 'P', 'Q'],
                                    'coefficients': [{'name': 'Wstride', 'default': 1},
                                                    {'name': 'Hstride', 'default': 1},
                                                    {'name': 'Wdilation', 'default': 1},
                                                    {'name': 'Hdilation', 'default': 1}],
                                    },
                            'instance': {'C': 256, 'K': 512, 'R': 3, 'S': 3, 'P': 56, 'Q': 56, 'H': 1, 'N': 16,
                                        'Wstride': 1, 'Hstride': 1, 'Wdilation': 1, 'Hdilation': 1
                                        }}}
                        if 'type' in layer_problem['problem'].keys() and layer_problem['problem']['type'] == 'T2D':
                            problem['problem']['shape']['data_spaces'] = [
                                            {'name': 'Weights',
                                            'projection': [[['H']], [['C']], [['K']], [['R']], [['S']]]},
                                            {'name': 'Outputs', 'projection': [[['N']], [['H']], [['K']],
                                                                                [['R', 'Wdilation'],
                                                                                ['P', 'Wstride']],
                                                                                [['S', 'Hdilation'],
                                                                                ['Q', 'Hstride']]],
                                            'read_write': True},
                                            {'name': 'Inputs', 'projection': [[['N']], [['H']], [['C']], [['Q']], [['P']]]}]
                            problem['problem']['instance']['type'] = 'T2D'
                        else:
                            problem['problem']['shape']['data_spaces'] = [
                                            {'name': 'Weights',
                                            'projection': [[['H']], [['C']], [['K']], [['R']], [['S']]]},
                                            {'name': 'Inputs', 'projection': [[['N']], [['H']], [['C']],
                                                                                [['R', 'Wdilation'],
                                                                                ['P', 'Wstride']],
                                                                                [['S', 'Hdilation'],
                                                                                ['Q', 'Hstride']]]},
                                            {'name': 'Outputs',
                                            'projection': [[['N']], [['H']], [['K']], [['Q']], [['P']]],
                                            'read_write': True}]
                            problem['problem']['instance']['type'] = 'C2D'
                        if 'H' in layer_problem['problem'].keys():
                            problem['problem']['instance']['H'] = layer_problem['problem']['H']
                        else:
                            problem['problem']['instance']['H'] = 1
                        if 'type' in layer_problem['problem'].keys() and layer_problem['problem']['type'] == 'BMM':
                            problem['problem']['instance']['N'] = layer_problem['problem']['N']
                            problem['problem']['instance']['H'] = layer_problem['problem']['H'] * batch_size
                        else:
                            problem['problem']['instance']['N'] = layer_problem['problem']['N'] * batch_size
                        problem['problem']['instance']['K'] = layer_problem['problem']['K']
                        problem['problem']['instance']['C'] = layer_problem['problem']['C']
                        problem['problem']['instance']['P'] = layer_problem['problem']['P']
                        problem['problem']['instance']['Q'] = layer_problem['problem']['Q']
                        problem['problem']['instance']['R'] = layer_problem['problem']['R']
                        problem['problem']['instance']['S'] = layer_problem['problem']['S']
                        problem['problem']['instance']['Wstride'] = layer_problem['problem']['Wstride']
                        problem['problem']['instance']['Hstride'] = layer_problem['problem']['Hstride']
                        problem['problem']['instance']['Wdilation'] = layer_problem['problem']['Wdilation']
                        problem['problem']['instance']['Hdilation'] = layer_problem['problem']['Hdilation']

                        problems_list.append(problem)
        
        else:
            with open(os.path.join(benchmark_dir, '{}_workload/layers.yaml'.format(workload)), 'r') as fd:
                layers = yaml.load(fd, Loader=yaml.SafeLoader)
        
            report_dir = os.path.join(opt.report_dir, '{}'.format(search_mode), '{}'.format(mapper), '{}'.format(accelerator),'obj_{}'.format(opt.optim_obj),
                                    '{}_input{}_all_DNN'.format(workload, batch_size))
            os.makedirs(report_dir, exist_ok=True)
            problems_list = []
            for layer_id in range(len(layers)):
                layer = layers[layer_id]
                with open(os.path.join(benchmark_dir, '{}_workload/{}.yaml'.format(workload, layer)), 'r') as fd:
                    layer_problem = yaml.load(fd, Loader=yaml.SafeLoader)
                    problem = {'problem': {
                        'shape': {'name': 'CNN_Layer', 'dimensions': ['H', 'C', 'K', 'R', 'S', 'N', 'P', 'Q'],
                                'coefficients': [{'name': 'Wstride', 'default': 1},
                                                {'name': 'Hstride', 'default': 1},
                                                {'name': 'Wdilation', 'default': 1},
                                                {'name': 'Hdilation', 'default': 1}],
                                },
                        'instance': {'C': 256, 'K': 512, 'R': 3, 'S': 3, 'P': 56, 'Q': 56, 'H': 1, 'N': 16,
                                    'Wstride': 1, 'Hstride': 1, 'Wdilation': 1, 'Hdilation': 1
                                    }}}
                    if 'type' in layer_problem['problem'].keys() and layer_problem['problem']['type'] == 'T2D':
                        problem['problem']['shape']['data_spaces'] = [
                                        {'name': 'Weights',
                                        'projection': [[['H']], [['C']], [['K']], [['R']], [['S']]]},
                                        {'name': 'Outputs', 'projection': [[['N']], [['H']], [['K']],
                                                                            [['R', 'Wdilation'],
                                                                            ['P', 'Wstride']],
                                                                            [['S', 'Hdilation'],
                                                                            ['Q', 'Hstride']]],
                                        'read_write': True},
                                        {'name': 'Inputs', 'projection': [[['N']], [['H']], [['C']], [['Q']], [['P']]]}]
                        problem['problem']['instance']['type'] = 'T2D'
                    else:
                        problem['problem']['shape']['data_spaces'] = [
                                        {'name': 'Weights',
                                        'projection': [[['H']], [['C']], [['K']], [['R']], [['S']]]},
                                        {'name': 'Inputs', 'projection': [[['N']], [['H']], [['C']],
                                                                            [['R', 'Wdilation'],
                                                                            ['P', 'Wstride']],
                                                                            [['S', 'Hdilation'],
                                                                            ['Q', 'Hstride']]]},
                                        {'name': 'Outputs',
                                        'projection': [[['N']], [['H']], [['K']], [['Q']], [['P']]],
                                        'read_write': True}]
                        problem['problem']['instance']['type'] = 'C2D'
                    if 'H' in layer_problem['problem'].keys():
                        problem['problem']['instance']['H'] = layer_problem['problem']['H']
                    else:
                        problem['problem']['instance']['H'] = 1
                    if 'type' in layer_problem['problem'].keys() and layer_problem['problem']['type'] == 'BMM':
                        problem['problem']['instance']['N'] = layer_problem['problem']['N']
                        problem['problem']['instance']['H'] = layer_problem['problem']['H'] * batch_size
                    else:
                        problem['problem']['instance']['N'] = layer_problem['problem']['N'] * batch_size
                    problem['problem']['instance']['K'] = layer_problem['problem']['K']
                    problem['problem']['instance']['C'] = layer_problem['problem']['C']
                    problem['problem']['instance']['P'] = layer_problem['problem']['P']
                    problem['problem']['instance']['Q'] = layer_problem['problem']['Q']
                    problem['problem']['instance']['R'] = layer_problem['problem']['R']
                    problem['problem']['instance']['S'] = layer_problem['problem']['S']
                    problem['problem']['instance']['Wstride'] = layer_problem['problem']['Wstride']
                    problem['problem']['instance']['Hstride'] = layer_problem['problem']['Hstride']
                    problem['problem']['instance']['Wdilation'] = layer_problem['problem']['Wdilation']
                    problem['problem']['instance']['Hdilation'] = layer_problem['problem']['Hdilation']

                    problems_list.append(problem)
        if search_mode == "UF":
            ME = MappingExplorer(problems_list, accelerator_dir, accelerator, mapper, type, version, report_dir, opt.optim_obj, expanded_scope, weight_num, solver=solver, all_DNN=all_DNN)
            print(opt.population)
            chkpt = ME.run(num_population=opt.population, num_generations=opt.epochs)
        elif search_mode == "HF":
            HF = HardwareFirstSearch(problems_list, accelerator_dir, accelerator, mapper, type, version, report_dir, opt.optim_obj, all_DNN=all_DNN)
            print(opt.population)
            chkpt = HF.run()
        elif search_mode == "MF":
            MF = MappingFirstSearch(problems_list, accelerator_dir, accelerator, mapper, type, version, report_dir, opt.optim_obj, all_DNN=all_DNN)
            print(opt.population)
            chkpt = MF.run(num_population=opt.population, num_generations=opt.epochs)
    else:
        with open(os.path.join(benchmark_dir, '{}_workload/layers.yaml'.format(workload)), 'r') as fd:
            layers = yaml.load(fd, Loader=yaml.SafeLoader)
        
        layer = layers[layer_id]
        
        report_dir = os.path.join(opt.report_dir, '{}'.format(mapper), '{}'.format(accelerator),'obj_{}'.format(opt.optim_obj),
                                '{}_input{}'.format(workload, batch_size), 'layer-{}'.format(layer_id), 'expand-{}'.format(expanded_scope))
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(benchmark_dir, '{}_workload/{}.yaml'.format(workload, layer)), 'r') as fd:
            layer_problem = yaml.load(fd, Loader=yaml.SafeLoader)
            problem = {'problem': {
                'shape': {'name': 'CNN_Layer', 'dimensions': ['H', 'C', 'K', 'R', 'S', 'N', 'P', 'Q'],
                        'coefficients': [{'name': 'Wstride', 'default': 1},
                                        {'name': 'Hstride', 'default': 1},
                                        {'name': 'Wdilation', 'default': 1},
                                        {'name': 'Hdilation', 'default': 1}],
                        },
                'instance': {'C': 256, 'K': 512, 'R': 3, 'S': 3, 'P': 56, 'Q': 56, 'H': 1, 'N': 16,
                            'Wstride': 1, 'Hstride': 1, 'Wdilation': 1, 'Hdilation': 1
                            }}}
            if 'type' in layer_problem['problem'].keys() and layer_problem['problem']['type'] == 'T2D':
                problem['problem']['shape']['data_spaces'] = [
                                {'name': 'Weights',
                                'projection': [[['H']], [['C']], [['K']], [['R']], [['S']]]},
                                {'name': 'Outputs', 'projection': [[['N']], [['H']], [['K']],
                                                                    [['R', 'Wdilation'],
                                                                    ['P', 'Wstride']],
                                                                    [['S', 'Hdilation'],
                                                                    ['Q', 'Hstride']]],
                                'read_write': True},
                                {'name': 'Inputs', 'projection': [[['N']], [['H']], [['C']], [['Q']], [['P']]]}]
                problem['problem']['instance']['type'] = 'T2D'
            else:
                problem['problem']['shape']['data_spaces'] = [
                                {'name': 'Weights',
                                'projection': [[['H']], [['C']], [['K']], [['R']], [['S']]]},
                                {'name': 'Inputs', 'projection': [[['N']], [['H']], [['C']],
                                                                    [['R', 'Wdilation'],
                                                                    ['P', 'Wstride']],
                                                                    [['S', 'Hdilation'],
                                                                    ['Q', 'Hstride']]]},
                                {'name': 'Outputs',
                                'projection': [[['N']], [['H']], [['K']], [['Q']], [['P']]],
                                'read_write': True}]
                problem['problem']['instance']['type'] = 'C2D'
            if 'H' in layer_problem['problem'].keys():
                problem['problem']['instance']['H'] = layer_problem['problem']['H']
            else:
                problem['problem']['instance']['H'] = 1
            if 'type' in layer_problem['problem'].keys() and layer_problem['problem']['type'] == 'BMM':
                problem['problem']['instance']['N'] = layer_problem['problem']['N']
                problem['problem']['instance']['H'] = layer_problem['problem']['H'] * batch_size
            else:
                problem['problem']['instance']['N'] = layer_problem['problem']['N'] * batch_size
            problem['problem']['instance']['K'] = layer_problem['problem']['K']
            problem['problem']['instance']['C'] = layer_problem['problem']['C']
            problem['problem']['instance']['P'] = layer_problem['problem']['P']
            problem['problem']['instance']['Q'] = layer_problem['problem']['Q']
            problem['problem']['instance']['R'] = layer_problem['problem']['R']
            problem['problem']['instance']['S'] = layer_problem['problem']['S']
            problem['problem']['instance']['Wstride'] = layer_problem['problem']['Wstride']
            problem['problem']['instance']['Hstride'] = layer_problem['problem']['Hstride']
            problem['problem']['instance']['Wdilation'] = layer_problem['problem']['Wdilation']
            problem['problem']['instance']['Hdilation'] = layer_problem['problem']['Hdilation']
        
        with open(os.path.join(accelerator_dir, mapper, accelerator, 'problem.yaml'), 'w') as fd:
            yaml.dump(problem, fd)
        
        with open(os.path.join(config_dir, '{}.yaml'.format(expanded_config)), 'r') as fd:
            expanded_dict = yaml.load(fd, Loader=yaml.SafeLoader)


        weight_matrix = None
        if weight_matrix_config:
            with open(os.path.join(config_dir, '{}.json'.format(weight_matrix_config)), 'r') as fd:
                data = json.load(fd)
            matrix_list = []
            for key, matrix in data.items():
                matrix_list.append(matrix)
            weight_matrix = matrix_list[:16]
            
        ME = MappingExplorer(problem['problem']['instance'], accelerator_dir, accelerator, mapper, type, version, report_dir, opt.optim_obj, expanded_scope, expanded_dict, parameter_dimension, weight_matrix=weight_matrix, solver=solver)
        print(opt.population)
        chkpt = ME.run(num_population=opt.population, num_generations=opt.epochs)
        # with open(os.path.join(report_dir, 'env_chkpt.plt'), 'wb') as fd:
        #     pickle.dump(chkpt, fd)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optim_obj', type=str, default="latency", help='optimization objective')
    parser.add_argument('--population', type=int, default=20, help='number of population')
    parser.add_argument('--epochs', type=int, default=10, help='number of generations/epochs')
    parser.add_argument('--report_dir', type=str, default='../report', help='The report directory')
    parser.add_argument('--accelerator', type=str, default='Simba', help='accelerator accelerator')
    parser.add_argument('--mapper', type=str, default='MARS', help='mapper tool')
    parser.add_argument('--type', type=str, default='arch', help='accelerator type')
    parser.add_argument('--version', type=str, default='v1', help='timeloop arch format version')
    parser.add_argument('--workload', type=str, default=None)
    parser.add_argument('--layer_id', type=int, default=None)
    parser.add_argument('--expanded_scope', type=float, default=1.0, help='expand the scope of the workload, range from 0 to 1')
    parser.add_argument('--expanded_config', type=str, default='expanded_config', help='The configuration file for the expansion of the workload')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--parameter_dimension', type=int, default=2)
    parser.add_argument('--weight_matrix_config', type=str, default=None)
    parser.add_argument('--solver', type=str, default='lp')
    parser.add_argument('--all_DNN', action='store_true', help='启用 all_DNN 模式（无需额外参数）')
    parser.add_argument('--weight_num', type=int, default=10)
    parser.add_argument('--all_workload', action='store_true', help='启用 all_DNN 模式（无需额外参数）')
    parser.add_argument('--search_mode', type=str, default='UF', help='搜索策略：硬件优先（HF），映射优先（MF），以及默认统一搜索（UF）')
    main()