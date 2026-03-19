import sys
import os
from typing import List, Dict
from accelergy.raw_inputs_2_dicts import RawInputs2Dicts
from accelergy.system_state import SystemState
from accelergy.component_class import ComponentClass
from accelergy.arch_dict_2_obj import arch_dict_2_obj
from accelergy.plug_in_path_to_obj import plug_in_path_to_obj
from accelergy.primitive_component import PrimitiveComponent
from accelergy.compound_component import CompoundComponent
from accelergy.ART_generator import AreaReferenceTableGenerator
from accelergy.ERT_generator import EnergyReferenceTableGenerator
# from accelergy.utils import INFO, ERROR_CLEAN_EXIT, accelergy_dumper
from collections import OrderedDict
import logging

# 1. 重新定义INFO/ERROR_CLEAN_EXIT（新版移除了这些常量）
def INFO(msg):
    logging.info(f"[Accelergy INFO] {msg}")

def ERROR_CLEAN_EXIT(msg):
    logging.error(f"[Accelergy ERROR] {msg}")
    sys.exit(1)

# 2. 重新定义accelergy_dumper（新版yaml格式化器逻辑）
def accelergy_dumper(dump_dict, stream=None, indent=2, sort_keys=False):
    """兼容版accelergy_dumper，保持和旧版一致的格式化规则"""
    return yaml.dump(
        dump_dict,
        stream=stream,
        indent=indent,
        sort_keys=sort_keys,
        default_flow_style=False
    )


class AccelergyHandler:
    def __init__(self, in_files: List[str]):
        sys.stdout = open(os.devnull, 'w')
        self.accelergy_version = 0.4

        self.output_prefix = ""
        path_arglist = in_files
        self.precision = 5

        # ----- Global Storage of System Info
        self.system_state = SystemState()
        self.system_state.set_accelergy_version(self.accelergy_version)

        # ----- Load Raw Inputs to Parse into Dicts
        raw_input_info = {'path_arglist': path_arglist, 'parser_version': self.accelergy_version}
        self.raw_dicts = RawInputs2Dicts(raw_input_info)
        sys.stdout = sys.__stdout__


    def update_depth(self, arch, buffer_name: str, new_depth: int):
        if isinstance(arch, list):
            for k,v in enumerate(arch):
                if isinstance(v, dict) or isinstance(v, OrderedDict) or isinstance(v, list):
                    arch[k] = self.update_depth(v, buffer_name, new_depth)
            return arch
                        
        for k, v in arch.items():
            if isinstance(v, dict) or isinstance(v, OrderedDict) or isinstance(v, list):
                arch[k] = self.update_depth(v, buffer_name, new_depth)
            elif k == 'name' and buffer_name in v:
                arch['attributes']['depth'] = new_depth
                return arch
                
        return arch

    def update_inst(self, arch, buffer_name: str, new_inst: int):
        if isinstance(arch, list):
            for k,v in enumerate(arch):
                if isinstance(v, dict) or isinstance(v, OrderedDict) or isinstance(v, list):
                    arch[k] = self.update_inst(v, buffer_name, new_inst)
            return arch
                        
        for k, v in arch.items():
            if isinstance(v, dict) or isinstance(v, OrderedDict) or isinstance(v, list):
                arch[k] = self.update_inst(v, buffer_name, new_inst)
            elif k == 'name' and buffer_name in v:
                arch['attributes']['instances'] = new_inst
                return arch
                
        return arch


    '''
        @brief: update energy reference table and area reference table
        @updates: list of tuples, such as [('buffer', 1), ('buffer', 2)]
    '''
    def update_accelergy(self, depth_updates: Dict[str, int], inst_updates: Dict[str, int]):
        sys.stdout = open(os.devnull, 'w')
        # ----- Global Storage of System Info
        self.system_state = SystemState()
        self.system_state.set_accelergy_version(self.accelergy_version)
        
        for buffer, depth in depth_updates.items():
            self.raw_dicts.hier_arch_spec_dict = self.update_depth(self.raw_dicts.hier_arch_spec_dict, 
                buffer, depth)
        
        for buffer, inst in inst_updates.items():
            self.raw_dicts.hier_arch_spec_dict = self.update_inst(self.raw_dicts.hier_arch_spec_dict, 
                buffer, inst)

        # ----- Determine what operations should be performed
        available_inputs = self.raw_dicts.get_available_inputs()

        # ---- Detecting config only cases and gracefully exiting
        if len(available_inputs) == 0:
            INFO("no input is provided, exiting...")
            sys.exit(0)
                
        # ----- Interpret the input architecture description using only the input information (w/o class definitions)
        self.system_state.set_hier_arch_spec(self.raw_dicts.get_hier_arch_spec_dict())

        # ----- Add the Component Classes
        for pc_name, pc_info in self.raw_dicts.get_pc_classses().items():
            self.system_state.add_pc_class(ComponentClass(pc_info))
        for cc_name, cc_info in self.raw_dicts.get_cc_classses().items():
            self.system_state.add_cc_class(ComponentClass(cc_info))

        # ----- Set Architecture Spec (all attributes defined)
        arch_obj = arch_dict_2_obj(self.raw_dicts.get_flatten_arch_spec_dict(), 
            self.system_state.cc_classes, self.system_state.pc_classes)
        self.system_state.set_arch_spec(arch_obj)

        # ERT/ERT_summary/energy estimates/ART/ART summary need to be generated without provided ERT
        #                ----> all components need to be defined
        # ----- Add the Fully Defined Components (all flattened out)

        for arch_component in self.system_state.arch_spec:
            if arch_component.get_class_name() in self.system_state.pc_classes:
                class_name = arch_component.get_class_name()
                pc = PrimitiveComponent({'component': arch_component, 'pc_class': self.system_state.pc_classes[class_name]})
                self.system_state.add_pc(pc)
            elif arch_component.get_class_name() in self.system_state.cc_classes:
                cc = CompoundComponent({'component': arch_component, 'pc_classes':self.system_state.pc_classes, 
                    'cc_classes':self.system_state.cc_classes})
                self.system_state.add_cc(cc)
            else:
                ERROR_CLEAN_EXIT('Cannot find class name %s specified in architecture'%arch_component.get_class())

        # ----- Add all available plug-ins
        self.system_state.add_plug_ins(plug_in_path_to_obj(self.raw_dicts.get_estimation_plug_in_paths(), 
            self.output_prefix))
        sys.stdout = sys.__stdout__


    def get_ert_art(self) -> (dict, dict):
        sys.stdout = open(os.devnull, 'w')
        # ----- Generate Energy Reference Table
        ert_gen = EnergyReferenceTableGenerator({'parser_version': self.accelergy_version,
                                                                                                'pcs': self.system_state.pcs,
                                                                                                'ccs': self.system_state.ccs,
                                                                                                'plug_ins': self.system_state.plug_ins,
                                                                                                'precision': self.precision})

        # ----- Generate Area Reference Table
        art_gen = AreaReferenceTableGenerator({'parser_version': self.accelergy_version,
                                                                                        'pcs': self.system_state.pcs,
                                                                                        'ccs': self.system_state.ccs,
                                                                                        'plug_ins': self.system_state.plug_ins,
                                                                                        'precision': self.precision})
        
        ert = ert_gen.get_ERT().get_ERT()
        art = art_gen.get_ART().get_ART()
        sys.stdout = sys.__stdout__
        return (ert, art)
    
    def get_arch(self) -> dict:
        return self.raw_dicts.get_hier_arch_spec_dict()
    

import sys
import os
import tempfile
import yaml
import unittest
from typing import List


# 生成测试用的最小架构配置文件（临时文件）
def create_test_arch_file() -> str:
    arch_yaml = {
        "version": 0.4,
        "architecture": {
            "components": [
                {
                    "name": "GlobalBuffer",
                    "class": "SRAM",
                    "attributes": {"depth": 512, "instances": 1}
                },
                {
                    "name": "PEInputBuffer",
                    "class": "SRAM",
                    "attributes": {"depth": 128, "instances": 16}
                }
            ]
        }
    }
    # 创建临时文件
    tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(arch_yaml, tmp_file)
    tmp_file.close()
    return tmp_file.name

class TestAccelergyHandlerCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. 打印当前Accelergy版本（关键）
        try:
            import accelergy
            print(f"当前安装的Accelergy版本: {accelergy.__version__}")
        except AttributeError:
            print("无法获取Accelergy版本（v0.5+需通过CLI查看：accelergy --version）")
        
        # 2. 创建测试用架构文件
        cls.test_arch_path = create_test_arch_file()
        cls.in_files = [cls.test_arch_path]

    @classmethod
    def tearDownClass(cls):
        # 删除临时文件
        os.unlink(cls.test_arch_path)

    def test_init(self):
        """测试初始化是否成功（核心导入/类实例化）"""
        try:
            handler = AccelergyHandler(self.in_files)
            self.assertIsInstance(handler, AccelergyHandler)
            print("✅ 初始化成功")
        except ImportError as e:
            self.fail(f"❌ 导入失败（版本不兼容）：{e}")
        except Exception as e:
            self.fail(f"❌ 初始化失败：{e}")

    def test_update_accelergy(self):
        """测试配置更新方法"""
        handler = AccelergyHandler(self.in_files)
        # 构造测试用更新参数
        depth_updates = {"GlobalBuffer": 1024}
        inst_updates = {"PEInputBuffer": 32}
        try:
            handler.update_accelergy(depth_updates, inst_updates)
            print("✅ update_accelergy执行成功")
        except Exception as e:
            self.fail(f"❌ update_accelergy执行失败：{e}")

    def test_get_ert_art(self):
        """测试ERT/ART生成（核心功能）"""
        handler = AccelergyHandler(self.in_files)
        depth_updates = {"GlobalBuffer": 1024}
        inst_updates = {"PEInputBuffer": 32}
        handler.update_accelergy(depth_updates, inst_updates)
        
        try:
            ert, art = handler.get_ert_art()
            self.assertIsInstance(ert, dict)
            self.assertIsInstance(art, dict)
            print("✅ get_ert_art执行成功，ART内容：", art)
        except Exception as e:
            self.fail(f"❌ get_ert_art执行失败：{e}")

    def test_get_arch(self):
        """测试架构配置获取"""
        handler = AccelergyHandler(self.in_files)
        try:
            arch = handler.get_arch()
            self.assertIsInstance(arch, dict)
            print("✅ get_arch执行成功")
        except Exception as e:
            self.fail(f"❌ get_arch执行失败：{e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
