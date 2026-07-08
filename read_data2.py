import os
import glob
import pandas as pd
import re

def extract_value_from_file(file_path):
    """从文件中提取Cycles值"""
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            # 使用正则表达式查找Cycles行
            # match = re.search(r'Cycles:\s*(\d+)', content)
            # match = re.search(r'Energy:\s*(\d+\.\d+)', content)
            match = re.search(r'EDP\(J\*cycle\):\s*([\de.+]+)', content)
            if match:
                return match.group(1)  # 返回捕获的数字部分
            else:
                print(f"警告: 在文件 {file_path} 中未找到Cycles值")
                return None
    except Exception as e:
        print(f"错误: 读取文件 {file_path} 时出错: {e}")
        return None

def main():
    # 配置参数
    main_folder = "./report/UF/Saturn/Gemmini/obj_EDP/resnet50_1_input1_all_DNN"  # 主文件夹路径
    # main_folder = "./fame_report/arch_Simba_v1/obj_edp/deepbench_input1"  # 主文件夹路径
    target_file = "timeloop-model.stats.txt"  # 目标文件名
    output_excel = "extracted_cycles.xlsx"  # 输出Excel文件名
    
    # 存储提取的值和对应的子文件夹名
    results = []
    
    # 遍历主文件夹下的所有一级子文件夹
    for first_level_folder in os.listdir(main_folder):
        first_level_path = os.path.join(main_folder, first_level_folder)
        # 确保是文件夹而不是文件
        if os.path.isdir(first_level_path):
            
            # 查找目标文件
            file_path = os.path.join(first_level_path, target_file)
            
            # 检查文件是否存在
            if os.path.exists(file_path):
                # 提取值
                value = extract_value_from_file(file_path)
                
                # 如果成功提取到值，添加到结果列表
                if value is not None:
                    results.append({
                        "一级文件夹": first_level_folder,
                        "Cycles值": value
                    })
            else:
                print(f"警告: 在路径 {first_level_path} 中未找到文件 {target_file}")
    
    # # 将结果转换为DataFrame并保存到Excel
    # if results:
    #     df = pd.DataFrame(results)
    #     df.to_excel(output_excel, index=False)
    #     print(f"成功将 {len(results)} 个Cycles值保存到 {output_excel}")
    # else:
    #     print("未提取到任何Cycles值")

    # 在生成results列表后，保存Excel前添加排序
    if results:
        # 按二级文件夹中的数字排序
        results.sort(key=lambda x: int(x["一级文件夹"].split('-')[1]))
        
        df = pd.DataFrame(results)
        df.to_excel(output_excel, index=False)
        print(f"成功将 {len(results)} 个Cycles值保存到 {output_excel}")

if __name__ == "__main__":
    main()