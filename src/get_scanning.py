import os
import pandas as pd

def get_scanning_path(tooling_number, base_dir):
    # 寻找各个文件夹
    for folder in os.listdir(base_dir):
        # 检查文件夹名是否为 A+B 或 C+D 之一
        if folder in ['A+B', 'C+D']:
            # 构建工装目录路径
            tooling_path = os.path.join(base_dir, folder, str(tooling_number))
            scanning_path = os.path.join(tooling_path, 'Scanning')

            # 检查 Scanning 目录是否存在
            if os.path.exists(scanning_path):
                return scanning_path

    return None  # 如果没有找到则返回 None

def read_tooling_info(excel_file):
    df = pd.read_excel(excel_file)
    return df.iloc[:, 6].tolist()
