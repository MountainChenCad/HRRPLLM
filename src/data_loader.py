import os
import glob
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm

def get_target_label(filename):
    """从文件名提取目标类型标签"""
    return os.path.basename(filename).split('_')[0]

def load_hrrp_data(dataset_path, data_variable_name, expected_length=None, target_length=None):
    """
    加载指定路径下的所有.mat HRRP文件。

    Args:
        dataset_path (str): 数据集文件夹路径。
        data_variable_name (str): .mat文件中存储HRRP数据的变量名 (如 'CoHH' 或 'data')。
        expected_length (int, optional): 期望的HRRP原始长度，用于校验。
        target_length (int, optional): 如果提供，将HRRP数据填充或截断到此长度。

    Returns:
        tuple: (hrrp_samples, labels, filenames)
               hrrp_samples (list of np.array): HRRP样本列表。
               labels (list of str): 对应的标签列表。
               filenames (list of str): 对应的原始文件名列表。
    """
    hrrp_samples = []
    labels = []
    filenames = []
    
    mat_files = glob.glob(os.path.join(dataset_path, "*.mat"))
    if not mat_files:
        print(f"警告: 在路径 {dataset_path} 下未找到任何 .mat 文件。")
        return [], [], []

    print(f"正在从 {dataset_path} 加载数据...")
    for filepath in tqdm(mat_files, desc=f"加载 {os.path.basename(dataset_path)}"):
        try:
            mat_data = loadmat(filepath)
            hrrp_vector = mat_data[data_variable_name].flatten().astype(float)

            if expected_length and len(hrrp_vector) != expected_length:
                print(f"警告: 文件 {filepath} HRRP长度 {len(hrrp_vector)} 与期望长度 {expected_length} 不符，已跳过。")
                continue
            
            if target_length:
                if len(hrrp_vector) < target_length:
                    # 补零
                    padding = np.zeros(target_length - len(hrrp_vector))
                    hrrp_vector = np.concatenate((hrrp_vector, padding))
                elif len(hrrp_vector) > target_length:
                    # 截断 (从开头)
                    hrrp_vector = hrrp_vector[:target_length]
            
            hrrp_samples.append(hrrp_vector)
            labels.append(get_target_label(filepath))
            filenames.append(os.path.basename(filepath))
        except KeyError:
            print(f"警告: 文件 {filepath} 中未找到变量 '{data_variable_name}'，已跳过。")
        except Exception as e:
            print(f"警告: 加载文件 {filepath} 失败: {e}，已跳过。")
            
    if not hrrp_samples:
        print(f"错误: 未能从 {dataset_path} 成功加载任何HRRP样本。请检查文件和data_variable_name。")

    return hrrp_samples, labels, filenames

if __name__ == '__main__':
    # 测试加载
    from config import SIMULATED_DATA_PATH, MEASURED_DATA_PATH, TARGET_HRRP_LENGTH
    
    sim_hrrps, sim_labels, _ = load_hrrp_data(SIMULATED_DATA_PATH, 'CoHH', expected_length=1000, target_length=TARGET_HRRP_LENGTH)
    print(f"\n加载仿真数据: {len(sim_hrrps)} 个样本. 第一个样本长度: {len(sim_hrrps[0]) if sim_hrrps else 'N/A'}")
    
    meas_hrrps, meas_labels, _ = load_hrrp_data(MEASURED_DATA_PATH, 'data', expected_length=500, target_length=TARGET_HRRP_LENGTH)
    print(f"加载实测数据: {len(meas_hrrps)} 个样本. 第一个样本长度: {len(meas_hrrps[0]) if meas_hrrps else 'N/A'}")

    if sim_hrrps:
        print("仿真数据标签示例:", list(set(sim_labels))[:5])
    if meas_hrrps:
        print("实测数据标签示例:", list(set(meas_labels))[:5])