import os
import glob
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle
import random

from feature_extractor import extract_scattering_centers_peak_detection 

try:
    import config as config_module_for_data_utils
except ImportError:
    class FallbackConfigDU: RANDOM_STATE = 42 
    config_module_for_data_utils = FallbackConfigDU()
    print("警告(data_utils): 未能导入主config，使用默认RANDOM_STATE。")


def get_target_label_from_filename(filename, prefix_len=None):
    base = os.path.basename(filename); return base[:prefix_len] if prefix_len else base.split('_')[0]

def load_and_preprocess_mat_files(dataset_config, target_hrrp_length, max_samples_per_dataset=None):
    dataset_path = dataset_config["path"]; data_var = dataset_config["data_var"]
    label_prefix_len = dataset_config.get("label_prefix_len")
    hrrp_vectors, labels = [], []
    mat_files_all = glob.glob(os.path.join(dataset_path, "*.mat"))
    if not mat_files_all: print(f"警告: {dataset_path} 无.mat文件。"); return np.array([]), np.array([])

    mat_files_to_process = mat_files_all
    if max_samples_per_dataset and len(mat_files_all) > max_samples_per_dataset:
        current_random_state = getattr(config_module_for_data_utils, 'RANDOM_STATE', None)
        if current_random_state is not None: random.seed(current_random_state)
        mat_files_to_process = random.sample(mat_files_all, max_samples_per_dataset)
    
    for filepath in tqdm(mat_files_to_process, desc=f"加载 .mat ({os.path.basename(dataset_path)})", leave=False, unit="file"):
        try:
            hrrp_vector = loadmat(filepath)[data_var].flatten().astype(float)
            if len(hrrp_vector) < 10: continue
            if len(hrrp_vector) < target_hrrp_length:
                hrrp_vector = np.concatenate((hrrp_vector, np.zeros(target_hrrp_length - len(hrrp_vector))))
            elif len(hrrp_vector) > target_hrrp_length:
                hrrp_vector = hrrp_vector[:target_hrrp_length]
            hrrp_vectors.append(hrrp_vector)
            labels.append(get_target_label_from_filename(filepath, label_prefix_len))
        except Exception as e: print(f"警告: 文件 {filepath} 处理失败: {e}，跳过。")
    if not hrrp_vectors: print(f"错误: 未能从 {dataset_path} 加载任何有效HRRP。")
    return np.array(hrrp_vectors), np.array(labels)

def prepare_npy_data_and_scattering_centers(config):
    for dataset_key, dataset_conf in config.AVAILABLE_DATASETS.items():
        processed_path = os.path.join(config.PROCESSED_DATA_DIR, dataset_key)
        os.makedirs(processed_path, exist_ok=True)
        paths = {p_name: os.path.join(processed_path, f"{p_name}.{ext}") for p_name, ext in 
                 [("X_train","npy"), ("y_train","npy"), ("X_test","npy"), ("y_test","npy"), 
                  ("label_encoder","pkl"), ("X_train_scatter_centers","pkl"), ("X_test_scatter_centers","pkl")]}

        hrrp_files_exist = all(os.path.exists(p) for p in [paths["X_train"], paths["y_train"], paths["X_test"], paths["y_test"], paths["label_encoder"]])
        sc_files_exist = config.SCATTERING_CENTER_EXTRACTION["enabled"] and \
                         all(os.path.exists(paths[p_sc]) for p_sc in ["X_train_scatter_centers", "X_test_scatter_centers"])
        regenerate_hrrp = config.PREPROCESS_MAT_TO_NPY or not hrrp_files_exist
        regenerate_sc = config.SCATTERING_CENTER_EXTRACTION["enabled"] and (regenerate_hrrp or not sc_files_exist)
        
        if regenerate_hrrp:
            print(f"为数据集 '{dataset_key}' 生成或重新生成 .npy 文件...")
            max_samples = dataset_conf.get("max_samples_to_load", None)
            all_X_h, all_y_o = load_and_preprocess_mat_files(dataset_conf, config.TARGET_HRRP_LENGTH, max_samples)
            if all_X_h.size == 0: print(f"数据集 '{dataset_key}' 加载数据失败。"); continue

            le = LabelEncoder().fit(all_y_o); all_y_e = le.transform(all_y_o)
            with open(paths["label_encoder"], 'wb') as f: pickle.dump(le, f)

            indices = np.arange(len(all_X_h)); test_s = config.TEST_SPLIT_SIZE
            can_stratify = len(np.unique(all_y_e)) > 1 and not np.any(np.bincount(all_y_e) < 2)
            if len(all_X_h) * test_s < 1 and len(all_X_h) > 1: test_s = 1.0 / len(all_X_h)
            
            tr_idx, te_idx = train_test_split(indices, test_size=test_s, random_state=config.RANDOM_STATE, stratify=all_y_e if can_stratify else None) \
                                if len(all_X_h) > 1 else (indices, np.array([],dtype=int)) if len(all_X_h)==1 else (np.array([],dtype=int),np.array([],dtype=int))

            X_tr_h, y_tr_e = all_X_h[tr_idx], all_y_e[tr_idx]
            X_te_h, y_te_e = all_X_h[te_idx], all_y_e[te_idx]
            np.save(paths["X_train"],X_tr_h); np.save(paths["y_train"],y_tr_e)
            np.save(paths["X_test"],X_te_h); np.save(paths["y_test"],y_te_e)
            print(f"HRRP .npy 已保存 (元训练/支撑池: {X_tr_h.shape[0]}, 元测试/查询集: {X_te_h.shape[0]})")
        
        if regenerate_sc:
            if not regenerate_hrrp: 
                X_tr_h = np.load(paths["X_train"], allow_pickle=True)
                X_te_h = np.load(paths["X_test"], allow_pickle=True)
            sc_cfg = config.SCATTERING_CENTER_EXTRACTION
            print(f"为 '{dataset_key}' 提取或重新提取散射中心 (方法: {sc_cfg['method']})...")
            X_tr_sc = [extract_scattering_centers_peak_detection(h, **{k:v for k,v in sc_cfg.items() if k in ["prominence","min_distance","max_centers_to_keep","normalize_hrrp_before_extraction","normalization_type_for_hrrp"]}) for h in tqdm(X_tr_h,desc=f"提取训练SC ({dataset_key})",leave=False, unit="samp")]
            with open(paths["X_train_scatter_centers"],'wb') as f: pickle.dump(X_tr_sc,f)
            X_te_sc = [extract_scattering_centers_peak_detection(h, **{k:v for k,v in sc_cfg.items() if k in ["prominence","min_distance","max_centers_to_keep","normalize_hrrp_before_extraction","normalization_type_for_hrrp"]}) for h in tqdm(X_te_h,desc=f"提取测试SC ({dataset_key})",leave=False, unit="samp")]
            with open(paths["X_test_scatter_centers"],'wb') as f: pickle.dump(X_te_sc,f)
            print(f"SC .pkl 已更新 (训练: {len(X_tr_sc)}, 测试: {len(X_te_sc)})")
        elif config.SCATTERING_CENTER_EXTRACTION["enabled"]: print(f"数据集 '{dataset_key}' 跳过SC提取 (文件已存在)。")
        else: print(f"数据集 '{dataset_key}' 跳过SC提取 (已禁用)。")
        if not regenerate_hrrp and not regenerate_sc: print(f"数据集 '{dataset_key}': 文件已存在且无需更新。")
    print("数据准备阶段完成。")

def load_processed_data(dataset_name_key, config, load_scattering_centers=True):
    # (此函数与上一版本一致，不再重复)
    processed_path = os.path.join(config.PROCESSED_DATA_DIR, dataset_name_key)
    paths = {p_name: os.path.join(processed_path, f"{p_name}.{ext}") for p_name, ext in 
             [("X_train","npy"), ("y_train","npy"), ("X_test","npy"), ("y_test","npy"), 
              ("label_encoder","pkl"), ("X_train_scatter_centers","pkl"), ("X_test_scatter_centers","pkl")]}
    req_files = [paths[p] for p in ["X_train","y_train","X_test","y_test","label_encoder"]]
    if not all(os.path.exists(p) for p in req_files):
        print(f"错误: {dataset_name_key} 的 .npy/pkl 文件不完整于 {processed_path}"); return (None,)*8

    X_tr_h, y_tr_o, X_te_h, y_te_o, le, c_names = load_npy_data_internal(dataset_name_key, config, paths)
    X_tr_sc, X_te_sc = None, None
    if load_scattering_centers and config.SCATTERING_CENTER_EXTRACTION["enabled"]:
        if os.path.exists(paths["X_train_scatter_centers"]) and os.path.exists(paths["X_test_scatter_centers"]):
            try:
                with open(paths["X_train_scatter_centers"],'rb') as f: X_tr_sc = pickle.load(f)
                with open(paths["X_test_scatter_centers"],'rb') as f: X_te_sc = pickle.load(f)
            except Exception as e: print(f"加载SC文件出错 for '{dataset_name_key}': {e}"); X_tr_sc,X_te_sc=None,None
    return X_tr_h, y_tr_o, X_te_h, y_te_o, X_tr_sc, X_te_sc, le, c_names

def load_npy_data_internal(dataset_name_key, config, paths_dict):
    # (此函数与上一版本一致，不再重复)
    try:
        X_train = np.load(paths_dict["X_train"], allow_pickle=True) 
        y_train_encoded = np.load(paths_dict["y_train"], allow_pickle=True)
        X_test = np.load(paths_dict["X_test"], allow_pickle=True)
        y_test_encoded = np.load(paths_dict["y_test"], allow_pickle=True)
        with open(paths_dict["label_encoder"], 'rb') as f: label_encoder = pickle.load(f)
        class_names = list(label_encoder.classes_)
        y_train_original = label_encoder.inverse_transform(y_train_encoded) if y_train_encoded.size > 0 else np.array([])
        y_test_original = label_encoder.inverse_transform(y_test_encoded) if y_test_encoded.size > 0 else np.array([])
        return X_train, y_train_original, X_test, y_test_original, label_encoder, class_names
    except Exception as e:
        print(f"加载HRRP .npy 或LabelEncoder时出错 for '{dataset_name_key}': {e}")
        return (None,)*6

# build_task_support_set 函数不再需要，因为示例选择逻辑已移至 dynamic_neighbor_selector
# 并且是基于与当前查询的相似度从整个支撑集池中选择，而不是构建固定的N-Way K-Shot任务支撑集。
# 如果严格需要TableTime那种预先构建一个固定的N-Way K-Shot“任务支撑集”，然后再从中选，则需要保留并修改。
# 但根据您“从整个任务当中所有的支撑样本（不用是同类）去选最近邻的几个样本”的描述，
# 当前的 dynamic_neighbor_selector.py 的 select_prompt_examples_from_task_support 更符合。

if __name__ == "__main__":
    # (测试代码与之前类似，但不再测试 build_task_support_set)
    class MockConfigDU_FSL:
        AVAILABLE_DATASETS = {"simulated_fsl_test": {"path": "datasets_test/simulated_hrrp_fsl_test", "data_var": "CoHH", "original_len": 1000, "max_samples_to_load": 60}}
        TARGET_HRRP_LENGTH = 1000; PREPROCESS_MAT_TO_NPY = True; PROCESSED_DATA_DIR = "data_processed_test_fsl_main"; TEST_SPLIT_SIZE = 0.5; RANDOM_STATE = 42
        SCATTERING_CENTER_EXTRACTION = {"enabled": True, "method": "peak_detection", "peak_prominence": 0.1, "peak_min_distance": 3, "max_centers_to_keep": 5, "normalize_hrrp_before_extraction": True, "normalization_type_for_hrrp": "max"}
    config_module_for_data_utils.RANDOM_STATE = MockConfigDU_FSL.RANDOM_STATE
    mock_config_fsl_main = MockConfigDU_FSL()
    test_sim_path_fsl_main = os.path.join(mock_config_fsl_main.AVAILABLE_DATASETS["simulated_fsl_test"]["path"])
    os.makedirs(test_sim_path_fsl_main, exist_ok=True); os.makedirs(mock_config_fsl_main.PROCESSED_DATA_DIR, exist_ok=True)
    if not glob.glob(os.path.join(test_sim_path_fsl_main, "*.mat")):
        print(f"创建虚拟.mat for FSL测试..."); from scipy.io import savemat
        for cl in ["TypeX", "TypeY", "TypeZ"]: 
            for i in range(20): savemat(os.path.join(test_sim_path_fsl_main, f"{cl}_s{i}.mat"), {"CoHH": np.random.rand(mock_config_fsl_main.AVAILABLE_DATASETS["simulated_fsl_test"]["original_len"], 1)})
        print("FSL虚拟文件创建完毕。")
    print("--- 测试 prepare_npy_data_and_scattering_centers for FSL ---"); prepare_npy_data_and_scattering_centers(mock_config_fsl_main)
    print("\n--- 测试 load_processed_data for FSL ---"); 
    load_res = load_processed_data("simulated_fsl_test", mock_config_fsl_main, load_scattering_centers=True)
    if load_res[0] is not None: print(f"数据加载成功 for simulated_fsl_test")
