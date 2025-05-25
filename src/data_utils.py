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
from scattering_center_encoder import encode_all_sc_sets_to_text 

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
            hrrp_vector_complex = loadmat(filepath)[data_var].flatten().astype(complex) 
            hrrp_vector = np.abs(hrrp_vector_complex).astype(float) 
            
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
            unique_labels, counts = np.unique(all_y_e, return_counts=True)
            can_stratify = len(unique_labels) > 1 and not np.any(counts < 2) 
            
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
            
            sc_params = {
                "prominence": sc_cfg.get("peak_prominence"), 
                "min_distance": sc_cfg.get("min_distance"),
                "max_centers_to_keep": sc_cfg.get("max_centers_to_keep"),
                "normalize_hrrp_before_extraction": sc_cfg.get("normalize_hrrp_before_extraction"),
                "normalization_type_for_hrrp": sc_cfg.get("normalization_type_for_hrrp")
            }
            sc_params = {k: v for k, v in sc_params.items() if v is not None}

            if X_tr_h.size > 0:
                X_tr_sc = [extract_scattering_centers_peak_detection(h, **sc_params) for h in tqdm(X_tr_h,desc=f"提取训练SC ({dataset_key})",leave=False, unit="samp")]
                with open(paths["X_train_scatter_centers"],'wb') as f: pickle.dump(X_tr_sc,f)
                print(f"训练SC .pkl 已更新 (数量: {len(X_tr_sc)})")
            else:
                print(f"训练数据为空 for '{dataset_key}', 跳过训练SC提取。")
                with open(paths["X_train_scatter_centers"],'wb') as f: pickle.dump([],f)


            if X_te_h.size > 0:
                X_te_sc = [extract_scattering_centers_peak_detection(h, **sc_params) for h in tqdm(X_te_h,desc=f"提取测试SC ({dataset_key})",leave=False, unit="samp")]
                with open(paths["X_test_scatter_centers"],'wb') as f: pickle.dump(X_te_sc,f)
                print(f"测试SC .pkl 已更新 (数量: {len(X_te_sc)})")
            else:
                print(f"测试数据为空 for '{dataset_key}', 跳过测试SC提取。")
                with open(paths["X_test_scatter_centers"],'wb') as f: pickle.dump([],f)
                
        elif config.SCATTERING_CENTER_EXTRACTION["enabled"]: print(f"数据集 '{dataset_key}' 跳过SC提取 (文件已存在)。")
        else: print(f"数据集 '{dataset_key}' 跳过SC提取 (已禁用)。")
        if not regenerate_hrrp and not regenerate_sc: print(f"数据集 '{dataset_key}': 文件已存在且无需更新。")
    print("数据准备阶段完成。")

def load_npy_data_internal(dataset_name_key, config, paths_dict):
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

def load_processed_data(dataset_name_key, config, load_scattering_centers=True):
    processed_path = os.path.join(config.PROCESSED_DATA_DIR, dataset_name_key)
    paths = {p_name: os.path.join(processed_path, f"{p_name}.{ext}") for p_name, ext in 
             [("X_train","npy"), ("y_train","npy"), ("X_test","npy"), ("y_test","npy"), 
              ("label_encoder","pkl"), ("X_train_scatter_centers","pkl"), ("X_test_scatter_centers","pkl")]}
    req_files = [paths[p] for p in ["X_train","y_train","X_test","y_test","label_encoder"]]
    if not all(os.path.exists(p) for p in req_files):
        print(f"错误: {dataset_name_key} 的 .npy/pkl 文件不完整于 {processed_path}"); return (None,)*8

    X_tr_h, y_tr_o, X_te_h, y_te_o, le, c_names = load_npy_data_internal(dataset_name_key, config, paths)
    if X_tr_h is None: return (None,)*8 

    X_tr_sc, X_te_sc = None, None
    if load_scattering_centers and config.SCATTERING_CENTER_EXTRACTION["enabled"]:
        # Handle cases where SC files might be empty lists if HRRP data was empty
        try:
            if os.path.exists(paths["X_train_scatter_centers"]):
                with open(paths["X_train_scatter_centers"],'rb') as f: X_tr_sc = pickle.load(f)
            else: X_tr_sc = [] # Default to empty list if file not found
            
            if os.path.exists(paths["X_test_scatter_centers"]):
                with open(paths["X_test_scatter_centers"],'rb') as f: X_te_sc = pickle.load(f)
            else: X_te_sc = [] # Default to empty list if file not found

        except Exception as e: 
            print(f"加载SC文件出错 for '{dataset_name_key}': {e}"); X_tr_sc,X_te_sc=[],[] # Fallback to empty lists
        
        if not X_tr_sc and X_tr_h.size > 0 : print(f"警告: {dataset_name_key} 的训练散射中心数据为空，但HRRP数据存在。")
        if not X_te_sc and X_te_h.size > 0 : print(f"警告: {dataset_name_key} 的测试散射中心数据为空，但HRRP数据存在。")

    return X_tr_h, y_tr_o, X_te_h, y_te_o, X_tr_sc, X_te_sc, le, c_names

def build_fsl_tasks(
    # MODIFIED: Tasks are now built from a single data pool, typically meta-test for evaluation
    X_data_pool_hrrp, y_data_pool_original, X_data_pool_sc_list,
    label_encoder, fsl_config, sc_encoding_config, random_state
):
    """
    构建FSL评估任务列表。
    每个任务从数据池 (通常是元测试集) 中采样 N 个类别，
    每个类别 K 个支撑样本和 Q 个查询样本。
    """
    np.random.seed(random_state)
    random.seed(random_state)

    n_way = fsl_config["n_way"]
    k_shot_support = fsl_config["k_shot_support"]
    q_shot_query = fsl_config["q_shot_query"]
    num_tasks = fsl_config["num_fsl_tasks"]
    
    tasks = []
    
    if X_data_pool_hrrp is None or X_data_pool_hrrp.size == 0:
        print("警告 (build_fsl_tasks): 提供的数据池为空。无法创建任务。")
        return []
    if X_data_pool_sc_list is not None and len(X_data_pool_hrrp) != len(X_data_pool_sc_list):
        print("警告 (build_fsl_tasks): HRRP数据池和SC列表长度不匹配。SC数据可能不可靠。")
        # Proceeding, but SCs might be misaligned or missing for some samples.
        # Consider making X_data_pool_sc_list mandatory or handling this more gracefully if SCs are critical.

    unique_pool_classes = np.unique(y_data_pool_original)
    
    # 按类别组织数据池的索引
    pool_class_indices = {cls: np.where(y_data_pool_original == cls)[0] for cls in unique_pool_classes}

    # 筛选出在数据池中至少有 K + Q 个样本的类别
    valid_classes_for_task_sampling = [
        cls for cls in unique_pool_classes if len(pool_class_indices[cls]) >= (k_shot_support + q_shot_query)
    ]

    if len(valid_classes_for_task_sampling) < n_way:
        print(f"警告 (build_fsl_tasks): 数据池中满足 K+Q ({k_shot_support+q_shot_query}) 样本条件的类别数 ({len(valid_classes_for_task_sampling)}) 少于 N-way ({n_way})。无法创建足够的任务。")
        return []

    for task_idx in tqdm(range(num_tasks), desc="构建FSL评估任务", leave=False, unit="task"):
        task_support_hrrp_list, task_support_sc_list_list, task_support_labels_list = [], [], []
        task_query_hrrp_list, task_query_sc_list_list, task_query_labels_list = [], [], []
        
        # 1. 从有效类别中选择 N 个类别
        if len(valid_classes_for_task_sampling) < n_way : # Should have been caught above, but good for safety
            # print(f"  任务 {task_idx+1}/{num_tasks}: 可用于构建当前任务的有效类别不足。已尝试所有组合或已达任务上限。")
            break # Stop trying to make more tasks if not enough classes
            
        selected_classes_for_task = np.random.choice(valid_classes_for_task_sampling, size=n_way, replace=False)
        
        current_task_valid = True
        for cls_name in selected_classes_for_task:
            class_specific_pool_indices = pool_class_indices[cls_name]
            
            # 从该类别的样本中随机选择 K+Q 个不同的样本索引
            if len(class_specific_pool_indices) < (k_shot_support + q_shot_query): # Should be caught by valid_classes check
                current_task_valid = False; break 
            
            selected_indices_for_class = np.random.choice(
                class_specific_pool_indices, 
                size=(k_shot_support + q_shot_query), 
                replace=False
            )
            
            # 划分支撑集和查询集
            np.random.shuffle(selected_indices_for_class) # 打乱以随机分配
            support_indices_for_class = selected_indices_for_class[:k_shot_support]
            query_indices_for_class = selected_indices_for_class[k_shot_support:] # Remaining are query

            # 添加到任务列表
            task_support_hrrp_list.extend(X_data_pool_hrrp[support_indices_for_class])
            task_support_labels_list.extend(y_data_pool_original[support_indices_for_class])
            if X_data_pool_sc_list:
                task_support_sc_list_list.extend([X_data_pool_sc_list[i] for i in support_indices_for_class])
            
            task_query_hrrp_list.extend(X_data_pool_hrrp[query_indices_for_class])
            task_query_labels_list.extend(y_data_pool_original[query_indices_for_class])
            if X_data_pool_sc_list:
                task_query_sc_list_list.extend([X_data_pool_sc_list[i] for i in query_indices_for_class])

        if not current_task_valid:
            # print(f"  任务 {task_idx+1}/{num_tasks}: 构建时样本不足。跳过。")
            continue
        
        # 编码SC为文本
        task_support_sc_texts = encode_all_sc_sets_to_text(task_support_sc_list_list, sc_encoding_config) if X_data_pool_sc_list and task_support_sc_list_list else [""] * len(task_support_hrrp_list)
        task_query_sc_texts = encode_all_sc_sets_to_text(task_query_sc_list_list, sc_encoding_config) if X_data_pool_sc_list and task_query_sc_list_list else [""] * len(task_query_hrrp_list)

        # 确保每个任务都有 N*K 支撑和 N*Q 查询
        expected_support_size = n_way * k_shot_support
        expected_query_size = n_way * q_shot_query
        if len(task_support_labels_list) != expected_support_size or len(task_query_labels_list) != expected_query_size:
            print(f"  警告: 任务 {task_idx+1} 样本数量不符预期。支撑: {len(task_support_labels_list)}/{expected_support_size}, 查询: {len(task_query_labels_list)}/{expected_query_size}。可能由于类别内样本不足 K+Q。跳过此任务。")
            continue


        tasks.append({
            "support_hrrp": np.array(task_support_hrrp_list),
            "support_sc_list": task_support_sc_list_list,
            "support_labels": np.array(task_support_labels_list),
            "support_sc_texts": task_support_sc_texts,
            "query_hrrp": np.array(task_query_hrrp_list),
            "query_sc_list": task_query_sc_list_list,
            "query_labels": np.array(task_query_labels_list),
            "query_sc_texts": task_query_sc_texts,
            "task_classes": selected_classes_for_task, 
        })
        
    if not tasks:
        print(f"警告 (build_fsl_tasks): 未能成功构建任何FSL任务。请检查数据池中各类别样本数量是否满足 K+Q，以及N-way设置。")
    elif len(tasks) < num_tasks:
        print(f"警告 (build_fsl_tasks): 成功构建了 {len(tasks)} 个FSL任务，少于请求的 {num_tasks} 个。可能是由于类别或样本不足。")
    else:
        print(f"成功构建了 {len(tasks)} 个FSL任务。")
    return tasks


if __name__ == "__main__":
    class MockConfigDU_FSL_Strict: 
        AVAILABLE_DATASETS = {"simulated_fsl_strict_test": {"path": "datasets_test/simulated_hrrp_fsl_strict", "data_var": "CoHH", "original_len": 1000, "max_samples_to_load": 150}} # 5 classes * 30 samples
        TARGET_HRRP_LENGTH = 1000; PREPROCESS_MAT_TO_NPY = True; PROCESSED_DATA_DIR = "data_processed_test_fsl_strict"; 
        TEST_SPLIT_SIZE = 0.01 # Make meta-train very small, meta-test large for this test
        RANDOM_STATE = 42
        SCATTERING_CENTER_EXTRACTION = {"enabled": True, "method": "peak_detection", 
                                        "peak_prominence": 0.1, 
                                        "min_distance": 3,      
                                        "max_centers_to_keep": 5, 
                                        "normalize_hrrp_before_extraction": True, 
                                        "normalization_type_for_hrrp": "max"}
        SCATTERING_CENTER_ENCODING = {"format": "list_of_dicts", "precision_pos": 0, "precision_amp": 3, "TARGET_HRRP_LENGTH_INFO": TARGET_HRRP_LENGTH}
        FSL_TASK_SETUP = {"n_way": 3, "k_shot_support": 5, "q_shot_query": 1, "num_fsl_tasks": 10} 
        LIMIT_TEST_SAMPLES = None # Let build_fsl_tasks use the full meta-test set loaded

    config_module_for_data_utils.RANDOM_STATE = MockConfigDU_FSL_Strict.RANDOM_STATE
    mock_config_fsl_strict = MockConfigDU_FSL_Strict()
    
    test_sim_path_fsl_strict = os.path.join(mock_config_fsl_strict.AVAILABLE_DATASETS["simulated_fsl_strict_test"]["path"])
    os.makedirs(test_sim_path_fsl_strict, exist_ok=True); os.makedirs(mock_config_fsl_strict.PROCESSED_DATA_DIR, exist_ok=True)
    
    # Create data: 5 classes, each with 30 samples. K+Q = 5+1=6. So enough samples.
    num_mock_classes = 5
    samples_per_mock_class = 30
    if not glob.glob(os.path.join(test_sim_path_fsl_strict, "*.mat")):
        print(f"创建虚拟.mat for FSL Strict测试..."); from scipy.io import savemat
        for cl_idx in range(num_mock_classes): 
            cl = f"Cls{chr(65+cl_idx)}" # ClsA, ClsB, ...
            for i in range(samples_per_mock_class): 
                random_data_complex = np.random.rand(mock_config_fsl_strict.TARGET_HRRP_LENGTH) + \
                                      1j * np.random.rand(mock_config_fsl_strict.TARGET_HRRP_LENGTH)
                savemat(os.path.join(test_sim_path_fsl_strict, f"{cl}_s{i}.mat"), {"CoHH": random_data_complex.reshape(-1, 1)})
        print("FSL Strict虚拟文件创建完毕。")

    print("--- 测试 prepare_npy_data_and_scattering_centers for FSL Strict ---"); 
    prepare_npy_data_and_scattering_centers(mock_config_fsl_strict)
    
    print("\n--- 测试 load_processed_data for FSL Strict ---"); 
    load_res_strict = load_processed_data("simulated_fsl_strict_test", mock_config_fsl_strict, load_scattering_centers=True)
    if load_res_strict[0] is not None: 
        print(f"数据加载成功 for simulated_fsl_strict_test")
        # We will use the meta-test portion as the data pool for task building in this test
        _, _, X_pool_h, y_pool_o, _, X_pool_sc, le_strict, c_names_strict = load_res_strict
        print(f"  数据池 (来自元测试部分) HRRP: {X_pool_h.shape}, 标签: {y_pool_o.shape}")
        if X_pool_sc: print(f"  数据池 SC列表长度: {len(X_pool_sc)}")

        print("\n--- 测试 build_fsl_tasks (Strict N-way K-shot Q-query) ---")
        fsl_tasks_strict = build_fsl_tasks(
            X_pool_h, y_pool_o, X_pool_sc, # Using meta-test as the pool
            le_strict, mock_config_fsl_strict.FSL_TASK_SETUP, 
            mock_config_fsl_strict.SCATTERING_CENTER_ENCODING, 
            mock_config_fsl_strict.RANDOM_STATE
        )
        if fsl_tasks_strict:
            print(f"成功构建了 {len(fsl_tasks_strict)} 个FSL任务。")
            cfg = mock_config_fsl_strict.FSL_TASK_SETUP
            expected_S_size = cfg['n_way'] * cfg['k_shot_support']
            expected_Q_size = cfg['n_way'] * cfg['q_shot_query']
            for i, task in enumerate(fsl_tasks_strict[:2]): 
                print(f"  任务 {i+1}:")
                print(f"    类别: {task['task_classes']}")
                print(f"    支撑集样本数: {len(task['support_labels'])} (预期: {expected_S_size})")
                print(f"    查询集样本数: {len(task['query_labels'])} (预期: {expected_Q_size})")
                # Check individual class counts if needed by looking at task['support_labels'] and task['query_labels']
                unique_s_labels, s_counts = np.unique(task['support_labels'], return_counts=True)
                print(f"      支撑集各类别样本数: {dict(zip(unique_s_labels, s_counts))}")
                unique_q_labels, q_counts = np.unique(task['query_labels'], return_counts=True)
                print(f"      查询集各类别样本数: {dict(zip(unique_q_labels, q_counts))}")

                assert len(task['support_labels']) == expected_S_size
                assert len(task['query_labels']) == expected_Q_size
                for cls_in_task in task['task_classes']:
                    assert np.sum(task['support_labels'] == cls_in_task) == cfg['k_shot_support']
                    assert np.sum(task['query_labels'] == cls_in_task) == cfg['q_shot_query']

        else:
            print("未能构建FSL任务 (Strict)。")
