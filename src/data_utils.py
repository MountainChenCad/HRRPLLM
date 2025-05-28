# src/data_utils.py
import os
import glob
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle
import random
from scipy.spatial.distance import euclidean

from feature_extractor import extract_scattering_centers_peak_detection
from scattering_center_encoder import encode_single_sc_set_to_text, encode_all_sc_sets_to_text

try:
    from dynamic_neighbor_selector import sc_set_to_feature_vector
except ImportError:
    print("警告 (data_utils): 无法从 dynamic_neighbor_selector 导入 sc_set_to_feature_vector。原型计算将受影响。")


    def sc_set_to_feature_vector(sc_set, max_centers, feature_type):  # Dummy
        print(f"错误: sc_set_to_feature_vector 未加载, 无法为原型创建特征向量。")
        if feature_type == "pos_amp_flat":
            return np.zeros(max_centers * 2)
        else:
            return np.zeros(max_centers)

try:
    import config as config_module_for_data_utils  # Used for RANDOM_STATE fallback primarily
except ImportError:
    class FallbackConfigDU:
        RANDOM_STATE = 42
        # SCATTERING_CENTER_EXTRACTION = {"max_centers_to_keep": 10} # No longer needed here directly for fallback


    config_module_for_data_utils = FallbackConfigDU()
    print("警告(data_utils): 未能导入主config，使用默认RANDOM_STATE。")


def get_target_label_from_filename(filename, prefix_len=None):
    base = os.path.basename(filename);
    return base[:prefix_len] if prefix_len else base.split('_')[0]


def load_and_preprocess_mat_files(dataset_config, target_hrrp_length, max_samples_per_dataset=None):
    # ... (no changes in this function)
    dataset_path = dataset_config["path"];
    data_var = dataset_config["data_var"]
    label_prefix_len = dataset_config.get("label_prefix_len")
    hrrp_vectors, labels = [], []
    mat_files_all = glob.glob(os.path.join(dataset_path, "*.mat"))
    if not mat_files_all: print(f"警告: {dataset_path} 无.mat文件。"); return np.array([]), np.array([])

    mat_files_to_process = mat_files_all
    if max_samples_per_dataset and len(mat_files_all) > max_samples_per_dataset:
        current_random_state = getattr(config_module_for_data_utils, 'RANDOM_STATE', None)
        if current_random_state is not None: random.seed(current_random_state)
        mat_files_to_process = random.sample(mat_files_all, max_samples_per_dataset)

    for filepath in tqdm(mat_files_to_process, desc=f"加载 .mat ({os.path.basename(dataset_path)})", leave=False,
                         unit="file"):
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
        except Exception as e:
            print(f"警告: 文件 {filepath} 处理失败: {e}，跳过。")

    if not hrrp_vectors: print(f"错误: 未能从 {dataset_path} 加载任何有效HRRP。")
    return np.array(hrrp_vectors), np.array(labels)


def prepare_npy_data_and_scattering_centers(config):  # config here is the CurrentRunConfig object
    for dataset_key, dataset_conf in config.AVAILABLE_DATASETS.items():
        # If running for a specific dataset via CLI, we might only want to process that one.
        # However, the current loop processes all. This could be refined if needed.
        # For now, ensure that if config.dataset_key is set, we only process that.
        # This check should ideally be outside this function or this function should take dataset_key.
        # For simplicity of this fix, we assume this function is called in a context where it should process
        # all datasets defined in config.AVAILABLE_DATASETS, or the caller filters.
        # The main_experiment.py calls this and then focuses on config.dataset_key.

        processed_path = os.path.join(config.PROCESSED_DATA_DIR, dataset_key)
        os.makedirs(processed_path, exist_ok=True)
        paths = {p_name: os.path.join(processed_path, f"{p_name}.{ext}") for p_name, ext in
                 [("X_train", "npy"), ("y_train", "npy"), ("X_test", "npy"), ("y_test", "npy"),
                  ("label_encoder", "pkl"), ("X_train_scatter_centers", "pkl"), ("X_test_scatter_centers", "pkl")]}

        hrrp_files_exist = all(os.path.exists(p) for p in
                               [paths["X_train"], paths["y_train"], paths["X_test"], paths["y_test"],
                                paths["label_encoder"]])

        # MODIFIED: Access SC config from config.sc_extraction_config
        current_sc_extraction_config = config.sc_extraction_config

        sc_files_exist = current_sc_extraction_config["enabled"] and \
                         all(os.path.exists(paths[p_sc]) for p_sc in
                             ["X_train_scatter_centers", "X_test_scatter_centers"])

        regenerate_hrrp = config.PREPROCESS_MAT_TO_NPY or not hrrp_files_exist
        regenerate_sc = current_sc_extraction_config["enabled"] and (regenerate_hrrp or not sc_files_exist)

        if regenerate_hrrp:
            print(f"为数据集 '{dataset_key}' 生成或重新生成 .npy 文件...")
            max_samples = dataset_conf.get("max_samples_to_load", None)  # Use dataset specific max_samples
            all_X_h, all_y_o = load_and_preprocess_mat_files(dataset_conf, config.TARGET_HRRP_LENGTH, max_samples)
            if all_X_h.size == 0: print(f"数据集 '{dataset_key}' 加载数据失败。"); continue

            le = LabelEncoder().fit(all_y_o);
            all_y_e = le.transform(all_y_o)
            with open(paths["label_encoder"], 'wb') as f:
                pickle.dump(le, f)

            indices = np.arange(len(all_X_h));
            test_s = config.TEST_SPLIT_SIZE
            unique_labels, counts = np.unique(all_y_e, return_counts=True)
            can_stratify = len(unique_labels) > 1 and not np.any(counts < 2)

            if len(all_X_h) > 0 and len(all_X_h) * test_s < 1: test_s = 1.0 / len(all_X_h) if len(all_X_h) > 0 else 0

            if len(all_X_h) > 1:
                tr_idx, te_idx = train_test_split(indices, test_size=test_s, random_state=config.RANDOM_STATE,
                                                  stratify=all_y_e if can_stratify else None)
            elif len(all_X_h) == 1:
                tr_idx, te_idx = indices, np.array([], dtype=int)
            else:  # len(all_X_h) == 0
                tr_idx, te_idx = np.array([], dtype=int), np.array([], dtype=int)

            X_tr_h, y_tr_e = all_X_h[tr_idx], all_y_e[tr_idx]
            X_te_h, y_te_e = all_X_h[te_idx], all_y_e[te_idx]
            np.save(paths["X_train"], X_tr_h);
            np.save(paths["y_train"], y_tr_e)
            np.save(paths["X_test"], X_te_h);
            np.save(paths["y_test"], y_te_e)
            print(f"HRRP .npy 已保存 (元训练/支撑池: {X_tr_h.shape[0]}, 元测试/查询集: {X_te_h.shape[0]})")

        if regenerate_sc:
            if not regenerate_hrrp:
                X_tr_h = np.load(paths["X_train"], allow_pickle=True) if os.path.exists(paths["X_train"]) else np.array(
                    [])
                X_te_h = np.load(paths["X_test"], allow_pickle=True) if os.path.exists(paths["X_test"]) else np.array(
                    [])

            # MODIFIED: sc_cfg is now current_sc_extraction_config
            sc_cfg = current_sc_extraction_config
            print(f"为 '{dataset_key}' 提取或重新提取散射中心 (方法: {sc_cfg['method']})...")
            print(
                f"  使用SC提取参数: MaxCenters={sc_cfg['max_centers_to_keep']}, Prominence={sc_cfg['peak_prominence']}, MinDist={sc_cfg['min_distance']}")

            sc_params = {
                "prominence": sc_cfg.get("peak_prominence"),
                "min_distance": sc_cfg.get("min_distance"),
                "max_centers_to_keep": sc_cfg.get("max_centers_to_keep"),
                "normalize_hrrp_before_extraction": sc_cfg.get("normalize_hrrp_before_extraction"),
                "normalization_type_for_hrrp": sc_cfg.get("normalization_type_for_hrrp")
            }
            sc_params = {k: v for k, v in sc_params.items() if v is not None}

            if X_tr_h.size > 0:
                X_tr_sc = [extract_scattering_centers_peak_detection(h, **sc_params) for h in
                           tqdm(X_tr_h, desc=f"提取训练SC ({dataset_key})", leave=False, unit="samp")]
                with open(paths["X_train_scatter_centers"], 'wb') as f:
                    pickle.dump(X_tr_sc, f)
                print(f"训练SC .pkl 已更新 (数量: {len(X_tr_sc)})")
            else:
                # print(f"训练数据为空 for '{dataset_key}', 跳过训练SC提取。")
                with open(paths["X_train_scatter_centers"], 'wb') as f:
                    pickle.dump([], f)

            if X_te_h.size > 0:
                X_te_sc = [extract_scattering_centers_peak_detection(h, **sc_params) for h in
                           tqdm(X_te_h, desc=f"提取测试SC ({dataset_key})", leave=False, unit="samp")]
                with open(paths["X_test_scatter_centers"], 'wb') as f:
                    pickle.dump(X_te_sc, f)
                print(f"测试SC .pkl 已更新 (数量: {len(X_te_sc)})")
            else:
                # print(f"测试数据为空 for '{dataset_key}', 跳过测试SC提取。")
                with open(paths["X_test_scatter_centers"], 'wb') as f:
                    pickle.dump([], f)

        elif current_sc_extraction_config["enabled"]:
            print(f"数据集 '{dataset_key}' 跳过SC提取 (文件已存在且无需重新生成)。")
        else:
            print(f"数据集 '{dataset_key}' 跳过SC提取 (已禁用)。")

        if not regenerate_hrrp and not regenerate_sc and current_sc_extraction_config["enabled"]:
            print(f"数据集 '{dataset_key}': HRRP和SC文件均已存在且无需更新。")
        elif not regenerate_hrrp and not current_sc_extraction_config["enabled"]:
            print(f"数据集 '{dataset_key}': HRRP文件已存在，SC提取已禁用。")

    print("数据准备阶段完成。")


def load_npy_data_internal(dataset_name_key, config,
                           paths_dict):  # config not directly used here, but kept for signature
    # ... (no changes in this function)
    try:
        X_train = np.load(paths_dict["X_train"], allow_pickle=True)
        y_train_encoded = np.load(paths_dict["y_train"], allow_pickle=True)
        X_test = np.load(paths_dict["X_test"], allow_pickle=True)
        y_test_encoded = np.load(paths_dict["y_test"], allow_pickle=True)
        with open(paths_dict["label_encoder"], 'rb') as f:
            label_encoder = pickle.load(f)
        class_names = list(label_encoder.classes_)
        y_train_original = label_encoder.inverse_transform(y_train_encoded) if y_train_encoded.size > 0 else np.array(
            [])
        y_test_original = label_encoder.inverse_transform(y_test_encoded) if y_test_encoded.size > 0 else np.array([])
        return X_train, y_train_original, X_test, y_test_original, label_encoder, class_names
    except Exception as e:
        print(f"加载HRRP .npy 或LabelEncoder时出错 for '{dataset_name_key}': {e}")
        return (None,) * 6


def load_processed_data(dataset_name_key, config, load_scattering_centers=True):  # config is CurrentRunConfig
    processed_path = os.path.join(config.PROCESSED_DATA_DIR, dataset_name_key)
    paths = {p_name: os.path.join(processed_path, f"{p_name}.{ext}") for p_name, ext in
             [("X_train", "npy"), ("y_train", "npy"), ("X_test", "npy"), ("y_test", "npy"),
              ("label_encoder", "pkl"), ("X_train_scatter_centers", "pkl"), ("X_test_scatter_centers", "pkl")]}
    req_files = [paths[p] for p in ["X_train", "y_train", "X_test", "y_test", "label_encoder"]]
    if not all(os.path.exists(p) for p in req_files):
        print(f"错误: {dataset_name_key} 的 .npy/pkl 文件不完整于 {processed_path}");
        return (None,) * 8

    X_tr_h, y_tr_o, X_te_h, y_te_o, le, c_names = load_npy_data_internal(dataset_name_key, config, paths)
    if X_tr_h is None and X_te_h is None: return (None,) * 8

    X_tr_sc, X_te_sc = [], []
    # MODIFIED: Access SC config from config.sc_extraction_config
    if load_scattering_centers and config.sc_extraction_config["enabled"]:
        try:
            if os.path.exists(paths["X_train_scatter_centers"]):
                with open(paths["X_train_scatter_centers"], 'rb') as f: X_tr_sc = pickle.load(f)
            # Add similar check for X_tr_h as below for X_te_h if needed

            if os.path.exists(paths["X_test_scatter_centers"]):
                with open(paths["X_test_scatter_centers"], 'rb') as f: X_te_sc = pickle.load(f)
            if X_te_h is not None and X_te_h.size > 0 and not X_te_sc:
                print(f"警告: {dataset_name_key} 的测试散射中心数据为空或未加载，但HRRP测试数据存在。")

        except Exception as e:
            print(f"加载SC文件出错 for '{dataset_name_key}': {e}");
            X_tr_sc, X_te_sc = [], []

    return X_tr_h, y_tr_o, X_te_h, y_te_o, X_tr_sc, X_te_sc, le, c_names


def build_fsl_tasks(
        X_data_pool_hrrp, y_data_pool_original, X_data_pool_sc_list,
        label_encoder, fsl_config, sc_encoding_config, sc_extraction_config,
        random_state
):
    # ... (Prototype calculation logic inside build_fsl_tasks)
    # This function already expects sc_extraction_config and sc_encoding_config correctly.
    # Make sure max_centers_for_proto_feat uses the passed sc_extraction_config:
    np.random.seed(random_state)
    random.seed(random_state)

    n_way = fsl_config["n_way"]
    k_shot_support = fsl_config["k_shot_support"]
    q_shot_query = fsl_config["q_shot_query"]
    num_tasks = fsl_config["num_fsl_tasks"]
    sc_feature_type_for_proto = fsl_config.get("sc_feature_type_for_prototype", "pos_amp_flat")
    # MODIFIED: Get max_centers_to_keep from the passed sc_extraction_config
    max_centers_for_proto_feat = sc_extraction_config["max_centers_to_keep"]

    tasks = []

    if X_data_pool_hrrp is None or X_data_pool_hrrp.size == 0:
        print("警告 (build_fsl_tasks): 提供的数据池为空。无法创建任务。")
        return []

    if sc_extraction_config["enabled"] and X_data_pool_sc_list is None:
        # print("警告 (build_fsl_tasks): SC提取已启用，但X_data_pool_sc_list为None。将视为空SC列表。")
        X_data_pool_sc_list = [[] for _ in range(len(X_data_pool_hrrp))]

    if sc_extraction_config["enabled"] and X_data_pool_sc_list is not None and len(X_data_pool_hrrp) != len(
            X_data_pool_sc_list):
        print(
            f"警告 (build_fsl_tasks): HRRP数据池 ({len(X_data_pool_hrrp)}) 和SC列表 ({len(X_data_pool_sc_list)}) 长度不匹配。")
        if len(X_data_pool_sc_list) < len(X_data_pool_hrrp):
            X_data_pool_sc_list.extend([[] for _ in range(len(X_data_pool_hrrp) - len(X_data_pool_sc_list))])
        else:
            X_data_pool_sc_list = X_data_pool_sc_list[:len(X_data_pool_hrrp)]

    unique_pool_classes = np.unique(y_data_pool_original)
    pool_class_indices = {cls: np.where(y_data_pool_original == cls)[0] for cls in unique_pool_classes}
    valid_classes_for_task_sampling = [
        cls for cls in unique_pool_classes if len(pool_class_indices[cls]) >= (k_shot_support + q_shot_query)
    ]

    if len(valid_classes_for_task_sampling) < n_way:
        print(
            f"警告 (build_fsl_tasks): 数据池中满足 K+Q ({k_shot_support + q_shot_query}) 样本条件的类别数 ({len(valid_classes_for_task_sampling)}) 少于 N-way ({n_way})。")
        return []

    for task_idx in tqdm(range(num_tasks), desc="构建FSL评估任务", leave=False, unit="task"):
        task_support_hrrp_actual_samples, task_support_sc_list_actual_samples, task_support_labels_actual_samples = [], [], []
        task_query_hrrp_list, task_query_sc_list_list, task_query_labels_list = [], [], []
        task_prototype_sc_texts_for_prompt = []
        task_prototype_labels_for_prompt = []

        if len(valid_classes_for_task_sampling) < n_way: break
        selected_classes_for_task = np.random.choice(valid_classes_for_task_sampling, size=n_way, replace=False)

        current_task_fully_valid = True
        for cls_name in selected_classes_for_task:
            class_specific_pool_indices = pool_class_indices[cls_name]

            if len(class_specific_pool_indices) < (k_shot_support + q_shot_query):
                current_task_fully_valid = False;
                break

            selected_indices_for_class_in_pool = np.random.choice(
                class_specific_pool_indices, size=(k_shot_support + q_shot_query), replace=False
            )

            np.random.shuffle(selected_indices_for_class_in_pool)
            current_class_support_indices = selected_indices_for_class_in_pool[:k_shot_support]
            current_class_query_indices = selected_indices_for_class_in_pool[k_shot_support:]

            task_support_hrrp_actual_samples.extend(X_data_pool_hrrp[current_class_support_indices])
            task_support_labels_actual_samples.extend(y_data_pool_original[current_class_support_indices])
            if X_data_pool_sc_list and sc_extraction_config["enabled"]:
                task_support_sc_list_actual_samples.extend(
                    [X_data_pool_sc_list[i] for i in current_class_support_indices])
            else:
                task_support_sc_list_actual_samples.extend([[] for _ in current_class_support_indices])

            task_query_hrrp_list.extend(X_data_pool_hrrp[current_class_query_indices])
            task_query_labels_list.extend(y_data_pool_original[current_class_query_indices])
            if X_data_pool_sc_list and sc_extraction_config["enabled"]:
                task_query_sc_list_list.extend([X_data_pool_sc_list[i] for i in current_class_query_indices])
            else:
                task_query_sc_list_list.extend([[] for _ in current_class_query_indices])

            if k_shot_support > 0 and sc_extraction_config["enabled"]:
                class_k_support_sc_lists = [X_data_pool_sc_list[i] for i in current_class_support_indices if
                                            X_data_pool_sc_list]

                prototype_sc_list_for_class = []  # Default
                if class_k_support_sc_lists:  # Only proceed if there are SC lists for support
                    if k_shot_support == 1:
                        prototype_sc_list_for_class = class_k_support_sc_lists[0]
                    else:  # K > 1
                        try:
                            class_support_feature_vectors = np.array([
                                sc_set_to_feature_vector(sc_l, max_centers_for_proto_feat, sc_feature_type_for_proto)
                                for sc_l in class_k_support_sc_lists if sc_l  # Ensure sc_l is not empty itself
                            ])
                            if class_support_feature_vectors.ndim > 1 and class_support_feature_vectors.shape[
                                0] > 0:  # Check if any valid vectors were created
                                mean_feature_vector = np.mean(class_support_feature_vectors, axis=0)
                                distances_to_mean = [euclidean(fv, mean_feature_vector) for fv in
                                                     class_support_feature_vectors]
                                if distances_to_mean:  # Ensure distances were calculated
                                    closest_sample_idx_in_k = np.argmin(distances_to_mean)
                                    prototype_sc_list_for_class = class_k_support_sc_lists[closest_sample_idx_in_k]
                            elif class_support_feature_vectors.ndim == 1 and class_support_feature_vectors.size > 0 and k_shot_support == 1:  # Single sample, single vector
                                prototype_sc_list_for_class = class_k_support_sc_lists[0]


                        except Exception as e_proto:
                            # print(f"  Error calculating prototype for class {cls_name} in task {task_idx}: {e_proto}. Using empty SC list.")
                            prototype_sc_list_for_class = []

                prototype_sc_text = encode_single_sc_set_to_text(prototype_sc_list_for_class, sc_encoding_config)
                task_prototype_sc_texts_for_prompt.append(prototype_sc_text)
                task_prototype_labels_for_prompt.append(cls_name)
            elif k_shot_support == 0:
                pass
            elif not sc_extraction_config["enabled"] and k_shot_support > 0:
                task_prototype_sc_texts_for_prompt.append("散射中心信息不可用 (提取已禁用)")
                task_prototype_labels_for_prompt.append(cls_name)

        if not current_task_fully_valid: continue

        task_query_sc_texts = encode_all_sc_sets_to_text(task_query_sc_list_list, sc_encoding_config) if \
        sc_extraction_config["enabled"] else ["散射中心信息不可用"] * len(task_query_hrrp_list)

        expected_S_size = n_way * k_shot_support
        expected_Q_size = n_way * q_shot_query
        if len(task_support_labels_actual_samples) != expected_S_size or len(task_query_labels_list) != expected_Q_size:
            # print(f"  警告: 任务 {task_idx+1} 样本数量不符预期。跳过。")
            continue
        if k_shot_support > 0 and (
                len(task_prototype_sc_texts_for_prompt) != n_way or len(task_prototype_labels_for_prompt) != n_way):
            # print(f"  警告: 任务 {task_idx+1} 原型数量不为N。跳过。")
            continue

        tasks.append({
            "support_hrrp_actual": np.array(task_support_hrrp_actual_samples),
            "support_sc_list_actual": task_support_sc_list_actual_samples,
            "support_labels_actual": np.array(task_support_labels_actual_samples),
            "support_prototypes_sc_texts": task_prototype_sc_texts_for_prompt,
            "support_prototypes_labels": task_prototype_labels_for_prompt,
            "query_hrrp": np.array(task_query_hrrp_list),
            "query_sc_list": task_query_sc_list_list,
            "query_labels": np.array(task_query_labels_list),
            "query_sc_texts": task_query_sc_texts,
            "task_classes": selected_classes_for_task,
        })

    if not tasks:
        print(f"警告 (build_fsl_tasks): 未能成功构建任何FSL任务。")
    elif len(tasks) < num_tasks:
        print(f"警告 (build_fsl_tasks): 成功构建了 {len(tasks)} 个FSL任务，少于请求的 {num_tasks} 个。")
    # else: print(f"成功构建了 {len(tasks)} 个FSL任务。") # Reduce verbosity
    return tasks


if __name__ == "__main__":
    class MockConfigDU_FSL_Proto:
        AVAILABLE_DATASETS = {
            "simulated_fsl_proto_test": {"path": "datasets_test/simulated_hrrp_fsl_proto", "data_var": "CoHH",
                                         "original_len": 1000, "max_samples_to_load": 150}}
        TARGET_HRRP_LENGTH = 1000;
        PREPROCESS_MAT_TO_NPY = True;
        PROCESSED_DATA_DIR = "data_processed_test_fsl_proto";
        TEST_SPLIT_SIZE = 0.01
        RANDOM_STATE = 42
        # For this test, sc_extraction_config is directly an attribute of MockConfigDU_FSL_Proto
        sc_extraction_config = {"enabled": True, "method": "peak_detection",
                                "peak_prominence": 0.1, "min_distance": 3,
                                "max_centers_to_keep": 5,
                                "normalize_hrrp_before_extraction": True,
                                "normalization_type_for_hrrp": "max"}
        sc_encoding_config = {"format": "list_of_dicts", "precision_pos": 0, "precision_amp": 3,
                              "TARGET_HRRP_LENGTH_INFO": TARGET_HRRP_LENGTH}
        FSL_TASK_SETUP = {"n_way": 3, "k_shot_support": 2, "q_shot_query": 1, "num_fsl_tasks": 5,
                          "sc_feature_type_for_prototype": "pos_amp_flat"}
        LIMIT_TEST_SAMPLES = None


    config_module_for_data_utils.RANDOM_STATE = MockConfigDU_FSL_Proto.RANDOM_STATE
    mock_config_fsl_proto = MockConfigDU_FSL_Proto()

    test_sim_path_fsl_proto = os.path.join(mock_config_fsl_proto.AVAILABLE_DATASETS["simulated_fsl_proto_test"]["path"])
    os.makedirs(test_sim_path_fsl_proto, exist_ok=True);
    os.makedirs(mock_config_fsl_proto.PROCESSED_DATA_DIR, exist_ok=True)

    num_mock_classes = 5;
    samples_per_mock_class = 30
    if not glob.glob(os.path.join(test_sim_path_fsl_proto, "*.mat")):
        print(f"创建虚拟.mat for FSL Proto测试...");
        from scipy.io import savemat

        for cl_idx in range(num_mock_classes):
            cl = f"ProtoCls{chr(65 + cl_idx)}"
            for i in range(samples_per_mock_class):
                random_data_complex = np.random.rand(mock_config_fsl_proto.TARGET_HRRP_LENGTH) + \
                                      1j * np.random.rand(mock_config_fsl_proto.TARGET_HRRP_LENGTH)
                savemat(os.path.join(test_sim_path_fsl_proto, f"{cl}_s{i}.mat"),
                        {"CoHH": random_data_complex.reshape(-1, 1)})
        print("FSL Proto虚拟文件创建完毕。")

    print("--- 测试 prepare_npy_data_and_scattering_centers for FSL Proto ---");
    prepare_npy_data_and_scattering_centers(mock_config_fsl_proto)  # Pass the mock_config directly

    print("\n--- 测试 load_processed_data for FSL Proto ---");
    load_res_proto = load_processed_data("simulated_fsl_proto_test", mock_config_fsl_proto,
                                         load_scattering_centers=True)

    if load_res_proto[0] is not None or load_res_proto[2] is not None:
        print(f"数据加载成功 for simulated_fsl_proto_test")
        _, _, X_pool_h, y_pool_o, _, X_pool_sc, le_proto, c_names_proto = load_res_proto

        if X_pool_h is None or X_pool_h.size == 0:
            print("错误: 元测试数据池在加载后为空，无法进行build_fsl_tasks测试。")
        else:
            print(f"  数据池 (来自元测试部分) HRRP: {X_pool_h.shape}, 标签: {y_pool_o.shape}")
            if X_pool_sc is not None: print(f"  数据池 SC列表长度: {len(X_pool_sc)}")

            print("\n--- 测试 build_fsl_tasks (with Prototypes) ---")
            fsl_tasks_proto = build_fsl_tasks(
                X_pool_h, y_pool_o, X_pool_sc,
                le_proto, mock_config_fsl_proto.FSL_TASK_SETUP,
                mock_config_fsl_proto.sc_encoding_config,  # Pass specific attribute
                mock_config_fsl_proto.sc_extraction_config,  # Pass specific attribute
                mock_config_fsl_proto.RANDOM_STATE
            )
            if fsl_tasks_proto:
                print(f"成功构建了 {len(fsl_tasks_proto)} 个带原型的FSL任务。")
                cfg = mock_config_fsl_proto.FSL_TASK_SETUP
                expected_Proto_S_count = cfg['n_way']
                expected_Q_size = cfg['n_way'] * cfg['q_shot_query']
                for i, task in enumerate(fsl_tasks_proto[:1]):
                    print(f"  任务 {i + 1}:")
                    print(f"    类别: {task['task_classes']}")
                    print(
                        f"    支撑原型SC文本数: {len(task['support_prototypes_sc_texts'])} (预期: {expected_Proto_S_count})")
                    print(f"    查询集样本数: {len(task['query_labels'])} (预期: {expected_Q_size})")
                    assert len(task['support_prototypes_sc_texts']) == expected_Proto_S_count
                    assert len(task['query_labels']) == expected_Q_size
            else:
                print("未能构建带原型的FSL任务。")
    else:
        print("加载FSL Proto测试数据失败。")