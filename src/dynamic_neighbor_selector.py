import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import random

from feature_extractor import extract_scattering_centers_peak_detection

try:
    from dtaidistance import dtw
    DTAIDISTANCE_AVAILABLE = True
except ImportError:
    DTAIDISTANCE_AVAILABLE = False

def calculate_dtw_distance_hrrp(seq1, seq2):
    if not DTAIDISTANCE_AVAILABLE: raise EnvironmentError("DTW需要dtaidistance")
    return dtw.distance_fast(np.array(seq1, dtype=np.double), np.array(seq2, dtype=np.double), use_pruning=True)

def sc_set_to_feature_vector(sc_set, max_centers, feature_type="pos_amp_flat"):
    if not sc_set: return np.zeros(max_centers * 2 if feature_type == "pos_amp_flat" else max_centers)
    if feature_type == "pos_amp_flat":
        ft = np.zeros(max_centers * 2)
        for i in range(min(len(sc_set),max_centers)): ft[i*2]=sc_set[i][0]; ft[i*2+1]=sc_set[i][1]
        return ft
    elif feature_type == "amps_only_padded":
        ft = np.zeros(max_centers)
        for i in range(min(len(sc_set), max_centers)): ft[i] = sc_set[i][1]
        return ft
    elif feature_type == "pos_only_padded":
        ft = np.zeros(max_centers)
        for i in range(min(len(sc_set), max_centers)): ft[i] = sc_set[i][0]
        return ft
    raise ValueError(f"不支持的SC特征类型: {feature_type}")

def select_k_most_similar_from_support_pool(
    query_hrrp_if_needed,           # 当前查询样本的原始HRRP
    query_sc_list_if_needed,        # 当前查询样本的SC列表
    support_pool_hrrps,             # 整个支撑集池的HRRPs
    support_pool_sc_lists,          # 整个支撑集池的SC列表
    support_pool_labels_original,   # 整个支撑集池的标签
    support_pool_sc_texts_for_prompt, # 整个支撑集池的SC文本 (用于最终放入Prompt)
    total_k_shots_for_prompt,       # 总共要为Prompt选择多少个示例
    similarity_metric,
    sc_extraction_config,           # 完整的SC提取配置
    sc_feature_type_for_similarity  # 例如 "pos_amp_flat"
    ):
    """
    从整个支撑集池中，为给定的查询样本选择总共k个最相似的示例放入Prompt。
    这些示例可能来自不同的类别。
    """
    if total_k_shots_for_prompt == 0 or not support_pool_hrrps.any():
        return []

    # 1. 准备查询样本的特征 (用于计算相似度)
    query_feature_for_sim_calc = None
    sc_params_for_extraction = {k:v for k,v in sc_extraction_config.items() if k not in ["enabled","method"]}

    if similarity_metric == "euclidean_on_sc":
        current_query_sc_list = query_sc_list_if_needed
        if current_query_sc_list is None: 
            if query_hrrp_if_needed is None: raise ValueError("euclidean_on_sc需要查询HRRP或SC列表")
            current_query_sc_list = extract_scattering_centers_peak_detection(query_hrrp_if_needed, **sc_params_for_extraction)
        query_feature_for_sim_calc = sc_set_to_feature_vector(
            current_query_sc_list, sc_extraction_config["max_centers_to_keep"], sc_feature_type_for_similarity
        )
    elif similarity_metric in ["dtw_on_hrrp", "euclidean_on_hrrp"]:
        if query_hrrp_if_needed is None: raise ValueError(f"{similarity_metric} 需要查询HRRP")
        query_feature_for_sim_calc = query_hrrp_if_needed
    else:
        raise ValueError(f"不支持的相似度度量: {similarity_metric}")

    # 2. 准备支撑集池的特征 (用于计算相似度)
    support_pool_features_for_sim_calc = []
    if similarity_metric == "euclidean_on_sc":
        support_pool_features_for_sim_calc = np.array([
            sc_set_to_feature_vector(sc_l, sc_extraction_config["max_centers_to_keep"], sc_feature_type_for_similarity) 
            for sc_l in support_pool_sc_lists # 使用传入的 support_pool_sc_lists
        ])
    elif "on_hrrp" in similarity_metric:
        support_pool_features_for_sim_calc = support_pool_hrrps # 使用传入的 support_pool_hrrps
    
    if len(support_pool_features_for_sim_calc) == 0: return []

    # 3. 计算查询样本与支撑集池中所有样本的距离
    distances_to_query = []
    # print(f"  Query feat shape: {query_feature_for_sim_calc.shape if query_feature_for_sim_calc is not None else 'None'}")
    for i in range(len(support_pool_features_for_sim_calc)):
        support_sample_feature = support_pool_features_for_sim_calc[i]
        # print(f"  Support feat {i} shape: {support_sample_feature.shape if support_sample_feature is not None else 'None'}")
        dist = float('inf')
        if query_feature_for_sim_calc is not None and support_sample_feature is not None:
            if "euclidean" in similarity_metric:
                dist = euclidean(query_feature_for_sim_calc, support_sample_feature)
            elif similarity_metric == "dtw_on_hrrp":
                try: dist = calculate_dtw_distance_hrrp(query_feature_for_sim_calc, support_sample_feature)
                except Exception as e: print(f"DTW计算错误 (support idx {i}): {e}"); dist = float('inf')
        distances_to_query.append((dist, i)) # (距离, 在支撑集池中的索引)

    # 4. 按距离排序并选择前 total_k_shots_for_prompt 个
    distances_to_query.sort(key=lambda x: x[0])
    
    selected_prompt_examples = []
    num_to_select = min(total_k_shots_for_prompt, len(distances_to_query))

    for i in range(num_to_select):
        _, pool_idx = distances_to_query[i] # pool_idx 是在原始支撑集池中的索引
        sc_text_for_prompt = support_pool_sc_texts_for_prompt[pool_idx]
        label_for_prompt = support_pool_labels_original[pool_idx]
        selected_prompt_examples.append((sc_text_for_prompt, label_for_prompt))
            
    return selected_prompt_examples

if __name__ == "__main__":
    class MockSCConfigDNS: # DNS for Dynamic Neighbor Selector
        SCATTERING_CENTER_EXTRACTION = {"enabled": True, "method": "peak_detection", "peak_prominence": 0.1, "peak_min_distance": 3, "max_centers_to_keep": 5, "normalize_hrrp_before_extraction": True, "normalization_type_for_hrrp": "max"}
        FSL_TASK_SETUP = {"k_shots_for_prompt_from_task_support": 3, "similarity_metric": "euclidean_on_sc", "sc_feature_type_for_similarity": "pos_amp_flat"}
        SCATTERING_CENTER_ENCODING = {"format": "list_of_dicts", "precision_pos": 0, "precision_amp": 2}


    mock_config_dns = MockSCConfigDNS()
    sc_ext_cfg_dns = mock_config_dns.SCATTERING_CENTER_EXTRACTION
    fsl_cfg_dns = mock_config_dns.FSL_TASK_SETUP

    query_hrrp_dns = np.array([0.1,0.5,0.2,0.9,0.3,0.1,0.7,0.4,0.1,0.6,0.15,0.75])
    query_sc_list_dns = extract_scattering_centers_peak_detection(query_hrrp_dns, **{k:v for k,v in sc_ext_cfg_dns.items() if k not in ["enabled","method"]})

    support_hrrps_dns = [np.array([0.1,0.55,0.22,0.85,0.33,0.15,0.75,0.44,0.12,0.66,0.12,0.72]), 
                         np.array([0.8,0.2,0.7,0.1,0.6,0.3,0.5,0.1,0.9,0.2,0.77,0.11]), 
                         np.array([0.15,0.45,0.18,0.92,0.28,0.12,0.68,0.41,0.09,0.63,0.22,0.65]),
                         np.array([0.05,0.1,0.8,0.7,0.2,0.5,0.3,0.9,0.1,0.4,0.75,0.25])]
    support_labels_dns = ["TypeA", "TypeB", "TypeA", "TypeC"]
    support_sc_lists_dns = [extract_scattering_centers_peak_detection(h, **{k:v for k,v in sc_ext_cfg_dns.items() if k not in ["enabled","method"]}) for h in support_hrrps_dns]
    
    from scattering_center_encoder import encode_all_sc_sets_to_text
    support_sc_texts_dns = encode_all_sc_sets_to_text(support_sc_lists_dns, mock_config_dns.SCATTERING_CENTER_ENCODING)
    
    print("--- 测试 select_k_most_similar_from_support_pool ---")
    examples_selected = select_k_most_similar_from_support_pool(
        query_hrrp_if_needed=query_hrrp_dns,
        query_sc_list_if_needed=query_sc_list_dns,
        support_pool_hrrps=np.array(support_hrrps_dns), # 确保是numpy array
        support_pool_sc_lists=support_sc_lists_dns,
        support_pool_labels_original=np.array(support_labels_dns), # 确保是numpy array
        support_pool_sc_texts_for_prompt=support_sc_texts_dns,
        total_k_shots_for_prompt=fsl_cfg_dns["k_shots_for_prompt_from_task_support"],
        similarity_metric=fsl_cfg_dns["similarity_metric"],
        sc_extraction_config=sc_ext_cfg_dns,
        sc_feature_type_for_similarity=fsl_cfg_dns["sc_feature_type_for_similarity"]
    )
    print(f"\n为查询样本动态选择的 {len(examples_selected)} 个 (总共K个) Few-Shot 示例:")
    for sc_text, label in examples_selected: print(f"类别: {label}\nSC文本(部分):\n{sc_text[:100]}...\n")
    assert len(examples_selected) == fsl_cfg_dns["k_shots_for_prompt_from_task_support"]
    print("select_k_most_similar_from_support_pool 测试通过。")
