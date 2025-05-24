import numpy as np
from tqdm import tqdm

def normalize_hrrp_sample(hrrp_sample, normalize_enabled=True, normalization_type="max"):
    """对单个HRRP样本进行幅度归一化。"""
    if not normalize_enabled or hrrp_sample is None or len(hrrp_sample) == 0:
        return hrrp_sample
    
    hrrp_sample_float = hrrp_sample.astype(float)
    
    if normalization_type == "max":
        max_val = np.max(hrrp_sample_float)
        if max_val > 1e-9: # 避免除以零
            return hrrp_sample_float / max_val
        else:
            return hrrp_sample_float # 如果全是零或接近零，不处理
    elif normalization_type == "energy":
        energy = np.sqrt(np.sum(hrrp_sample_float**2))
        if energy > 1e-9:
            return hrrp_sample_float / energy
        else:
            return hrrp_sample_float
    else:
        raise ValueError("未知的归一化类型. 请选择 'max' 或 'energy'.")

def hrrp_sample_to_text(hrrp_sample, precision=3, use_space_separator=True, value_separator=", "):
    """将单个（可能已归一化）HRRP样本转换为文本序列。"""
    if hrrp_sample is None:
        return ""
        
    # 1. 精度控制和整数转换
    scale_factor = 10**precision
    int_values = np.round(hrrp_sample * scale_factor).astype(int)
    
    # 2. 数字字符串化
    text_parts = []
    for val in int_values:
        s_val = str(val)
        if use_space_separator:
            # 确保即使是单个数字也被分隔开，例如 0 -> "0", 10 -> "1 0"
            text_parts.append(" ".join(list(s_val)))
        else:
            text_parts.append(s_val)
            
    # 3. 序列拼接
    return value_separator.join(text_parts)

def preprocess_hrrps_for_llm(hrrp_samples, 
                            normalize_enabled=True, 
                            normalization_type="max",
                            precision=3, 
                            use_space_separator=True, 
                            value_separator=", "):
    """
    对一批HRRP样本进行完整的预处理流程，转换为LLM输入文本。
    """
    processed_texts = []
    print("正在预处理HRRP样本并转换为文本...")
    for sample in tqdm(hrrp_samples, desc="预处理"):
        normalized_sample = normalize_hrrp_sample(sample, normalize_enabled, normalization_type)
        text_sequence = hrrp_sample_to_text(normalized_sample, precision, use_space_separator, value_separator)
        processed_texts.append(text_sequence)
    return processed_texts

def preprocess_hrrps_for_baseline(hrrp_samples, normalize_enabled=True, normalization_type="max"):
    """为传统基线方法预处理HRRP样本（仅归一化）。"""
    processed_samples = []
    print("正在为基线方法预处理HRRP样本...")
    for sample in tqdm(hrrp_samples, desc="基线预处理"):
        normalized_sample = normalize_hrrp_sample(sample, normalize_enabled, normalization_type)
        processed_samples.append(normalized_sample)
    return np.array(processed_samples)


if __name__ == '__main__':
    from config import PREPROCESSING
    
    # 测试
    test_hrrp1 = np.array([0.0, 0.1, 0.5, 1.0, 0.3, 0.05])
    test_hrrp2 = np.array([10, 20, 5, 15]) * 100 # 未归一化

    print("--- LLM 预处理测试 ---")
    # 测试归一化 + 空格分隔
    text1_norm_space = hrrp_sample_to_text(
        normalize_hrrp_sample(test_hrrp1, True, "max"),
        PREPROCESSING['precision'], True, PREPROCESSING['value_separator']
    )
    print(f"原始: {test_hrrp1}\n归一化+空格: {text1_norm_space}")

    # 测试不归一化 + 无空格分隔
    text2_nonorm_nospace = hrrp_sample_to_text(
        normalize_hrrp_sample(test_hrrp2, False), # 禁用归一化
        PREPROCESSING['precision'], False, PREPROCESSING['value_separator']
    )
    print(f"原始: {test_hrrp2}\n无归一化+无空格: {text2_nonorm_nospace}")

    # 测试整个pipeline
    all_samples = [test_hrrp1, test_hrrp2]
    all_texts = preprocess_hrrps_for_llm(all_samples, 
                                       PREPROCESSING['normalize'],
                                       PREPROCESSING['normalization_type'],
                                       PREPROCESSING['precision'],
                                       PREPROCESSING['use_space_separator'],
                                       PREPROCESSING['value_separator'])
    print("\nPipelin 输出:")
    for original, text_repr in zip(all_samples, all_texts):
        print(f"原始: {original} -> 文本: {text_repr}")

    print("\n--- 基线预处理测试 ---")
    baseline_features = preprocess_hrrps_for_baseline([test_hrrp1, test_hrrp2],
                                                  PREPROCESSING['normalize'],
                                                  PREPROCESSING['normalization_type'])
    print("基线特征 (仅归一化):")
    for bf in baseline_features:
        print(bf)