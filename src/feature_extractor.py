import numpy as np
from scipy.signal import find_peaks

def _normalize_hrrp_internal(hrrp_sample, norm_type="max"):
    if norm_type == "max":
        max_val = np.max(hrrp_sample) 
        if max_val > 1e-9: 
            return hrrp_sample / max_val
        else:
            return hrrp_sample 
    elif norm_type == "energy":
        energy = np.sqrt(np.sum(hrrp_sample**2))
        if energy > 1e-9:
            return hrrp_sample / energy
        else:
            return hrrp_sample
    else:
        raise ValueError(f"未知的归一化类型: {norm_type}")


def extract_scattering_centers_peak_detection(
    hrrp_sample, 
    prominence=0.1,         # Parameter name in function
    min_distance=5,         # Parameter name in function
    max_centers_to_keep=10, 
    normalize_hrrp_before_extraction=True, 
    normalization_type_for_hrrp="max" 
    ):
    if hrrp_sample is None or len(hrrp_sample) == 0:
        return []

    processed_hrrp = hrrp_sample.astype(float) 
    hrrp_for_peak_detection = processed_hrrp

    if normalize_hrrp_before_extraction:
        hrrp_for_peak_detection = _normalize_hrrp_internal(processed_hrrp, normalization_type_for_hrrp)
        if np.all(hrrp_for_peak_detection < 1e-9): 
            return []

    peaks_indices, properties = find_peaks(
        hrrp_for_peak_detection, 
        prominence=prominence, 
        distance=min_distance # Uses the parameter name directly
    )

    if len(peaks_indices) == 0:
        return []

    peak_amplitudes = hrrp_for_peak_detection[peaks_indices] 
    centers = sorted(list(zip(peaks_indices, peak_amplitudes)), key=lambda x: x[1], reverse=True)
    
    if max_centers_to_keep is not None and len(centers) > max_centers_to_keep:
        centers = centers[:max_centers_to_keep]
        
    return centers 

if __name__ == "__main__":
    try:
        from config import SCATTERING_CENTER_EXTRACTION as test_sc_config_module
    except ImportError:
        print("警告: 无法从config导入SCATTERING_CENTER_EXTRACTION，使用默认测试参数。")
        # MODIFIED: test_sc_config keys should align with how they are used or function parameters
        test_sc_config_module = {
            "peak_prominence": 0.15, # This key is used in data_utils to map to "prominence"
            "min_distance": 5,       # This key is used in data_utils to map to "min_distance"
            "max_centers_to_keep": 10,
            "normalize_hrrp_before_extraction": True,
            "normalization_type_for_hrrp": "max"
        }

    test_hrrp_magnitude = np.array([0.1, 0.2, 0.1, 0.8, 0.3, 0.2, 0.1, 0.9, 0.7, 0.2, 0.05, 0.6, 0.1, 0.05, 1.0, 0.1])
    print(f"测试用 HRRP (幅度): {test_hrrp_magnitude}")

    print(f"\n--- 测试1: 使用配置进行提取 ---")
    # MODIFIED: Map from test_sc_config_module keys to function parameter names
    params_for_extraction = {
        "prominence": test_sc_config_module.get("peak_prominence"), 
        "min_distance": test_sc_config_module.get("min_distance"),
        "max_centers_to_keep": test_sc_config_module.get("max_centers_to_keep"),
        "normalize_hrrp_before_extraction": test_sc_config_module.get("normalize_hrrp_before_extraction"),
        "normalization_type_for_hrrp": test_sc_config_module.get("normalization_type_for_hrrp")
    }
    params_for_extraction = {k:v for k,v in params_for_extraction.items() if v is not None}


    centers1 = extract_scattering_centers_peak_detection(
        test_hrrp_magnitude.copy(), 
        **params_for_extraction
    )
    if centers1:
        print(f"提取的散射中心 (配置: 归一化={params_for_extraction.get('normalize_hrrp_before_extraction')}, 类型={params_for_extraction.get('normalization_type_for_hrrp')}):")
        for pos, amp in centers1: print(f"  位置 (索引): {pos}, 幅度: {amp:.3f}")
    else: print("  未找到散射中心。")

    print(f"\n--- 测试2: 不归一化提取 ---")
    params_no_norm = params_for_extraction.copy()
    params_no_norm["normalize_hrrp_before_extraction"] = False
    centers2 = extract_scattering_centers_peak_detection(
        test_hrrp_magnitude.copy(),
        **params_no_norm
    )
    if centers2:
        print(f"提取的散射中心 (不归一化):")
        for pos, amp in centers2: print(f"  位置 (索引): {pos}, 幅度: {amp:.3f}")
    else: print("  未找到散射中心。")

    print(f"\n--- 测试3: 全零输入 ---")
    zero_hrrp = np.zeros_like(test_hrrp_magnitude)
    centers3 = extract_scattering_centers_peak_detection(zero_hrrp, **params_for_extraction)
    if not centers3: print("全零输入，未找到散射中心 (符合预期)。")
    else: print(f"全零输入但找到散射中心: {centers3} (异常)")