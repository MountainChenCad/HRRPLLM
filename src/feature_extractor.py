import numpy as np
from scipy.signal import find_peaks

def _normalize_hrrp_internal(hrrp_sample, norm_type="max"):
    """
    内部辅助函数，用于对HRRP样本进行幅度归一化。
    确保输入是浮点数并且已取绝对值（如果需要）。
    """
    if norm_type == "max":
        max_val = np.max(hrrp_sample) # 假设输入已是正数/幅度
        if max_val > 1e-9:
            return hrrp_sample / max_val
        else:
            return hrrp_sample # 全零或接近零
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
    prominence=0.1, 
    min_distance=5, 
    max_centers_to_keep=10, 
    normalize_hrrp_before_extraction=True, 
    normalization_type_for_hrrp="max" 
    ):
    """
    使用简单的峰值检测提取散射中心。
    """
    if hrrp_sample is None or len(hrrp_sample) == 0:
        return []

    processed_hrrp = np.abs(hrrp_sample.astype(float))
    hrrp_for_peak_detection = processed_hrrp

    if normalize_hrrp_before_extraction:
        hrrp_for_peak_detection = _normalize_hrrp_internal(processed_hrrp, normalization_type_for_hrrp)
        if np.all(hrrp_for_peak_detection < 1e-9): 
            return []

    peaks_indices, properties = find_peaks(
        hrrp_for_peak_detection, 
        prominence=prominence, 
        distance=min_distance
    )

    if len(peaks_indices) == 0:
        return []

    peak_amplitudes = hrrp_for_peak_detection[peaks_indices]
    centers = sorted(zip(peaks_indices, peak_amplitudes), key=lambda x: x[1], reverse=True)
    
    if max_centers_to_keep is not None and len(centers) > max_centers_to_keep:
        centers = centers[:max_centers_to_keep]
        
    return centers 

if __name__ == "__main__":
    try:
        from config import SCATTERING_CENTER_EXTRACTION as test_sc_config
    except ImportError:
        print("警告: 无法从config导入SCATTERING_CENTER_EXTRACTION，使用默认测试参数。")
        test_sc_config = {
            "peak_prominence": 0.15,
            "peak_min_distance": 5,
            "max_centers_to_keep": 10,
            "normalize_hrrp_before_extraction": True,
            "normalization_type_for_hrrp": "max"
        }

    test_hrrp_orig = np.array([10, 20, 10, 80, 30, 20, 10, 90, 70, 20, 5, 60, 10, 5, 100, 10])
    print(f"原始HRRP: {test_hrrp_orig}")

    print(f"\n--- 测试1: 使用配置进行提取 ---")
    centers1 = extract_scattering_centers_peak_detection(
        test_hrrp_orig.copy(), 
        prominence=test_sc_config["peak_prominence"],
        min_distance=test_sc_config["peak_min_distance"],
        max_centers_to_keep=test_sc_config["max_centers_to_keep"],
        normalize_hrrp_before_extraction=test_sc_config["normalize_hrrp_before_extraction"],
        normalization_type_for_hrrp=test_sc_config["normalization_type_for_hrrp"]
    )
    if centers1:
        print(f"提取的散射中心 (配置: 归一化={test_sc_config['normalize_hrrp_before_extraction']}, 类型={test_sc_config['normalization_type_for_hrrp']}):")
        for pos, amp in centers1: print(f"  位置 (索引): {pos}, 幅度: {amp:.3f}")
    else: print("  未找到散射中心。")