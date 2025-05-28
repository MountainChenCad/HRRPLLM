import numpy as np
from tqdm import tqdm

def encode_single_sc_set_to_text(scattering_centers, encoding_config):
    """
    将单个HRRP样本提取的散射中心列表编码为文本。
    """
    if not scattering_centers:
        return "未检测到显著散射中心。"

    fmt = encoding_config.get("format", "list_of_dicts")
    prec_pos = encoding_config.get("precision_pos", 0)
    prec_amp = encoding_config.get("precision_amp", 3)
    output_parts = []

    if fmt == "list_of_dicts":
        for pos, amp in scattering_centers:
            output_parts.append(f"  {{'range index': {pos:.{prec_pos}f}, 'normalized amplitude': {amp:.{prec_amp}f}}}")
        return "[\n" + ",\n".join(output_parts) + "\n]"
    elif fmt == "condensed_string":
        center_sep = encoding_config.get("center_separator", "; ")
        pos_amp_sep = encoding_config.get("pos_amp_separator", ":")
        for pos, amp in scattering_centers:
            output_parts.append(f"{pos:.{prec_pos}f}{pos_amp_sep}{amp:.{prec_amp}f}")
        return center_sep.join(output_parts)
    else:
        raise ValueError(f"未知的散射中心编码格式: {fmt}")

def encode_all_sc_sets_to_text(all_scattering_centers_data, encoding_config):
    """
    将一批HRRP样本的散射中心数据列表编码为文本列表。
    """
    encoded_texts = []
    # print(f"将散射中心编码为 '{encoding_config.get('format')}' 格式文本...") # 减少打印
    for sc_set in tqdm(all_scattering_centers_data, desc="编码散射中心为文本", leave=False):
        encoded_texts.append(encode_single_sc_set_to_text(sc_set, encoding_config))
    return encoded_texts

if __name__ == "__main__":
    try:
        from config import SCATTERING_CENTER_ENCODING as mock_encoding_config
    except ImportError:
        mock_encoding_config = {"format": "list_of_dicts", "precision_pos": 0, "precision_amp": 3}
        print("无法从config导入SCATTERING_CENTER_ENCODING，使用默认测试。")

    test_sc_data1 = [(10, 0.955), (25, 0.812), (5, 0.750)] 
    test_sc_data2 = [] 
    test_sc_data3 = [(100, 0.5)]
    all_test_data = [test_sc_data1, test_sc_data2, test_sc_data3]

    print(f"--- 测试格式: {mock_encoding_config['format']} ---")
    encoded_results = encode_all_sc_sets_to_text(all_test_data, mock_encoding_config)
    for i, res_text in enumerate(encoded_results):
        print(f"样本 {i+1} 散射中心: {all_test_data[i]}")
        print(f"编码后文本:\n{res_text}\n")