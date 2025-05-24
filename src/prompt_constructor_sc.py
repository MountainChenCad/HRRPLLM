import random
try:
    from config import TARGET_HRRP_LENGTH 
except ImportError:
    TARGET_HRRP_LENGTH = 1000 # Fallback for standalone testing
    print("无法从config导入TARGET_HRRP_LENGTH，使用默认值1000。")


class PromptConstructorSC: 
    def __init__(self, dataset_name_key, class_names, sc_encoding_config):
        self.dataset_name_key = dataset_name_key
        self.class_names = class_names
        self.sc_encoding_config = sc_encoding_config # 来自 config.SCATTERING_CENTER_ENCODING
        self.context_header = self._build_context_header_for_sc()

    def _build_context_header_for_sc(self):
        task_definition = (
            "你是一位雷达目标识别专家，擅长通过分析目标的散射中心特性来识别目标类型。"
            "你的任务是根据提供的主要散射中心信息（位置索引和相对幅度），从候选列表中准确识别出目标。"
        )

        # 从sc_encoding_config获取信息，如果HRRP长度信息也存在那里
        hrrp_length_info = self.sc_encoding_config.get('TARGET_HRRP_LENGTH_INFO', TARGET_HRRP_LENGTH)


        sc_description = (
            "散射中心是目标上雷达回波能量集中的主要区域。它们通常对应于目标的几何不连续点、边缘、角点或强反射面。"
            "通过分析散射中心的数量、它们的相对位置（距离单元索引）以及各自的相对幅度，可以推断目标的尺寸、形状和结构特征。"
            f"在本任务中，提供的散射中心信息是从长度为 {hrrp_length_info} 的一维高分辨率距离像（HRRP）中提取并按幅度降序排列的。" # 使用变量
            "“位置索引”从0开始计数，代表在原始HRRP序列中的位置。"
        )

        if "simulated" in self.dataset_name_key.lower():
            dataset_info = f"当前分析的数据来源于 **仿真HRRP的散射中心数据**。"
        elif "measured" in self.dataset_name_key.lower():
            dataset_info = f"当前分析的数据来源于 **实测HRRP的散射中心数据**。"
        else:
            dataset_info = "当前分析的数据为HRRP的散射中心数据。"
        
        dataset_info += f" 候选目标类别包括：`{', '.join(self.class_names)}`。"

        reasoning_guidance = (
            "**推理步骤与要求：**\n"
            "1.  **审查散射中心分布**：仔细观察“测试样本散射中心”部分提供的数据。关注：\n"
            "    *   检测到的散射中心数量。\n"
            "    *   最强几个散射中心的位置索引及其相对幅度。\n"
            f"    *   散射中心在整个目标长度（0到{hrrp_length_info-1}）上的大致分布模式（例如，集中在前端、后端、均匀分布等）。\n" # 使用变量
            "2.  **参考邻近样本（如果提供）**：将测试样本的散射中心特征与“邻近训练样本参考”中的已知类别样本进行对比。\n"
            "3.  **综合判断**：结合你对不同类型目标散射中心分布规律的理解，判断测试样本最符合哪个候选类别。\n"
            "4.  **输出格式**：\n"
            "    *   在你的回答的第一行，请明确给出预测的目标类别，格式为：`预测目标类别：[此处填写候选类别中的一个名称]`\n"
            "    *   （可选，但推荐）在后续行中，请简要陈述你做出此判断的主要理由，例如基于散射中心的数量、位置或特定模式。"
        )
        
        header = (
            f"{task_definition}\n\n"
            f"**散射中心特性概述：**\n{sc_description}\n\n"
            f"**当前数据集与任务：**\n{dataset_info}\n\n"
            f"{reasoning_guidance}\n\n"
            f"------------------------------------\n"
        )
        return header

    def construct_prompt_with_sc(self, query_sc_text, neighbor_sc_examples=None): # <--- 确认方法名
        """
        构建基于散射中心的完整prompt。
        Args:
            query_sc_text (str): 当前待查询样本散射中心的文本编码。
            neighbor_sc_examples (list of tuples, optional): [(sc_text_1, label_1), ...]。
        """
        prompt = self.context_header

        if neighbor_sc_examples and len(neighbor_sc_examples) > 0:
            prompt += "**邻近训练样本参考（基于散射中心）：**\n"
            for i, (neighbor_text, neighbor_label) in enumerate(neighbor_sc_examples):
                prompt += f"\n--- 参考样本 {i+1} ---\n"
                prompt += f"已知目标类别：`{neighbor_label}`\n"
                prompt += f"其主要散射中心信息：\n{neighbor_text}\n"
            prompt += "------------------------------------\n"
        
        prompt += "**测试样本散射中心（请基于此进行预测）：**\n"
        prompt += f"{query_sc_text}\n\n"
        prompt += "预测目标类别：" # LLM将在此后继续

        return prompt

if __name__ == "__main__":
    try: 
        from config import SCATTERING_CENTER_ENCODING as mock_sc_encoding_config
        from config import TARGET_HRRP_LENGTH as mock_target_hrrp_length # 导入以用于测试
    except ImportError: 
        mock_sc_encoding_config = {"format": "list_of_dicts", "precision_pos": 0, "precision_amp": 3}
        mock_target_hrrp_length = 1000 # 如果config不可用，提供默认值
        print("无法从config导入，使用默认测试配置。")

    # 如果sc_encoding_config需要TARGET_HRRP_LENGTH，确保它被传递或可访问
    # 简单的做法是在mock_sc_encoding_config中加入它
    mock_sc_encoding_config['TARGET_HRRP_LENGTH_INFO'] = mock_target_hrrp_length


    from scattering_center_encoder import encode_single_sc_set_to_text 

    mock_dataset_name_sc = "simulated_sc_test"
    mock_class_names_sc = ["F-22", "T-72", "MQ-9"]
    
    constructor_sc = PromptConstructorSC(mock_dataset_name_sc, mock_class_names_sc, mock_sc_encoding_config)
    
    dummy_sc_1 = [(100, 0.9), (150, 0.7), (50, 0.6)]
    dummy_sc_2 = [(200, 0.95), (210, 0.88), (190, 0.8), (500, 0.5)]
    query_sc = [(98, 0.88), (152, 0.72), (45, 0.65)]

    sc_text_1 = encode_single_sc_set_to_text(dummy_sc_1, mock_sc_encoding_config)
    sc_text_2 = encode_single_sc_set_to_text(dummy_sc_2, mock_sc_encoding_config)
    query_sc_text = encode_single_sc_set_to_text(query_sc, mock_sc_encoding_config)

    mock_neighbors_sc = [
        (sc_text_1, "F-22"),
        (sc_text_2, "T-72")
    ]

    print("--- 测试 0-shot SC Prompt ---")
    prompt_0shot_sc = constructor_sc.construct_prompt_with_sc(query_sc_text) # 调用
    with open("test_prompt_0shot_sc.txt", "w", encoding="utf-8") as f:
        f.write(prompt_0shot_sc)
    print("0-shot SC prompt 已保存到 test_prompt_0shot_sc.txt")

    print("\n--- 测试 K-shot SC Prompt ---")
    prompt_kshot_sc = constructor_sc.construct_prompt_with_sc(query_sc_text, mock_neighbors_sc) # 调用
    with open("test_prompt_kshot_sc.txt", "w", encoding="utf-8") as f:
        f.write(prompt_kshot_sc)
    print("K-shot SC prompt 已保存到 test_prompt_kshot_sc.txt")