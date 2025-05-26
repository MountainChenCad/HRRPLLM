import random
try:
    # 尝试从config导入，主要为了TARGET_HRRP_LENGTH作为后备
    from config import TARGET_HRRP_LENGTH as FALLBACK_TARGET_HRRP_LENGTH
except ImportError:
    FALLBACK_TARGET_HRRP_LENGTH = 1000 # Fallback for standalone testing
    print("无法从config导入FALLBACK_TARGET_HRRP_LENGTH，使用默认值1000。")


class PromptConstructorSC: 
    def __init__(self, dataset_name_key, class_names_for_task, sc_encoding_config):
        """
        Args:
            dataset_name_key (str): 数据集标识符.
            class_names_for_task (list): 当前FSL任务中涉及的类别名称列表.
            sc_encoding_config (dict): 散射中心编码配置.
        """
        self.dataset_name_key = dataset_name_key
        self.class_names_for_task = class_names_for_task # 当前任务的类别
        self.sc_encoding_config = sc_encoding_config
        # TARGET_HRRP_LENGTH_INFO 应该在 sc_encoding_config 中提供
        self.hrrp_length_info = self.sc_encoding_config.get('TARGET_HRRP_LENGTH_INFO', FALLBACK_TARGET_HRRP_LENGTH)
        self.context_header = self._build_context_header_for_sc()

    def _build_context_header_for_sc(self):
        task_definition = (
            "你是一位雷达目标识别专家，擅长通过分析目标的散射中心特性来识别目标类型。"
            "你的任务是根据提供的主要散射中心信息（位置索引和相对幅度），从候选列表中准确识别出目标。"
        )
        
        sc_description = (
            "散射中心是目标上雷达回波能量集中的主要区域。它们通常对应于目标的几何不连续点、边缘、角点或强反射面。"
            "通过分析散射中心的数量、它们的相对位置（距离单元索引）以及各自的相对幅度，可以推断目标的尺寸、形状和结构特征。"
            f"在本任务中，提供的散射中心信息是从长度为 {self.hrrp_length_info} 的一维高分辨率距离像（HRRP）中提取并按幅度降序排列的。"
            "“位置索引”从0开始计数，代表在原始HRRP序列中的位置。"
            "“相对幅度”是经过归一化处理的（最大值为1）。"
        )

        if "simulated" in self.dataset_name_key.lower():
            dataset_info_prefix = f"当前分析的数据来源于 **仿真HRRP的散射中心数据**。"
        elif "measured" in self.dataset_name_key.lower():
            dataset_info_prefix = f"当前分析的数据来源于 **实测HRRP的散射中心数据**。"
        else:
            dataset_info_prefix = "当前分析的数据为HRRP的散射中心数据。"
        
        # 使用任务特定的类别列表
        dataset_info = dataset_info_prefix + f" 候选目标类别包括：`{', '.join(self.class_names_for_task)}`。"

        reasoning_guidance = (
            # "Let's think step by step.\n"
            "**推理步骤与要求：**\n"
            "1.  **审查测试样本散射中心**：仔细观察“测试样本散射中心”部分提供的数据。关注：\n"
            "    *   检测到的散射中心数量。\n"
            "    *   最强几个散射中心的位置索引及其相对幅度。\n"
            f"    *   散射中心在整个目标长度（0到{self.hrrp_length_info-1}）上的大致分布模式（例如，集中在前端、后端、均匀分布等）。\n"
            "2.  **参考支撑样本**：将测试样本的散射中心特征与“邻近训练样本参考”中的已知类别样本进行对比。\n"
            "    *   注意每个参考样本的已知类别，并比较其散射中心模式与测试样本的相似性。\n"
            "3.  **综合判断**：结合你对不同类型目标散射中心分布规律的理解，并基于与参考样本的对比，判断测试样本最符合哪个候选类别。\n"
            "4.  **输出格式**：\n"
            "    *   在你的回答的第一行，请明确给出预测的目标类别，格式为：`预测目标类别：[此处填写候选类别中的一个名称]`\n"
            "    *   在后续行中，请简要陈述你做出此判断的主要理由，例如基于散射中心的数量、位置、特定模式，或与哪个参考样本最相似。"
        )
        
        header = (
            f"{task_definition}\n\n"
            f"**散射中心特性概述：**\n{sc_description}\n\n"
            f"**当前数据集与任务：**\n{dataset_info}\n\n"
            f"{reasoning_guidance}\n\n"
            f"------------------------------------\n"
        )
        return header

    def construct_prompt_with_sc(self, query_sc_text, neighbor_sc_examples=None):
        """
        构建基于散射中心的完整prompt。
        Args:
            query_sc_text (str): 当前待查询样本散射中心的文本编码。
            neighbor_sc_examples (list of tuples, optional): [(sc_text_1, label_1), ...]。
                                                        这些是当前FSL任务的支撑样本。
        """
        prompt = self.context_header # context_header 现在使用 task-specific class_names

        if neighbor_sc_examples and len(neighbor_sc_examples) > 0:
            prompt += "**邻近训练样本参考（支撑集）：**\n"
            for i, (neighbor_text, neighbor_label) in enumerate(neighbor_sc_examples):
                prompt += f"\n--- 参考样本 {i+1} ---\n"
                prompt += f"已知目标类别：`{neighbor_label}`\n"
                prompt += f"其主要散射中心信息：\n{neighbor_text}\n"
            prompt += "------------------------------------\n"
        else:
            prompt += "**注意：本次预测无邻近训练样本参考（0-shot任务）。请基于散射中心特性概述和自身知识进行判断。**\n"
            prompt += "------------------------------------\n"
        
        prompt += "**测试样本散射中心（请基于此进行预测）：**\n"
        prompt += f"{query_sc_text}\n\n"
        prompt += "请严格按照输出格式要求回答。\n" # 强调格式
        prompt += "预测目标类别：" # LLM将在此后继续

        return prompt

if __name__ == "__main__":
    try: 
        from config import SCATTERING_CENTER_ENCODING as mock_sc_encoding_config_main
        from config import TARGET_HRRP_LENGTH as mock_target_hrrp_length_main
        # 确保 TARGET_HRRP_LENGTH_INFO 在 mock 配置中
        if 'TARGET_HRRP_LENGTH_INFO' not in mock_sc_encoding_config_main:
             mock_sc_encoding_config_main['TARGET_HRRP_LENGTH_INFO'] = mock_target_hrrp_length_main
    except ImportError: 
        mock_target_hrrp_length_main = 1000
        mock_sc_encoding_config_main = {
            "format": "list_of_dicts", 
            "precision_pos": 0, 
            "precision_amp": 3,
            "TARGET_HRRP_LENGTH_INFO": mock_target_hrrp_length_main
        }
        print("无法从config导入，使用默认测试配置。")

    from scattering_center_encoder import encode_single_sc_set_to_text # 确保导入

    mock_dataset_name_sc = "simulated_sc_test"
    # 假设这是一个3-way的任务
    mock_class_names_for_task_sc = ["F-22", "T-72", "MQ-9"] 
    
    constructor_sc = PromptConstructorSC(mock_dataset_name_sc, mock_class_names_for_task_sc, mock_sc_encoding_config_main)
    
    dummy_sc_1 = [(100, 0.9), (150, 0.7), (50, 0.6)]
    dummy_sc_2 = [(200, 0.95), (210, 0.88), (190, 0.8), (500, 0.5)]
    query_sc = [(98, 0.88), (152, 0.72), (45, 0.65)]

    sc_text_1 = encode_single_sc_set_to_text(dummy_sc_1, mock_sc_encoding_config_main)
    sc_text_2 = encode_single_sc_set_to_text(dummy_sc_2, mock_sc_encoding_config_main)
    query_sc_text_main = encode_single_sc_set_to_text(query_sc, mock_sc_encoding_config_main) # 避免与全局变量冲突

    mock_neighbors_sc = [
        (sc_text_1, "F-22"), # 支撑样本1
        (sc_text_2, "T-72")  # 支撑样本2
    ]

    print("--- 测试 0-shot SC Prompt ---")
    prompt_0shot_sc = constructor_sc.construct_prompt_with_sc(query_sc_text_main)
    # print(prompt_0shot_sc) # 打印查看
    with open("test_prompt_0shot_sc.txt", "w", encoding="utf-8") as f:
        f.write(prompt_0shot_sc)
    print("0-shot SC prompt 已保存到 test_prompt_0shot_sc.txt")

    print("\n--- 测试 K-shot SC Prompt ---")
    prompt_kshot_sc = constructor_sc.construct_prompt_with_sc(query_sc_text_main, mock_neighbors_sc)
    # print(prompt_kshot_sc) # 打印查看
    with open("test_prompt_kshot_sc.txt", "w", encoding="utf-8") as f:
        f.write(prompt_kshot_sc)
    print("K-shot SC prompt 已保存到 test_prompt_kshot_sc.txt")