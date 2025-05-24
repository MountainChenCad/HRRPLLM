import os

# --- 数据集配置 ---
AVAILABLE_DATASETS = {
    "simulated": {
        "path": "datasets/simulated_hrrp",
        "data_var": "CoHH",
        "original_len": 1000,
        "max_samples_to_load": 300 # 保持较小以测试
    },
    # "measured": { # 如果有实测数据，取消注释并配置
    #     "path": "datasets/measured_hrrp",
    #     "data_var": "data",
    #     "original_len": 500,
    #     "max_samples_to_load": 100
    # }
}
TARGET_HRRP_LENGTH = 1000 
PREPROCESS_MAT_TO_NPY = True # 首次运行或更改max_samples_to_load时设为True
PROCESSED_DATA_DIR = "data_processed"
TEST_SPLIT_SIZE = 0.3 # 元训练集 vs 元测试集
RANDOM_STATE = 42

# --- 散射中心提取配置 ---
SCATTERING_CENTER_EXTRACTION = {
    "enabled": True, 
    "method": "peak_detection", 
    "peak_prominence": 0.15,     
    "peak_min_distance": 5,     
    "max_centers_to_keep": 10,  
    "normalize_hrrp_before_extraction": True, 
    "normalization_type_for_hrrp": "max"
}

# --- 散射中心文本编码配置 ---
SCATTERING_CENTER_ENCODING = {
    "format": "list_of_dicts", 
    "precision_pos": 0,        
    "precision_amp": 3,        
    "center_separator": "; ",  
    "pos_amp_separator": ":",  
}

# --- FSL Episode/Task 和 Prompt 示例选择配置 ---
FSL_TASK_SETUP = {
    "enabled": True,  # <--- 确保这个键存在
    # N-Way: 将使用数据集中所有唯一类别作为 "N"
    "k_shots_for_prompt_from_task_support": 0, # 为每个类别从元训练集中随机选择K个样本作为Prompt示例
                                       # 如果设为0, 则执行Zero-Shot (仅上下文和查询)
    "similarity_metric": "euclidean_on_sc",    # "dtw_on_hrrp", "euclidean_on_hrrp", "euclidean_on_sc"
    "sc_feature_type_for_similarity": "pos_amp_flat" # 如果metric是 'euclidean_on_sc'
}

# --- LLM API 与 Prompt 配置 ---
# DEEPSEEK API 配置
OPENAI_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-59f809e831cf4859935d949b41985ae8") # 使用您的DeepSeek Key
OPENAI_PROXY_BASE_URL = "https://api.deepseek.com" # DeepSeek API Base URL

LLM_CALLER_PARAMS = {
    # "model_name": "gpt-4o-mini", # 旧的OpenAI模型
    "model_name": "deepseek-chat", # 或者 deepseek-coder, 根据DeepSeek文档选择合适的模型
    "temperature": 1.0, # DeepSeek的temperature可能需要调整，1.0对于分类来说太高了
    "top_p": 1.0,
    "max_tokens_completion": 200, 
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "api_retry_delay": 5,       
    "max_retries": 3            
}
LIMIT_TEST_SAMPLES = 20 # 大幅减少测试样本以便快速调试DeepSeek API调用

# --- 实验结果保存 ---
RESULTS_BASE_DIR = "results"

# --- 基线模型配置 ---
RUN_BASELINE_SVM = True # 是否运行基线
BASELINE_SVM_PARAMS = {
    "C": 1.0, "kernel": "rbf", "feature_type": "scattering_centers" 
}