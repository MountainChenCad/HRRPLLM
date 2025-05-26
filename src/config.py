import os

# --- 数据集配置 ---
AVAILABLE_DATASETS = {
    "simulated": {
        "path": "datasets/simulated_hrrp",
        "data_var": "CoHH",
        "original_len": 1000,
        "max_samples_to_load": 300 
    },
}
TARGET_HRRP_LENGTH = 1000 
PREPROCESS_MAT_TO_NPY = True 
PROCESSED_DATA_DIR = "data_processed"
TEST_SPLIT_SIZE = 0.3 
RANDOM_STATE = 42

# --- 散射中心提取配置 ---
SCATTERING_CENTER_EXTRACTION = {
    "enabled": True, 
    "method": "peak_detection", 
    "peak_prominence": 0.15,     
    "min_distance": 5,           
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
    "TARGET_HRRP_LENGTH_INFO": TARGET_HRRP_LENGTH 
}

# --- FSL Episode/Task 和 Prompt 示例选择配置 ---
FSL_TASK_SETUP = {
    "enabled": True,
    "n_way": 12,                 
    "k_shot_support": 1, # MODIFIED: Set K=1 for example, can be >1 for prototype calculation
    "q_shot_query": 1,          
    "num_fsl_tasks": 50,        
    "sc_feature_type_for_prototype": "pos_amp_flat" # MODIFIED: Added for prototype calculation
}

# --- LLM API 与 Prompt 配置 ---
# OPENAI_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-59f809e831cf4859935d949b41985ae8")
# OPENAI_PROXY_BASE_URL = "https://api.deepseek.com"

OPENAI_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-g0UhooVGGk8r9l3pDZjHbQ")
OPENAI_PROXY_BASE_URL = "https://llmapi.blsc.cn/v1/"

LLM_CALLER_PARAMS = {
    "model_name": "QwQ-N011-32B",
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens_completion": 200, 
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "api_retry_delay": 5,       
    "max_retries": 3            
}
LIMIT_TEST_SAMPLES = 2000 

# --- 实验结果保存 ---
RESULTS_BASE_DIR = "results"

# --- 基线模型配置 ---
RUN_BASELINE_SVM = True 
BASELINE_SVM_PARAMS = {
    "C": 1.0, "kernel": "rbf", "feature_type": "scattering_centers",
    "sc_feature_type_for_svm": "pos_amp_flat" 
}