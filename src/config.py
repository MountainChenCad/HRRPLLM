# src/config.py
import os

# --- Dataset Configuration (Defaults) ---
DEFAULT_DATASET_KEY = "measured"
AVAILABLE_DATASETS = {
    # "simulated": {
    #     "path": "datasets/simulated_hrrp",
    #     "data_var": "CoHH",
    #     "original_len": 1000,
    #     "max_samples_to_load": None
    # },
    "measured": {
        "path": "datasets/measured_hrrp",
        "data_var": "hrrp",
        "original_len": 500,
        "max_samples_to_load": None
    },
}
TARGET_HRRP_LENGTH = 1000 
PREPROCESS_MAT_TO_NPY = True 
PROCESSED_DATA_DIR = "data_processed"
TEST_SPLIT_SIZE = 0.3 
RANDOM_STATE = 42

# --- Scattering Center Extraction (Defaults) ---
SCATTERING_CENTER_EXTRACTION = {
    "enabled": True, 
    "method": "peak_detection", 
    "peak_prominence": 0.15,     
    "min_distance": 5,           
    "max_centers_to_keep": 10,  
    "normalize_hrrp_before_extraction": True, 
    "normalization_type_for_hrrp": "max" 
}

# --- Scattering Center Text Encoding (Defaults) ---
SCATTERING_CENTER_ENCODING = {
    "format": "list_of_dicts", 
    "precision_pos": 0,        
    "precision_amp": 3,        
    "center_separator": "; ",  
    "pos_amp_separator": ":",  
    "TARGET_HRRP_LENGTH_INFO": TARGET_HRRP_LENGTH 
}

# --- FSL Task Setup (Defaults - can be overridden by CLI) ---
DEFAULT_FSL_TASK_SETUP = {
    "enabled": True,
    "n_way": 5,                 
    "k_shot_support": 1,        
    "q_shot_query": 1,          
    "num_fsl_tasks": 20,       
    "sc_feature_type_for_prototype": "pos_amp_flat" 
}

# --- LLM API & Prompt Configuration (Defaults - can be overridden by CLI) ---
DEFAULT_LLM_CALLER_PARAMS = {
    "temperature": 0.1, # Base temperature, can be overridden for consistency
    "top_p": 1.0,
    "max_tokens_completion": 250, 
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "api_retry_delay": 10,       
    "max_retries": 5           
}
DEFAULT_NUM_CONSISTENCY_PATHS = 3 # Default to 1 (no ensemble). Override via CLI.
DEFAULT_CONSISTENCY_TEMPERATURE = 1.0 # Temperature to use if num_consistency_paths > 1

DEFAULT_API_KEY = "YOUR_DEFAULT_API_KEY_HERE"
DEFAULT_BASE_URL = "YOUR_DEFAULT_BASE_URL_HERE"
DEFAULT_API_PROVIDER = "openai" 

DEFAULT_LIMIT_TEST_SAMPLES = None 

RESULTS_BASE_DIR = "results" # For main experiments
RESULTS_ABLATION_DIR = "results_ablation" # For ablation studies
GENERATED_TABLES_DIR = "paper_tables" # For LaTeX tables

RUN_BASELINE_SVM = True 
BASELINE_SVM_PARAMS = {
    "C": 1.0, "kernel": "rbf", "feature_type": "scattering_centers",
    "sc_feature_type_for_svm": "pos_amp_flat" 
}
# Add Random Forest default params if needed
BASELINE_RF_PARAMS = {
    "n_estimators": 100, "feature_type": "scattering_centers",
    "sc_feature_type_for_rf": "pos_amp_flat"
}
RUN_BASELINE_RF = True