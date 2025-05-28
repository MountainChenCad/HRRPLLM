import os

# --- Dataset Configuration (Defaults) ---
# These can be overridden by command-line arguments if main_experiment.py is adapted
DEFAULT_DATASET_KEY = "simulated" # Default dataset to use if not specified via CLI
AVAILABLE_DATASETS = {
    "simulated": {
        "path": "datasets/simulated_hrrp",
        "data_var": "CoHH",
        "original_len": 1000,
        "max_samples_to_load": None # Load all by default, can be overridden by dataset config
    },
    # Add other dataset configurations here if needed
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
    "num_fsl_tasks": 20, # Number of FSL tasks to generate per run/scenario
    "sc_feature_type_for_prototype": "pos_amp_flat"
}

# --- LLM API & Prompt Configuration (Defaults - can be overridden by CLI) ---
DEFAULT_LLM_CALLER_PARAMS = {
    # "model_name" will come from CLI
    "temperature": 0.1, # Good for classification
    "top_p": 1.0,
    "max_tokens_completion": 250, # Increased slightly for potential reasoning
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "api_retry_delay": 10, # Increased delay
    "max_retries": 5 # Increased retries
}

# API Keys and Base URLs will primarily come from CLI or environment variables
# It's good practice not to hardcode API keys in config files.
# The bash script will pass these.
# Example default (placeholder, script should override)
DEFAULT_API_KEY = "YOUR_DEFAULT_API_KEY_HERE"
DEFAULT_BASE_URL = "YOUR_DEFAULT_BASE_URL_HERE"
DEFAULT_API_PROVIDER = "openai" # e.g., "openai", "anthropic", "google", "deepseek_platform"

# LIMIT_TEST_SAMPLES: This will be applied to the loaded meta-test set
# before FSL tasks are constructed from it.
# Can be overridden by CLI.
DEFAULT_LIMIT_TEST_SAMPLES = None # No limit by default

# --- Experiment Results Saving ---
RESULTS_BASE_DIR = "results"

# --- Baseline Model Configuration ---
RUN_BASELINE_SVM = True # Can be a global flag, or controlled per script run
BASELINE_SVM_PARAMS = {
    "C": 1.0, "kernel": "rbf", "feature_type": "scattering_centers",
    "sc_feature_type_for_svm": "pos_amp_flat" 
}