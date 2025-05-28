#!/bin/bash

# This script automates running multiple FSL experiments with different LLMs and scenarios.

# --- Global Configuration ---
# These can be overridden for specific model groups if needed
DEFAULT_N_WAY=5
DEFAULT_NUM_FSL_TASKS=20 # Number of FSL tasks per experiment run
DEFAULT_DATASET_KEY="simulated" # Ensure this dataset is configured in config.py
DEFAULT_LIMIT_TEST_SAMPLES="None" # Or a number like 100. "None" means no limit beyond dataset's max_samples_to_load.
ROOT_RESULTS_DIR="results" # Should match config.py

# FSL Scenarios to test (K_shot_support, Q_shot_query)
# Format: "K_VALUE,Q_VALUE"
FSL_SCENARIOS=(
    "1,1"  # Scenario 1: 5-Way 1-Shot 1-Query
#    "5,1"  # Scenario 2: 5-Way 5-Shot 1-Query (as per user's table example, K=3 was a typo in table)
)

# --- API Key Configuration ---
# It's best to set these as environment variables or use a secure vault.
# For this script, we'll define them here. Replace with your actual keys.
# Key for DeepSeek Platform / ZhipuAI GLM (on your supercomputer)
DEEPSEEK_PLATFORM_API_KEY="sk-c46c479bff48429dbdf15094c81f086e"
DEEPSEEK_PLATFORM_BASE_URL="https://api.deepseek.com"

CHAOSUAN_PLATFORM_API_KEY="sk-g0UhooVGGk8r9l3pDZjHbQ"
CHAOSUAN_BASE_URL="https://llmapi.blsc.cn/v1"

# Key for OpenAI, Anthropic, Google (via your proxy: api.openai-proxy.live)
PROXY_API_KEY="sk-fim815RQQnKhcbB7C5LbKjdkJsc7YyMjn6L6d6FReGJj5kUZ"

# --- Base URL / Endpoint Configuration ---
OPENAI_PROXY_BASE_URL="https://api.openai-proxy.live/v1"
ANTHROPIC_PROXY_BASE_URL="https://api.openai-proxy.live/anthropic"
GOOGLE_PROXY_API_ENDPOINT="https://api.openai-proxy.live/google"


# --- Model Definitions ---
# Structure: "MODEL_NAME|API_PROVIDER|API_KEY_VAR_NAME|BASE_URL_VAR_NAME_OR_ENDPOINT_VAR_NAME|GROUP_TAG"
# API_KEY_VAR_NAME and BASE_URL_VAR_NAME should refer to the variables defined above.
# For Google, the BASE_URL_VAR_NAME will be treated as the API_ENDPOINT.

MODELS_TO_TEST=(
    # A. 基准与快速迭代组 (免费/轻量) - Assuming these are on DeepSeek Platform or ZhipuAI GLM
    "GLM-4-Flash-P002|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupA_GLM-4-Flash-P002"
    "DeepSeek-R1-N011-Distill-Qwen-7B|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupA_DeepSeek-R1-N011-Distill-Qwen-7B"
    "GLM-Z1-Flash-P002|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupA_GLM-Z1-Flash-P002"

    # OpenAI Platform Models (via Proxy)
    "gpt-4o-mini|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL|GroupA_OpenAI_Mini"
    "gpt-3.5-turbo|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL|GroupA_OpenAI_3.5T"

    # B. 中坚力量与推理优化组
    "gpt-4o|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL|GroupB_OpenAI_4o"
    "GLM-4-Air-P002|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupB_GLM-4-Air-P002"
    "GLM-4-Long-P002|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupB_GLM-4-Long-P002"
    "GLM-Z1-Air-P002|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupB_GLM-Z1-Air-P002"
    "QwQ-N011-32B|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupB_QwQ-N011-32B"
    "Qwen3-32B|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupB_Qwen3-32B"
    "deepseek-coder|deepseek_platform|DEEPSEEK_PLATFORM_API_KEY|DEEPSEEK_PLATFORM_BASE_URL|GroupB_DeepSeek_Coder" # Example

    # Anthropic Models (via Proxy)
    "claude-3-haiku-20240307|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL|GroupB_Claude_Haiku"
    "claude-3-sonnet-20240229|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL|GroupB_Claude_Sonnet"
    "claude-3-opus-20240229|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL|GroupC_Claude_Opus"
    # Google Models (via Proxy)
    "gemini-1.5-flash-latest|google|PROXY_API_KEY|GOOGLE_PROXY_API_ENDPOINT|GroupB_Gemini_Flash"

#     C. 高性能与旗舰SOTA组
    "gpt-4-turbo|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL|GroupC_OpenAI_4T"
    "gpt-4|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL|GroupC_OpenAI_gpt4"
    "gemini-1.5-pro-latest|google|PROXY_API_KEY|GOOGLE_PROXY_API_ENDPOINT|GroupC_Gemini_Pro"
    "gpt-4.1|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL|GroupC_gpt-4.1"
    "o3-2025-04-16|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL|GroupC_o3-2025-04-16"
    "o1|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL|GroupC_o1"
    "claude-opus-4-20250514|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL|GroupC_claude-opus-4-20250514"
    "claude-sonnet-4-20250514|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL|GroupC_claude-sonnet-4-20250514"
    "claude-3-7-sonnet-20250219|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL|GroupC_claude-3-7-sonnet-20250219"
    "claude-3-5-sonnet-20241022|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL|GroupC_claude-3-5-sonnet-20241022"
    "gemini-2.5-flash-preview-04-17|google|PROXY_API_KEY|GOOGLE_PROXY_API_ENDPOINT|GroupC_gemini-2.5-flash-preview-04-17" # Your current model
    "GLM-4-Plus-P002|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupC_GLM-4-Plus-P002"

    # Add more models from your list here following the format:
    # "MODEL_IDENTIFIER_FOR_API|api_provider_name|API_KEY_VARIABLE|BASE_URL_or_ENDPOINT_VARIABLE|YourGroupTag"
    "Qwen3-235B-A22B|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupC_Qwen3-235B-A22B"
    "DeepSeek-V3-P001|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupC_DeepSeek-V3-P001"
    "GLM-4-Plus-P002|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupC_GLM-4-Plus-P002"
    "deepseek-chat|deepseek_platform|DEEPSEEK_PLATFORM_API_KEY|DEEPSEEK_PLATFORM_BASE_URL|GroupC_DeepSeek_V3"
    "deepseek-reasoner|deepseek_platform|DEEPSEEK_PLATFORM_API_KEY|DEEPSEEK_PLATFORM_BASE_URL|GroupC_DeepSeek_R1"

    "DeepSeek-R1-N011-Distill-Llama-8B|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupD_DeepSeek-R1-N011-Distill-Llama-8B"
    "DeepSeek-R1-N011-Distill-Qwen-14B|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupD_DeepSeek-R1-N011-Distill-Qwen-14B"
    "DeepSeek-R1-N011-Distill-Qwen-32B|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupD_DeepSeek-R1-N011-Distill-Qwen-32B"
)

# --- Script Execution ---
PYTHON_EXECUTABLE="python" # Or "python3"
MAIN_SCRIPT_PATH="src/main_experiment.py"

# Optional: Skip data preprocessing if you are sure it's done
# SKIP_PREPROCESSING_ARG="--skip_data_preprocessing"
SKIP_PREPROCESSING_ARG=""
SKIP_SVM_ARG="--skip_svm_baseline" # Usually good to skip SVM during LLM grid search

# Create root results directory if it doesn't exist
mkdir -p "$ROOT_RESULTS_DIR"

# Loop through each FSL scenario
for scenario in "${FSL_SCENARIOS[@]}"; do
    IFS=',' read -r k_shot q_shot <<< "$scenario"

    echo ""
    echo "**********************************************************************"
    echo "Starting FSL Scenario: N=${DEFAULT_N_WAY}, K-support=${k_shot}, Q-query=${q_shot}"
    echo "**********************************************************************"

    # Loop through each model configuration
    for model_config in "${MODELS_TO_TEST[@]}"; do
        IFS='|' read -r model_name api_provider api_key_var_name url_or_endpoint_var_name group_tag <<< "$model_config"

        # Dynamically get the API key and URL/Endpoint values
        current_api_key="${!api_key_var_name}"
        current_url_or_endpoint="${!url_or_endpoint_var_name}"

        experiment_tag_final="${group_tag}_N${DEFAULT_N_WAY}_K${k_shot}_Q${q_shot}"

        # Prepare command line arguments
        CMD_ARGS=(
            "--model_name" "$model_name"
            "--api_provider" "$api_provider"
            "--api_key" "$current_api_key"
            "--dataset_key" "$DEFAULT_DATASET_KEY"
            "--n_way" "$DEFAULT_N_WAY"
            "--k_shot_support" "$k_shot"
            "--q_shot_query" "$q_shot"
            "--num_fsl_tasks" "$DEFAULT_NUM_FSL_TASKS"
            "--experiment_tag" "$experiment_tag_final"
        )

        if [ "$api_provider" == "google" ]; then
            CMD_ARGS+=("--google_api_endpoint" "$current_url_or_endpoint")
        elif [ -n "$current_url_or_endpoint" ]; then # Only add base_url if it's not empty
             CMD_ARGS+=("--base_url" "$current_url_or_endpoint")
        fi

        if [ "$DEFAULT_LIMIT_TEST_SAMPLES" != "None" ]; then
            CMD_ARGS+=("--limit_test_samples" "$DEFAULT_LIMIT_TEST_SAMPLES")
        fi

        if [ -n "$SKIP_PREPROCESSING_ARG" ]; then
             CMD_ARGS+=($SKIP_PREPROCESSING_ARG)
        fi
        if [ -n "$SKIP_SVM_ARG" ]; then
             CMD_ARGS+=($SKIP_SVM_ARG)
        fi

        echo ""
        echo "----------------------------------------------------------------------"
        echo "Executing: Model=$model_name, Provider=$api_provider, Scenario: K=$k_shot, Q=$q_shot"
        echo "Experiment Tag: $experiment_tag_final"
        echo "Command: $PYTHON_EXECUTABLE $MAIN_SCRIPT_PATH ${CMD_ARGS[*]}"
        echo "----------------------------------------------------------------------"

        # Execute the Python script
        "$PYTHON_EXECUTABLE" "$MAIN_SCRIPT_PATH" "${CMD_ARGS[@]}"

        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "ERROR: Experiment for $model_name (K=$k_shot, Q=$q_shot) failed with exit code $EXIT_CODE."
            echo "Please check the logs."
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            # Decide if you want to continue or stop the script on error
            # read -p "Press Enter to continue or Ctrl+C to abort..."
        else
            echo "Successfully completed: $model_name (K=$k_shot, Q=$q_shot)"
        fi
        echo "----------------------------------------------------------------------"
        # Optional: Add a small delay to avoid overwhelming APIs or file systems
        sleep 5
    done # End model loop
done # End scenario loop

echo ""
echo "**********************************************************************"
echo "All experiments defined in the script have been attempted."
echo "**********************************************************************"