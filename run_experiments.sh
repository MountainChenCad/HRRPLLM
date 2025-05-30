#!/bin/bash

echo "Starting Main Comparative Experiments Script"

# --- Global Configuration ---
DEFAULT_N_WAY=3
DEFAULT_Q_QUERY=1 # Q=1 as per your FSL setup
DEFAULT_NUM_FSL_TASKS=30
DEFAULT_DATASET_KEY="measured"
DEFAULT_LIMIT_TEST_SAMPLES="None"
RESULTS_DIR_MAIN_EXP="results/main_experiments" # Specific directory for these main results
LLM_RESULTS_CSV="${RESULTS_DIR_MAIN_EXP}/llm_comparative_results.csv"
BASELINE_RESULTS_CSV="${RESULTS_DIR_MAIN_EXP}/baseline_results.csv"


# FSL Scenarios for LLMs (K_shot_support)
# We will iterate K values: 1 and 5
K_SHOT_VALUES=(1)

# --- API Key Configuration (Ensure these are correct) ---
DEEPSEEK_PLATFORM_API_KEY="sk-c46c479bff48429dbdf15094c81f086e"
DEEPSEEK_PLATFORM_BASE_URL="https://api.deepseek.com" # Ensure /v1 is added if needed by specific models/proxy

CHAOSUAN_PLATFORM_API_KEY="sk-g0UhooVGGk8r9l3pDZjHbQ"
CHAOSUAN_BASE_URL="https://llmapi.blsc.cn/v1"

PROXY_API_KEY="sk-fim815RQQnKhcbB7C5LbKjdkJsc7YyMjn6L6d6FReGJj5kUZ"
OPENAI_PROXY_BASE_URL="https://api.openai-proxy.live/v1"
ANTHROPIC_PROXY_BASE_URL="https://api.openai-proxy.live/anthropic"
GOOGLE_PROXY_API_ENDPOINT="https://api.openai-proxy.live/google"


# --- Model Definitions (from your provided list) ---
MODELS_TO_TEST=(
#    # A. Benchmark & Fast Iteration
#    "GLM-4-Flash-P002|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupA"
    "DeepSeek-R1-N011-Distill-Qwen-7B|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupA"
    "GLM-Z1-Flash-P002|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupA"
    "gpt-4o-mini|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL|GroupA"
    "gpt-3.5-turbo|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL|GroupA"

    # B. Mid-Tier & Inference Optimized
    "gpt-4o|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL|GroupB"
    "GLM-4-Air-P002|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupB"
    "GLM-4-Long-P002|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupB" # Note: Long context not directly tested by K-shot variation here
    "GLM-Z1-Air-P002|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupB"
    "QwQ-N011-32B|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupB"
    "Qwen3-32B|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupB"
    "deepseek-coder|deepseek_platform|DEEPSEEK_PLATFORM_API_KEY|DEEPSEEK_PLATFORM_BASE_URL|GroupB"
    "claude-3-haiku-20240307|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL|GroupB"
    "claude-3-sonnet-20240229|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL|GroupB"
    "gemini-1.5-flash-latest|google|PROXY_API_KEY|GOOGLE_PROXY_API_ENDPOINT|GroupB"

    # C. High-Performance & SOTA
    "gpt-4-turbo|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL|GroupC"
    "gpt-4|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL|GroupC"
    "gemini-1.5-pro-latest|google|PROXY_API_KEY|GOOGLE_PROXY_API_ENDPOINT|GroupC"
    "claude-3-opus-20240229|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL|GroupC"
    "gpt-4.1|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL|GroupC_Newer" # Assuming gpt-4.1 is a distinct model entry
    "o3-2025-04-16|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL|GroupC_Newer"
    "o1|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL|GroupC_Newer" # Ensure 'o1' is the correct API identifier
    "claude-opus-4-20250514|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL|GroupC_Newer"
    "claude-sonnet-4-20250514|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL|GroupC_Newer"
    "claude-3-7-sonnet-20250219|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL|GroupC_Newer"
    "claude-3-5-sonnet-20241022|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL|GroupC_Newer"
    "gemini-2.5-flash-preview-04-17|google|PROXY_API_KEY|GOOGLE_PROXY_API_ENDPOINT|GroupC_Newer"
    "GLM-4-Plus-P002|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupC_Zhipu"
    "Qwen3-235B-A22B|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupC_Zhipu"
    "DeepSeek-V3-P001|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupC_Zhipu" # This might be DeepSeek model on Zhipu platform
    "deepseek-chat|deepseek_platform|DEEPSEEK_PLATFORM_API_KEY|DEEPSEEK_PLATFORM_BASE_URL|GroupC_DeepSeek"
    "deepseek-reasoner|deepseek_platform|DEEPSEEK_PLATFORM_API_KEY|DEEPSEEK_PLATFORM_BASE_URL|GroupC_DeepSeek"

    # D. Specific Features/Distilled
    "DeepSeek-R1-N011-Distill-Llama-8B|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupD"
    "DeepSeek-R1-N011-Distill-Qwen-14B|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupD"
    "DeepSeek-R1-N011-Distill-Qwen-32B|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL|GroupD"
)

# --- Script Execution ---
PYTHON_EXECUTABLE="python3" # Ensure this is python3 if needed
MAIN_SCRIPT_PATH="src/main_experiment.py"
BASELINE_SCRIPT_PATH="src/baseline_evaluator.py"

# Create results directory
mkdir -p "$RESULTS_DIR_MAIN_EXP"
echo "Main experiment results will be saved in: $RESULTS_DIR_MAIN_EXP"
echo "LLM results CSV: $LLM_RESULTS_CSV"
echo "Baseline results CSV: $BASELINE_RESULTS_CSV"

# --- Run LLM Experiments ---
echo ""
echo "**********************************************************************"
echo "Starting LLM Comparative Experiments"
echo "**********************************************************************"

for k_shot in "${K_SHOT_VALUES[@]}"; do
    echo ""
    echo "===== Running for K-Shot (Support) = $k_shot ====="

    for model_config in "${MODELS_TO_TEST[@]}"; do
        IFS='|' read -r model_name api_provider api_key_var_name url_or_endpoint_var_name group_tag <<< "$model_config"

        current_api_key="${!api_key_var_name}"
        current_url_or_endpoint="${!url_or_endpoint_var_name}"

        # Experiment tag combines group and K-shot for clarity in results folder structure
        experiment_tag_final="${group_tag}_K${k_shot}"

        CMD_ARGS_LLM=(
            "--model_name" "$model_name"
            "--api_provider" "$api_provider"
            "--api_key" "$current_api_key"
            "--dataset_key" "$DEFAULT_DATASET_KEY"
            "--n_way" "$DEFAULT_N_WAY"
            "--k_shot_support" "$k_shot"
            "--q_shot_query" "$DEFAULT_Q_QUERY" # Q=1
            "--num_fsl_tasks" "$DEFAULT_NUM_FSL_TASKS"
            "--experiment_tag" "$experiment_tag_final"
            "--output_csv_llm" "$LLM_RESULTS_CSV"
            "--results_base_dir" "$RESULTS_DIR_MAIN_EXP" # Base for prompts/responses folders
            # Using default consistency (1 path) and SC settings for main comparison
        )

        if [ "$api_provider" == "google" ]; then
            CMD_ARGS_LLM+=("--google_api_endpoint" "$current_url_or_endpoint")
        elif [ -n "$current_url_or_endpoint" ]; then
            CMD_ARGS_LLM+=("--base_url" "$current_url_or_endpoint")
        fi

        if [ "$DEFAULT_LIMIT_TEST_SAMPLES" != "None" ]; then
            CMD_ARGS_LLM+=("--limit_test_samples" "$DEFAULT_LIMIT_TEST_SAMPLES")
        fi

        # For the first run of a model ONLY, force data preprocessing if needed
        # This is a simple way; more robust would be to check if SC files for default params exist
        if [[ "$k_shot" == "${K_SHOT_VALUES[0]}" ]]; then # Only for the first K-shot value
             CMD_ARGS_LLM+=("--force_data_preprocessing") # Ensures data is ready with default SC params
        fi


        echo ""
        echo "--- LLM RUN: Model=$model_name, K=$k_shot, Tag=$experiment_tag_final ---"
        echo "CMD: $PYTHON_EXECUTABLE $MAIN_SCRIPT_PATH ${CMD_ARGS_LLM[*]}"

        "$PYTHON_EXECUTABLE" "$MAIN_SCRIPT_PATH" "${CMD_ARGS_LLM[@]}"

        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "ERROR: LLM Experiment for $model_name (K=$k_shot) failed. Exit code $EXIT_CODE."
        else
            echo "Completed LLM: $model_name (K=$k_shot)"
        fi
        sleep 5
    done # End model loop
done # End K-shot loop for LLMs

# --- Run Baseline Experiments ---
# Baselines are run once as they don't depend on K-shot in the same way.
# They use the full meta-train/meta-test with default SC parameters.
echo ""
echo "**********************************************************************"
echo "Starting Baseline Model Experiments"
echo "**********************************************************************"

BASELINE_MODELS_FEATURES=(
    "SVM|raw_hrrp"
    "SVM|scattering_centers"
    "RF|scattering_centers" # Random Forest with SC features
)

# Ensure data is preprocessed with default SC settings if not already done
# (The first LLM run with K_SHOT_VALUES[0] and --force_data_preprocessing should have handled this)
# If you want to be absolutely sure:
# "$PYTHON_EXECUTABLE" "$MAIN_SCRIPT_PATH" --model_name "dummy" --api_provider "openai" --api_key "none" --force_data_preprocessing --num_fsl_tasks 0 --dataset_key "$DEFAULT_DATASET_KEY" > /dev/null 2>&1
# This is a bit hacky; better to rely on the first LLM run.

for baseline_config in "${BASELINE_MODELS_FEATURES[@]}"; do
    IFS='|' read -r model_type feature_type <<< "$baseline_config"

    CMD_ARGS_BASELINE=(
        "--dataset_key" "$DEFAULT_DATASET_KEY"
        "--model_type" "$model_type"
        "--feature_type" "$feature_type"
        "--output_csv" "$BASELINE_RESULTS_CSV"
    )

    echo ""
    echo "--- BASELINE RUN: Model=$model_type, Features=$feature_type ---"
    echo "CMD: $PYTHON_EXECUTABLE $BASELINE_SCRIPT_PATH ${CMD_ARGS_BASELINE[*]}"

    "$PYTHON_EXECUTABLE" "$BASELINE_SCRIPT_PATH" "${CMD_ARGS_BASELINE[@]}"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Baseline Experiment for $model_type ($feature_type) failed. Exit code $EXIT_CODE."
    else
        echo "Completed Baseline: $model_type ($feature_type)"
    fi
    sleep 2
done


echo ""
echo "**********************************************************************"
echo "All Main Comparative Experiments Attempted."
echo "LLM results are in: $LLM_RESULTS_CSV"
echo "Baseline results are in: $BASELINE_RESULTS_CSV"
echo "Detailed run outputs (prompts/responses) are in subfolders of: $RESULTS_DIR_MAIN_EXP"
echo "**********************************************************************"