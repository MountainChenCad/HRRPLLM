#!/bin/bash

echo "Starting Ablation Experiments Script"

# --- Base Configuration for Ablation ---
DEFAULT_N_WAY=3
DEFAULT_Q_QUERY=1
DEFAULT_NUM_FSL_TASKS=15 # Fewer tasks for faster ablation
DEFAULT_DATASET_KEY="measured"
DEFAULT_LIMIT_TEST_SAMPLES="None"
RESULTS_DIR_ABLATION="results_ablation" # Specific directory for ablation results
ABLATION_RESULTS_CSV="${RESULTS_DIR_ABLATION}/llm_ablation_results.csv"

PYTHON_EXECUTABLE="python3"
MAIN_SCRIPT_PATH="src/main_experiment.py"
# Baselines are not typically re-run for LLM ablations unless the ablation affects data processing shared with baselines.

# --- API Keys (same as run_experiments.sh) ---
DEEPSEEK_PLATFORM_API_KEY=""
DEEPSEEK_PLATFORM_BASE_URL="https://api.deepseek.com"

CHAOSUAN_PLATFORM_API_KEY=""
CHAOSUAN_BASE_URL="https://llmapi.blsc.cn/v1"

PROXY_API_KEY=""
OPENAI_PROXY_BASE_URL="https://api.openai-proxy.live/v1"
ANTHROPIC_PROXY_BASE_URL="https://api.openai-proxy.live/anthropic"
GOOGLE_PROXY_API_ENDPOINT="https://api.openai-proxy.live/google"


# --- Representative Models for Ablation ---
ABLATION_MODELS=(

    # Add one more if desired, e.g., a Claude or Gemini model
    "claude-opus-4-0|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL"
#    "claude-sonnet-4-20250514|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL"
#    "claude-3-7-sonnet-20250219|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL"
#    "claude-3-5-sonnet-20241022|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL"
    "gpt-4.1-2025-04-14|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL"
    "o4-mini-2025-04-16|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL"

#    "gemini-2.5-pro-preview-03-25|google|PROXY_API_KEY|GOOGLE_PROXY_API_ENDPOINT"

#    "claude-3-sonnet-20240229|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL"
#    "gpt-4o-mini|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL"
#    "deepseek-chat|deepseek_platform|DEEPSEEK_PLATFORM_API_KEY|DEEPSEEK_PLATFORM_BASE_URL"
#    "Qwen3-235B-A22B|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL"
#    "GLM-4-Plus-P002|zhipuai_glm|CHAOSUAN_PLATFORM_API_KEY|CHAOSUAN_BASE_URL"
#    "gemini-1.5-pro-latest|google|PROXY_API_KEY|GOOGLE_PROXY_API_ENDPOINT"
)

mkdir -p "$RESULTS_DIR_ABLATION"
echo "Ablation experiment results will be saved in: $RESULTS_DIR_ABLATION"
echo "Ablation LLM results CSV: $ABLATION_RESULTS_CSV"


# --- Function to run a single ablation experiment ---
run_ablation_experiment() {
    local model_name="$1"
    local api_provider="$2"
    local api_key_var="$3"
    local url_endpoint_var="$4"
    local k_shot="$5"
    local ablation_tag="$6" # Descriptive tag for the ablation condition
    local extra_cli_args_str="$7" # String of additional CLI args

    local current_api_key="${!api_key_var}"
    local current_url_or_endpoint="${!url_endpoint_var}"
    # Construct a unique experiment tag for folder naming and CSV logging
    local experiment_tag_final="Ablate_${ablation_tag}_K${k_shot}_${model_name//\//_}"
    experiment_tag_final=$(echo "$experiment_tag_final" | tr -d '[:space:]') # Remove spaces

    CMD_ARGS_LLM=(
        "--model_name" "$model_name"
        "--api_provider" "$api_provider"
        "--api_key" "$current_api_key"
        "--dataset_key" "$DEFAULT_DATASET_KEY"
        "--n_way" "$DEFAULT_N_WAY"
        "--k_shot_support" "$k_shot"
        "--q_shot_query" "$DEFAULT_Q_QUERY"
        "--num_fsl_tasks" "$DEFAULT_NUM_FSL_TASKS"
        "--experiment_tag" "$experiment_tag_final" # Used for subfolder naming
        "--output_csv_llm" "$ABLATION_RESULTS_CSV" # All ablations append to this CSV
        "--results_base_dir" "$RESULTS_DIR_ABLATION" # Base for prompts/responses folders
        # Default consistency for ablations (1 path), can be overridden by extra_cli_args_str
        "--num_consistency_paths" "3"
    )

    if [ "$api_provider" == "google" ]; then
        CMD_ARGS_LLM+=("--google_api_endpoint" "$current_url_or_endpoint")
    elif [ -n "$current_url_or_endpoint" ]; then
        CMD_ARGS_LLM+=("--base_url" "$current_url_or_endpoint")
    fi

    if [ "$DEFAULT_LIMIT_TEST_SAMPLES" != "None" ]; then
        CMD_ARGS_LLM+=("--limit_test_samples" "$DEFAULT_LIMIT_TEST_SAMPLES")
    fi

    # Add extra CLI args for the specific ablation
    if [ -n "$extra_cli_args_str" ]; then
        eval "CMD_ARGS_LLM+=($extra_cli_args_str)" # Use eval to handle spaces in args
    fi

    echo ""
    echo "--- ABLATION RUN: Model=$model_name, K=$k_shot, Ablation=$ablation_tag ---"
    echo "Extra CLI: $extra_cli_args_str"
    echo "Full CMD: $PYTHON_EXECUTABLE $MAIN_SCRIPT_PATH ${CMD_ARGS_LLM[*]}"

    "$PYTHON_EXECUTABLE" "$MAIN_SCRIPT_PATH" "${CMD_ARGS_LLM[@]}"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Ablation for $model_name (K=$k_shot, Ablation=$ablation_tag) failed. Exit code $EXIT_CODE."
    else
        echo "Completed Ablation: $model_name (K=$k_shot, Ablation=$ablation_tag)"
    fi
    sleep 5
}

# === Reference Point for Ablations (Full Prompt, K=1) ===
# This ensures data is preprocessed with default SC settings before SC ablations that modify them.
K_REF=1
for model_config in "${ABLATION_MODELS[@]}"; do
    IFS='|' read -r mn ap ak ue <<< "$model_config"
    run_ablation_experiment "$mn" "$ap" "$ak" "$ue" "$K_REF" "FullPromptK1" "--force_data_preprocessing" # Force preprocessing for default SC
done

# === 1. Prompt Component Ablation (using K=1) ===
K_PROMPT_ABLATION=1
PROMPT_ABLATIONS=(
    "NoSysInstr|--prompt_no_system_instruction"
    "NoBgKnow|--prompt_no_background_knowledge"
    "NoCandList|--prompt_no_candidate_list"
    "NoOutFmt|--prompt_no_output_format"
#    "NoSys_NoCand_NoFmt|--prompt_no_system_instruction --prompt_no_candidate_list --prompt_no_output_format"
)
for model_config in "${ABLATION_MODELS[@]}"; do
    IFS='|' read -r mn ap ak ue <<< "$model_config"
    for ablation_setting in "${PROMPT_ABLATIONS[@]}"; do
        IFS='|' read -r tag cli_arg <<< "$ablation_setting"
        run_ablation_experiment "$mn" "$ap" "$ak" "$ue" "$K_PROMPT_ABLATION" "Prompt_$tag" "$cli_arg"
    done
done

# === 2. Scattering Center Information Quality Ablation (using K=1, Full Prompt otherwise) ===
# These will force data preprocessing due to changes in SC parameters
K_SC_QUALITY_ABLATION=1
SC_QUALITY_ABLATIONS=(
    "SCMax3|--sc_max_centers_to_keep 3 --force_data_preprocessing"
    "SCMax15|--sc_max_centers_to_keep 15 --force_data_preprocessing"
    "SCAmpPrec1|--sc_encoding_precision_amp 1 --force_data_preprocessing"
    "SCAmpPrec5|--sc_encoding_precision_amp 5 --force_data_preprocessing"
)
for model_config in "${ABLATION_MODELS[@]}"; do
    IFS='|' read -r mn ap ak ue <<< "$model_config"
    for ablation_setting in "${SC_QUALITY_ABLATIONS[@]}"; do
        IFS='|' read -r tag cli_arg <<< "$ablation_setting"
        run_ablation_experiment "$mn" "$ap" "$ak" "$ue" "$K_SC_QUALITY_ABLATION" "SCQual_$tag" "$cli_arg"
    done
done
# IMPORTANT: After SC quality ablations, subsequent runs will use the *last modified* SC data.
# Run a baseline again with default SC settings if other ablations need the original SC files.
echo "Re-running baseline with default SC settings after SC quality ablations..."
for model_config in "${ABLATION_MODELS[@]}"; do
    IFS='|' read -r mn ap ak ue <<< "$model_config"
    # Force preprocessing to revert to default SC settings from config.py
    run_ablation_experiment "$mn" "$ap" "$ak" "$ue" "$K_REF" "FullPromptK1_PostSCQual" "--force_data_preprocessing"
done


# === 3. 0-Shot vs. Few-Shot (K=0 vs. K=1, K=5 - Full Prompt otherwise) ===
# K=1 is already run as "FullPromptK1_PostSCQual".
# We need K=0 and K=5 (or another K value if desired).
K_ZERO_SHOT=0
K_FIVE_SHOT=5

for model_config in "${ABLATION_MODELS[@]}"; do
    IFS='|' read -r mn ap ak ue <<< "$model_config"
    run_ablation_experiment "$mn" "$ap" "$ak" "$ue" "$K_ZERO_SHOT" "KShot0" ""
    run_ablation_experiment "$mn" "$ap" "$ak" "$ue" "$K_FIVE_SHOT" "KShot${K_FIVE_SHOT}" ""
done

## === 4. Consistency / Multi-path Ensemble Enhancement (using K=1, Full Prompt) ===
#K_CONSISTENCY=1
#CONSISTENCY_PATHS=(1 5) # Test with 3 and 5 paths
#CONSISTENCY_TEMP=1.0 # Use a higher temperature for diversity
#for model_config in "${ABLATION_MODELS[@]}"; do
#    IFS='|' read -r mn ap ak ue <<< "$model_config"
#    for num_paths in "${CONSISTENCY_PATHS[@]}"; do
#        cli_arg_consistency="--num_consistency_paths $num_paths --consistency_temperature $CONSISTENCY_TEMP"
#        run_ablation_experiment "$mn" "$ap" "$ak" "$ue" "$K_CONSISTENCY" "Consistency${num_paths}Paths" "$cli_arg_consistency"
#    done
#done


echo ""
echo "**********************************************************************"
echo "All Ablation Experiments Defined in the Script Have Been Attempted."
echo "Ablation LLM results are in: $ABLATION_RESULTS_CSV"
echo "Detailed run outputs (prompts/responses) are in subfolders of: $RESULTS_DIR_ABLATION"
echo "**********************************************************************"
