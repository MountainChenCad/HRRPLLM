#!/bin/bash

echo "Starting Ablation Experiments Script"

# --- Base Configuration for Ablation ---
DEFAULT_N_WAY=5
DEFAULT_Q_QUERY=1
DEFAULT_NUM_FSL_TASKS=10 # Reduce tasks for faster ablation runs, adjust as needed
DEFAULT_DATASET_KEY="simulated"
DEFAULT_LIMIT_TEST_SAMPLES="None"
ROOT_RESULTS_DIR="results_ablation" # Separate results directory for ablations

PYTHON_EXECUTABLE="python"
MAIN_SCRIPT_PATH="src/main_experiment.py"
SKIP_SVM_ARG="--skip_svm_baseline"

# --- API Keys (same as run_experiments.sh) ---
DEEPSEEK_PLATFORM_API_KEY="sk-g0UhooVGGk8r9l3pDZjHbQ"
PROXY_API_KEY="sk-fim815RQQnKhcbB7C5LbKjdkJsc7YyMjn6L6d6FReGJj5kUZ"
DEEPSEEK_PLATFORM_BASE_URL="https://api.deepseek.com/v1"
OPENAI_PROXY_BASE_URL="https://api.openai-proxy.live/v1"
ANTHROPIC_PROXY_BASE_URL="https://api.openai-proxy.live/anthropic"
GOOGLE_PROXY_API_ENDPOINT="https://api.openai-proxy.live/google"


# --- Representative Models for Ablation ---
# Choose 1-2 models that are fast or representative
# Format: "MODEL_NAME|API_PROVIDER|API_KEY_VAR_NAME|BASE_URL_VAR_NAME_OR_ENDPOINT_VAR_NAME"
ABLATION_MODELS=(
    "gpt-4o-mini|openai|PROXY_API_KEY|OPENAI_PROXY_BASE_URL"
    "deepseek-chat|deepseek_platform|DEEPSEEK_PLATFORM_API_KEY|DEEPSEEK_PLATFORM_BASE_URL"
    # "claude-3-haiku-20240307|anthropic|PROXY_API_KEY|ANTHROPIC_PROXY_BASE_URL"
    # "gemini-1.5-flash-latest|google|PROXY_API_KEY|GOOGLE_PROXY_API_ENDPOINT"
)

mkdir -p "$ROOT_RESULTS_DIR"

# --- Function to run a single ablation experiment ---
run_ablation_experiment() {
    local model_name="$1"
    local api_provider="$2"
    local api_key_var="$3"
    local url_endpoint_var="$4"
    local k_shot="$5"
    local tag_suffix="$6" # For experiment tag
    local extra_cli_args="$7" # String of additional CLI args for this specific ablation

    local current_api_key="${!api_key_var}"
    local current_url_or_endpoint="${!url_endpoint_var}"
    local experiment_tag_final="Ablation_${tag_suffix}_N${DEFAULT_N_WAY}_K${k_shot}_Q${DEFAULT_Q_QUERY}_${model_name//\//_}"

    CMD_ARGS=(
        "--model_name" "$model_name"
        "--api_provider" "$api_provider"
        "--api_key" "$current_api_key"
        "--dataset_key" "$DEFAULT_DATASET_KEY"
        "--n_way" "$DEFAULT_N_WAY"
        "--k_shot_support" "$k_shot"
        "--q_shot_query" "$DEFAULT_Q_QUERY"
        "--num_fsl_tasks" "$DEFAULT_NUM_FSL_TASKS"
        "--experiment_tag" "$experiment_tag_final"
        $SKIP_SVM_ARG
    )

    if [ "$api_provider" == "google" ]; then
        CMD_ARGS+=("--google_api_endpoint" "$current_url_or_endpoint")
    elif [ -n "$current_url_or_endpoint" ]; then
        CMD_ARGS+=("--base_url" "$current_url_or_endpoint")
    fi

    if [ "$DEFAULT_LIMIT_TEST_SAMPLES" != "None" ]; then
        CMD_ARGS+=("--limit_test_samples" "$DEFAULT_LIMIT_TEST_SAMPLES")
    fi

    # Add extra CLI args for the specific ablation
    if [ -n "$extra_cli_args" ]; then
        # Use eval to correctly parse space-separated args in the string
        eval "CMD_ARGS+=($extra_cli_args)"
    fi

    # Override RESULTS_BASE_DIR for ablation script
    CMD_ARGS+=("--results_base_dir" "$ROOT_RESULTS_DIR")


    echo ""
    echo "----------------------------------------------------------------------"
    echo "ABLATION RUN: Model=$model_name, K=$k_shot, Tag Suffix=$tag_suffix"
    echo "Extra CLI: $extra_cli_args"
    echo "Full Command: $PYTHON_EXECUTABLE $MAIN_SCRIPT_PATH ${CMD_ARGS[*]}"
    echo "----------------------------------------------------------------------"

    "$PYTHON_EXECUTABLE" "$MAIN_SCRIPT_PATH" "${CMD_ARGS[@]}"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Ablation for $model_name (K=$k_shot, Tag=$tag_suffix) failed with exit code $EXIT_CODE."
    else
        echo "Completed: $model_name (K=$k_shot, Tag=$tag_suffix)"
    fi
    echo "----------------------------------------------------------------------"
    sleep 3
}


# === I. Baseline (Full Prompt, K=1 and K=5 for reference) ===
K_BASELINE_1=1
K_BASELINE_5=5 # Or K=3 as per your table
for model_config in "${ABLATION_MODELS[@]}"; do
    IFS='|' read -r mn ap ak ue <<< "$model_config"
    run_ablation_experiment "$mn" "$ap" "$ak" "$ue" "$K_BASELINE_1" "BaselineK1" ""
    run_ablation_experiment "$mn" "$ap" "$ak" "$ue" "$K_BASELINE_5" "BaselineK${K_BASELINE_5}" ""
done

# === II. Prompt Component Ablation (using K=1) ===
K_PROMPT_ABLATION=1
PROMPT_ABLATIONS=(
    "NoSysInstr|--prompt_no_system_instruction"
    "NoBgKnow|--prompt_no_background_knowledge"
    "NoCandList|--prompt_no_candidate_list"
    "NoOutFmt|--prompt_no_output_format"
    "NoSys_NoCand_NoFmt|--prompt_no_system_instruction --prompt_no_candidate_list --prompt_no_output_format" # Extreme case
)
for model_config in "${ABLATION_MODELS[@]}"; do
    IFS='|' read -r mn ap ak ue <<< "$model_config"
    for ablation_setting in "${PROMPT_ABLATIONS[@]}"; do
        IFS='|' read -r tag cli_arg <<< "$ablation_setting"
        run_ablation_experiment "$mn" "$ap" "$ak" "$ue" "$K_PROMPT_ABLATION" "$tag" "$cli_arg"
    done
done

# === III. SC Information Quality Ablation (using K=1, Full Prompt otherwise) ===
K_SC_QUALITY_ABLATION=1
SC_QUALITY_ABLATIONS=(
    # Max Centers to Keep (Ensure --force_data_preprocessing is used if SC files need regeneration)
    "SC_Max3|--sc_max_centers_to_keep 3 --force_data_preprocessing"
    "SC_Max15|--sc_max_centers_to_keep 15 --force_data_preprocessing"
    # SC Encoding Precision (Amplitude)
    "SC_AmpPrec1|--sc_encoding_precision_amp 1 --force_data_preprocessing" # May need re-extraction if encoding done during SC saving
    "SC_AmpPrec5|--sc_encoding_precision_amp 5 --force_data_preprocessing"
)
# Run with default SC settings first (max 10, amp prec 3) - already covered by BaselineK1

for model_config in "${ABLATION_MODELS[@]}"; do
    IFS='|' read -r mn ap ak ue <<< "$model_config"
    for ablation_setting in "${SC_QUALITY_ABLATIONS[@]}"; do
        IFS='|' read -r tag cli_arg <<< "$ablation_setting"
        run_ablation_experiment "$mn" "$ap" "$ak" "$ue" "$K_SC_QUALITY_ABLATION" "$tag" "$cli_arg"
    done
done
# After SC quality ablations that force preprocessing, run a baseline again to ensure subsequent tests use default SC params
# if you don't want to re-process for every single ablation in other categories.
# This is a bit tricky to manage perfectly without very granular control or separate SC files per setting.
# For simplicity, the --force_data_preprocessing will re-extract SCs with the new parameters.

# === IV. 0-Shot vs. Few-Shot (K=0 vs. K=1, K=5 - Full Prompt otherwise) ===
# K=1 and K=5 are already run as baselines. We just need K=0.
K_ZERO_SHOT=0
for model_config in "${ABLATION_MODELS[@]}"; do
    IFS='|' read -r mn ap ak ue <<< "$model_config"
    run_ablation_experiment "$mn" "$ap" "$ak" "$ue" "$K_ZERO_SHOT" "ZeroShotK0" ""
done


echo ""
echo "**********************************************************************"
echo "All Ablation Experiments Defined in the Script Have Been Attempted."
echo "**********************************************************************"