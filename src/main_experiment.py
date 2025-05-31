# src/main_experiment.py
import os
import numpy as np
import json
import argparse
import csv  # For CSV output
from datetime import datetime
import random
import re
from collections import Counter  # For consistency voting
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import (
    DEFAULT_DATASET_KEY, AVAILABLE_DATASETS, TARGET_HRRP_LENGTH,
    PREPROCESS_MAT_TO_NPY, PROCESSED_DATA_DIR, TEST_SPLIT_SIZE, RANDOM_STATE,
    SCATTERING_CENTER_EXTRACTION as DEFAULT_SC_EXTRACTION_CONFIG,
    SCATTERING_CENTER_ENCODING as DEFAULT_SC_ENCODING_CONFIG,
    DEFAULT_FSL_TASK_SETUP, DEFAULT_LLM_CALLER_PARAMS,
    DEFAULT_NUM_CONSISTENCY_PATHS, DEFAULT_CONSISTENCY_TEMPERATURE,  # Added
    RESULTS_BASE_DIR, RUN_BASELINE_SVM, BASELINE_SVM_PARAMS,  # RUN_BASELINE_RF, BASELINE_RF_PARAMS,
    DEFAULT_LIMIT_TEST_SAMPLES
)
from data_utils import prepare_npy_data_and_scattering_centers, load_processed_data, build_fsl_tasks
from scattering_center_encoder import encode_all_sc_sets_to_text
from prompt_constructor_sc import PromptConstructorSC
# baseline_evaluator is now run separately by the bash script
from api_callers import OpenAICaller, AnthropicCaller, GoogleCaller


def parse_llm_output_for_label(llm_response_text, class_names, open_ended_match=False):
    # ... (remains the same as previous version with English prompt)
    if not llm_response_text: return None

    if open_ended_match:  # Try more general matching if no candidate list or format instruction
        processed_response_early = llm_response_text.lower().strip()
        # Remove common lead-in phrases if open-ended
        lead_ins = ["the predicted target class is", "predicted target class:", "my prediction is", "the target is",
                    "i believe the target is"]
        for lead_in in lead_ins:
            if processed_response_early.startswith(lead_in):
                processed_response_early = processed_response_early[len(lead_in):].strip()

        sorted_class_names_early = sorted(class_names, key=len, reverse=True)
        for cn_early in sorted_class_names_early:
            # Exact match or near exact match
            if re.search(r'\b' + re.escape(cn_early.lower()) + r'\b', processed_response_early):
                return cn_early
            if cn_early.lower() in processed_response_early:  # More lenient substring match
                # Add checks to avoid partial matches like "F-2" for "F-22" if "F-2" is not a class
                # This heuristic might need refinement based on observed LLM outputs
                if len(cn_early) > 3 or processed_response_early.index(
                        cn_early.lower()) == 0:  # Prioritize longer or starting matches
                    return cn_early

    # Original structured match
    match = re.search(r"Predicted Target Class:\s*([^\n`]+)`?", llm_response_text, re.IGNORECASE)
    if match:
        extracted_name = match.group(1).strip().rstrip('.')  # Remove trailing period
        for cn in class_names:
            if cn.lower() == extracted_name.lower(): return cn
        # If specific format used but name not in list, try a fuzzy match on extracted_name
        # This is optional and can be complex. For now, we rely on the general fallback.

    # General fallback
    processed_response = llm_response_text.lower()
    chars_to_remove = ['.', ',', ':', '"', '\'', '：', '`', '[', ']', '『', '』', '【', '】', '(', ')', '（', '）',
                       '*'];  # Added asterisk
    for char_to_remove in chars_to_remove:
        processed_response = processed_response.replace(char_to_remove, "")

    sorted_class_names = sorted(class_names, key=len, reverse=True)

    for cn in sorted_class_names:
        processed_cn = cn.lower()
        for char_to_remove in chars_to_remove:
            processed_cn = processed_cn.replace(char_to_remove, "")
        if not processed_cn: continue

        # Check for whole word match first if candidate list was provided (not open_ended)
        if not open_ended_match and re.search(r'\b' + re.escape(processed_cn) + r'\b', processed_response):
            return cn
        # Then general substring match
        if processed_cn in processed_response:
            return cn

    return None


def get_api_caller(provider_name, model_name, api_key, base_url, google_api_endpoint, llm_params):
    # ... (remains the same)
    common_params = {
        "model_name": model_name,
        "api_key": api_key,
        **llm_params
    }
    if provider_name.lower() in ["openai", "deepseek_platform", "zhipuai_glm"]:
        return OpenAICaller(base_url=base_url, **common_params)
    elif provider_name.lower() == "anthropic":
        return AnthropicCaller(base_url=base_url, **common_params)
    elif provider_name.lower() == "google":
        return GoogleCaller(google_api_endpoint=google_api_endpoint, **common_params)
    else:
        raise ValueError(f"Unsupported API provider: {provider_name}")


def run_fsl_experiment_main(current_config):
    dataset_key = current_config.dataset_key
    # ... (print statements remain similar, add consistency path info) ...
    print(
        f"\n{'=' * 40}\nProcessing Dataset: {dataset_key} with Model: {current_config.model_name} (Provider: {current_config.api_provider})\n"
        f"FSL: {current_config.fsl_setup['n_way']}W-{current_config.fsl_setup['k_shot_support']}K-{current_config.fsl_setup['q_shot_query']}Q, Tasks: {current_config.fsl_setup['num_fsl_tasks']}\n"
        f"Consistency Paths: {current_config.num_consistency_paths}, Consistency Temp: {current_config.consistency_temperature if current_config.num_consistency_paths > 1 else 'N/A'}\n"
        f"SC Max Centers: {current_config.sc_extraction_config['max_centers_to_keep']}, SC Amp Precision: {current_config.sc_encoding_config['precision_amp']}\n"
        f"Prompt Ablations: SysInstr={current_config.prompt_include_system_instruction}, BgKnow={current_config.prompt_include_background_knowledge}, CandList={current_config.prompt_include_candidate_list}, OutFmt={current_config.prompt_include_output_format}\n"
        f"Experiment Tag: {current_config.experiment_tag}\n{'=' * 40}"
    )

    load_result = load_processed_data(dataset_key, current_config, load_scattering_centers=True)
    if load_result[0] is None and load_result[2] is None:
        print(f"Dataset '{dataset_key}' failed to load. Skipping FSL experiment.");
        return {}  # Return empty dict on failure

    _, _, X_meta_test_hrrp, y_meta_test_original, \
        _, X_meta_test_sc_list, label_encoder, class_names_all_dataset = load_result

    if not current_config.sc_extraction_config["enabled"]:
        print(f"SC extraction disabled. Skipping FSL(SC) experiment.");
        return {}

    if X_meta_test_hrrp is None or X_meta_test_hrrp.size == 0:
        print(f"Meta-test data is empty for '{dataset_key}'. Cannot build FSL tasks. Skipping.");
        return {}

    if current_config.fsl_setup["k_shot_support"] > 0 and \
            (X_meta_test_sc_list is None or (
                    isinstance(X_meta_test_sc_list, list) and not X_meta_test_sc_list and X_meta_test_hrrp.size > 0)):
        print(f"Warning: SC data for meta-test set of '{dataset_key}' is missing or empty, but K>0.");

    data_pool_hrrp = X_meta_test_hrrp
    data_pool_original_labels = y_meta_test_original
    data_pool_sc_list = X_meta_test_sc_list

    if current_config.limit_test_samples is not None and current_config.limit_test_samples > 0 and \
            len(X_meta_test_hrrp) > 0 and current_config.limit_test_samples < len(X_meta_test_hrrp):
        # ... (limit test samples logic remains same)
        num_available = len(X_meta_test_hrrp)
        num_to_sample = min(current_config.limit_test_samples, num_available)
        print(f"Meta-test data pool will be sampled from {num_available} to {num_to_sample} for FSL task construction.")
        np.random.seed(current_config.RANDOM_STATE)
        indices = np.arange(num_available)
        sampled_indices = np.random.choice(indices, num_to_sample, replace=False)
        data_pool_hrrp = X_meta_test_hrrp[sampled_indices]
        data_pool_original_labels = y_meta_test_original[sampled_indices]
        if X_meta_test_sc_list:
            data_pool_sc_list = [X_meta_test_sc_list[i] for i in sampled_indices]
        else:
            data_pool_sc_list = None
        print(f"  Limited data pool size: {len(data_pool_hrrp)}")

    fsl_tasks = build_fsl_tasks(
        data_pool_hrrp, data_pool_original_labels, data_pool_sc_list,
        label_encoder, current_config.fsl_setup,
        current_config.sc_encoding_config,
        current_config.sc_extraction_config,
        current_config.RANDOM_STATE
    )
    if not fsl_tasks:
        print(f"Dataset '{dataset_key}' failed to build FSL tasks. Skipping experiment.");
        return {}

    # Determine temperature for LLM calls
    effective_temperature = current_config.consistency_temperature if current_config.num_consistency_paths > 1 else \
    current_config.llm_caller_params["temperature"]

    # Create a copy of llm_caller_params to modify temperature for this run
    current_run_llm_params = {**current_config.llm_caller_params, "temperature": effective_temperature}

    llm_api_caller = get_api_caller(
        current_config.api_provider, current_config.model_name, current_config.api_key,
        current_config.base_url, current_config.google_api_endpoint,
        current_run_llm_params  # Use params with potentially adjusted temperature
    )

    all_query_true_labels_original = []
    all_query_pred_llm_original_labels = []

    fsl_cfg = current_config.fsl_setup
    model_name_sanitized = current_config.model_name.replace("/", "_").replace(".", "_").replace(":", "-")
    # Create a unique run identifier for prompt/response saving if needed
    run_id_for_prompts = f"{model_name_sanitized}_K{fsl_cfg['k_shot_support']}_T{current_config.fsl_setup['num_fsl_tasks']}_{current_config.experiment_tag}"

    # --- MODIFIED: Results directory for prompts/responses is now outside the CSV writing part ---
    # It will be inside the main experiment directory defined by the bash script.
    # The bash script already creates a current_result_dir. We use that.
    # The CSV output is handled by appending to a file specified via CLI.

    # Example: current_config.RESULTS_BASE_DIR / dataset_key / model_name_sanitized / experiment_tag
    # This structure is created by the bash script.
    # For prompts and responses, we can create a subfolder there.
    # We'll use experiment_run_name for the folder, which is more specific.
    exp_name_parts_for_folder = [
        "FSL_SC_Proto", dataset_key, model_name_sanitized,
        f"{fsl_cfg['n_way']}W", f"{fsl_cfg['k_shot_support']}K", f"{fsl_cfg['q_shot_query']}Q",
        f"{len(fsl_tasks)}tasks"
    ]
    if current_config.experiment_tag: exp_name_parts_for_folder.append(current_config.experiment_tag)
    exp_folder_name = "_".join(exp_name_parts_for_folder)

    # Base results dir is passed by current_config now.
    specific_run_results_dir = os.path.join(current_config.RESULTS_BASE_DIR, dataset_key, model_name_sanitized,
                                            current_config.experiment_tag if current_config.experiment_tag else "default_run",
                                            exp_folder_name)
    os.makedirs(specific_run_results_dir, exist_ok=True)  # Ensure this specific run's folder exists

    prompts_responses_dir = os.path.join(specific_run_results_dir, "prompts_and_responses")
    os.makedirs(prompts_responses_dir, exist_ok=True)
    print(f"Debug Prompts/Responses will be saved to: {prompts_responses_dir}")

    print(f"\nStarting LLM predictions for {len(fsl_tasks)} FSL tasks (Run ID: {run_id_for_prompts})...")

    total_llm_calls = 0
    for task_idx, task_data in enumerate(
            tqdm(fsl_tasks, desc=f"FSL Task Processing ({model_name_sanitized})", leave=False, unit="task")):
        task_specific_class_names = list(task_data["task_classes"])
        prompt_constructor = PromptConstructorSC(
            dataset_key, task_specific_class_names,
            current_config.sc_encoding_config,
            include_system_instruction=current_config.prompt_include_system_instruction,
            include_background_knowledge=current_config.prompt_include_background_knowledge,
            include_candidate_list=current_config.prompt_include_candidate_list,
            include_output_format_instruction=current_config.prompt_include_output_format
        )

        prototype_sc_texts_for_prompt = task_data["support_prototypes_sc_texts"]
        prototype_labels_for_prompt = task_data["support_prototypes_labels"]

        few_shot_examples_for_prompt = []
        if fsl_cfg["k_shot_support"] > 0:
            if len(prototype_sc_texts_for_prompt) == len(prototype_labels_for_prompt) and \
                    len(prototype_sc_texts_for_prompt) == fsl_cfg['n_way']:
                for sc_text, label in zip(prototype_sc_texts_for_prompt, prototype_labels_for_prompt):
                    few_shot_examples_for_prompt.append((sc_text, label))
            else:
                few_shot_examples_for_prompt = []

        task_query_sc_texts = task_data["query_sc_texts"]
        task_query_labels = task_data["query_labels"]

        if not task_query_sc_texts or len(task_query_sc_texts) == 0: continue

        for query_idx_in_task, query_sc_text_for_prompt in enumerate(task_query_sc_texts):
            prompt_str = prompt_constructor.construct_prompt_with_sc(query_sc_text_for_prompt,
                                                                     few_shot_examples_for_prompt)

            # Store true label for this query
            all_query_true_labels_original.append(task_query_labels[query_idx_in_task])

            consistency_predictions = []
            for path_idx in range(current_config.num_consistency_paths):
                # Save prompt only for the first path of a consistency run to avoid too many files
                if path_idx == 0:
                    prompt_filename = f"prompt_task{task_idx:03d}_query{query_idx_in_task:03d}.txt"
                    prompt_filepath = os.path.join(prompts_responses_dir, prompt_filename)
                    try:
                        with open(prompt_filepath, "w", encoding="utf-8") as pf:
                            pf.write(
                                f"--- Task {task_idx}, Query {query_idx_in_task} (Path {path_idx + 1}/{current_config.num_consistency_paths}) ---\n")
                            pf.write(f"True Label: {task_query_labels[query_idx_in_task]}\n")
                            pf.write(f"Task Classes: {task_specific_class_names}\n")
                            pf.write(f"Effective Temperature: {effective_temperature}\n")
                            pf.write("--- Prompt: ---\n")
                            pf.write(prompt_str)
                    except Exception as e:
                        print(f"  Error saving prompt file {prompt_filepath}: {e}")

                llm_response = llm_api_caller.get_completion(prompt_str)
                total_llm_calls += 1

                # Save response for each path if num_consistency_paths > 1, or always
                response_filename = f"response_task{task_idx:03d}_query{query_idx_in_task:03d}_path{path_idx:02d}.txt"
                response_filepath = os.path.join(prompts_responses_dir, response_filename)

                parsed_label_for_path = None
                if llm_response:
                    try:
                        with open(response_filepath, "w", encoding="utf-8") as rf:
                            rf.write(llm_response)
                    except Exception as e:
                        print(f"  Error saving response file {response_filepath}: {e}")

                    open_ended = not current_config.prompt_include_candidate_list or not current_config.prompt_include_output_format
                    parsed_label_for_path = parse_llm_output_for_label(llm_response, task_specific_class_names,
                                                                       open_ended_match=open_ended)
                else:
                    try:
                        with open(response_filepath, "w", encoding="utf-8") as rf:
                            rf.write("LLM_RESPONSE_IS_NONE")
                    except Exception as e:
                        print(f"  Error saving empty response file {response_filepath}: {e}")

                if parsed_label_for_path:  # Only add valid parses to consistency list
                    consistency_predictions.append(parsed_label_for_path)

            # Aggregate consistency predictions
            if consistency_predictions:
                # Majority vote
                vote_counts = Counter(consistency_predictions)
                final_predicted_label = vote_counts.most_common(1)[0][0]
            else:  # No valid predictions from any path
                final_predicted_label = None

            all_query_pred_llm_original_labels.append(final_predicted_label)

    # --- Evaluation ---
    valid_preds_count = 0;
    y_true_eval, y_pred_eval = [], [];
    failed_count = 0
    for true_l, pred_l in zip(all_query_true_labels_original, all_query_pred_llm_original_labels):
        if pred_l is not None:
            y_true_eval.append(true_l);
            y_pred_eval.append(pred_l);
            valid_preds_count += 1
        else:
            failed_count += 1

    expected_total_queries = sum(len(t['query_labels']) for t in fsl_tasks)
    print(
        f"\nLLM Predictions Complete. Expected Queries: {expected_total_queries}. Total LLM API Calls: {total_llm_calls}.")
    print(
        f"Valid Aggregated Predictions: {valid_preds_count}/{len(all_query_true_labels_original)}. Failed/Unparsed (after aggregation): {failed_count}.")

    acc_val, f1_macro_val, report_dict_out = 0.0, 0.0, {}
    if valid_preds_count > 0:
        y_true_encoded = label_encoder.transform(y_true_eval)
        y_pred_encoded = label_encoder.transform(y_pred_eval)
        acc_val = accuracy_score(y_true_encoded, y_pred_encoded)
        f1_macro_val = f1_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
        # ... (classification report and confusion matrix saving can be kept if desired, but not for CSV)
        try:
            report_dict_out = classification_report(y_true_encoded, y_pred_encoded,
                                                    labels=label_encoder.transform(class_names_all_dataset),
                                                    target_names=class_names_all_dataset, output_dict=True,
                                                    zero_division=0, digits=3)
        except Exception:
            report_dict_out = {"accuracy": acc_val, "macro avg": {"f1-score": f1_macro_val}}

    # --- Prepare data for CSV output ---
    output_data_row = {
        'dataset_key': dataset_key,
        'model_name': current_config.model_name,
        'api_provider': current_config.api_provider,
        'experiment_tag': current_config.experiment_tag,
        'n_way': fsl_cfg['n_way'],
        'k_shot_support': fsl_cfg['k_shot_support'],
        'q_shot_query': fsl_cfg['q_shot_query'],
        'num_fsl_tasks': fsl_cfg['num_fsl_tasks'],
        'limit_test_samples': current_config.limit_test_samples if current_config.limit_test_samples is not None else "None",
        'temperature_llm': effective_temperature,  # Actual temp used for LLM
        'max_tokens_completion': current_config.llm_caller_params['max_tokens_completion'],
        'num_consistency_paths': current_config.num_consistency_paths,
        'prompt_sys_instr': current_config.prompt_include_system_instruction,
        'prompt_bg_know': current_config.prompt_include_background_knowledge,
        'prompt_cand_list': current_config.prompt_include_candidate_list,
        'prompt_out_fmt': current_config.prompt_include_output_format,
        'sc_max_centers': current_config.sc_extraction_config['max_centers_to_keep'],
        'sc_amp_prec': current_config.sc_encoding_config['precision_amp'],
        'accuracy': f"{acc_val:.4f}",
        'f1_macro': f"{f1_macro_val:.4f}",
        'valid_preds_count': valid_preds_count,
        'total_queries_eval': len(all_query_true_labels_original),
        'total_llm_api_calls': total_llm_calls
        # 'report_dict_json': json.dumps(report_dict_out) # Optional: if full report needed in CSV
    }
    return output_data_row  # Return the row to be written by the caller


def main():
    parser = argparse.ArgumentParser(description="Run HRRP FSL experiments with LLMs, including ablations.")
    # ... (all argparse arguments remain the same)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--api_provider", type=str, required=True,
                        choices=["openai", "anthropic", "google", "deepseek_platform", "zhipuai_glm"])
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--google_api_endpoint", type=str, default=None)
    parser.add_argument("--dataset_key", type=str, default=DEFAULT_DATASET_KEY)
    parser.add_argument("--experiment_tag", type=str, default="")
    parser.add_argument("--n_way", type=int, default=DEFAULT_FSL_TASK_SETUP["n_way"])
    parser.add_argument("--k_shot_support", type=int, default=DEFAULT_FSL_TASK_SETUP["k_shot_support"])
    parser.add_argument("--q_shot_query", type=int, default=DEFAULT_FSL_TASK_SETUP["q_shot_query"])
    parser.add_argument("--num_fsl_tasks", type=int, default=DEFAULT_FSL_TASK_SETUP["num_fsl_tasks"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_LLM_CALLER_PARAMS["temperature"])
    parser.add_argument("--max_tokens_completion", type=int, default=DEFAULT_LLM_CALLER_PARAMS["max_tokens_completion"])
    parser.add_argument("--limit_test_samples", type=int, default=None)
    parser.add_argument("--force_data_preprocessing", action='store_true')
    # parser.add_argument("--skip_svm_baseline", action='store_true') # This flag is for the bash script now
    # parser.add_argument("--skip_rf_baseline", action='store_true')  # This flag is for the bash script now

    parser.add_argument("--prompt_no_system_instruction", action='store_true')
    parser.add_argument("--prompt_no_background_knowledge", action='store_true')
    parser.add_argument("--prompt_no_candidate_list", action='store_true')
    parser.add_argument("--prompt_no_output_format", action='store_true')
    parser.add_argument("--sc_max_centers_to_keep", type=int, default=None)
    parser.add_argument("--sc_encoding_precision_amp", type=int, default=None)
    parser.add_argument("--num_consistency_paths", type=int, default=DEFAULT_NUM_CONSISTENCY_PATHS)
    parser.add_argument("--consistency_temperature", type=float, default=DEFAULT_CONSISTENCY_TEMPERATURE)
    parser.add_argument("--output_csv_llm", type=str, default="results/llm_experiments_log.csv",
                        help="Path to CSV file to append LLM experiment results.")
    parser.add_argument("--results_base_dir", type=str, default=RESULTS_BASE_DIR,
                        help="Base directory for saving detailed run outputs like prompts/responses.")

    args = parser.parse_args()

    # --- CurrentRunConfig class definition (ensure it's here or imported) ---
    class CurrentRunConfig:
        def __init__(self, cli_args):
            self.dataset_key = cli_args.dataset_key
            self.AVAILABLE_DATASETS = AVAILABLE_DATASETS
            self.TARGET_HRRP_LENGTH = TARGET_HRRP_LENGTH
            self.PREPROCESS_MAT_TO_NPY = True if cli_args.force_data_preprocessing or \
                                                 cli_args.sc_max_centers_to_keep is not None or \
                                                 cli_args.sc_encoding_precision_amp is not None \
                else PREPROCESS_MAT_TO_NPY

            self.PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
            self.TEST_SPLIT_SIZE = TEST_SPLIT_SIZE
            self.RANDOM_STATE = RANDOM_STATE

            # Set up scattering center extraction config
            self.sc_extraction_config = {**DEFAULT_SC_EXTRACTION_CONFIG}
            if cli_args.sc_max_centers_to_keep is not None:
                self.sc_extraction_config["max_centers_to_keep"] = cli_args.sc_max_centers_to_keep

            # IMPORTANT: Add SCATTERING_CENTER_EXTRACTION as an alias for compatibility
            self.SCATTERING_CENTER_EXTRACTION = self.sc_extraction_config

            self.sc_encoding_config = {**DEFAULT_SC_ENCODING_CONFIG}
            if cli_args.sc_encoding_precision_amp is not None:
                self.sc_encoding_config["precision_amp"] = cli_args.sc_encoding_precision_amp
            self.sc_encoding_config["TARGET_HRRP_LENGTH_INFO"] = self.TARGET_HRRP_LENGTH

            self.fsl_setup = {
                "enabled": True, "n_way": cli_args.n_way, "k_shot_support": cli_args.k_shot_support,
                "q_shot_query": cli_args.q_shot_query, "num_fsl_tasks": cli_args.num_fsl_tasks,
                "sc_feature_type_for_prototype": DEFAULT_FSL_TASK_SETUP["sc_feature_type_for_prototype"]
            }

            self.llm_caller_params = {
                "temperature": cli_args.temperature, "top_p": DEFAULT_LLM_CALLER_PARAMS["top_p"],
                "max_tokens_completion": cli_args.max_tokens_completion,
                "frequency_penalty": DEFAULT_LLM_CALLER_PARAMS["frequency_penalty"],
                "presence_penalty": DEFAULT_LLM_CALLER_PARAMS["presence_penalty"],
                "api_retry_delay": DEFAULT_LLM_CALLER_PARAMS["api_retry_delay"],
                "max_retries": DEFAULT_LLM_CALLER_PARAMS["max_retries"]
            }

            self.model_name = cli_args.model_name
            self.api_key = cli_args.api_key
            self.api_provider = cli_args.api_provider
            self.base_url = cli_args.base_url
            self.google_api_endpoint = cli_args.google_api_endpoint

            self.limit_test_samples = cli_args.limit_test_samples if cli_args.limit_test_samples is not None else DEFAULT_LIMIT_TEST_SAMPLES

            self.RESULTS_BASE_DIR = cli_args.results_base_dir

            self.experiment_tag = cli_args.experiment_tag

            self.prompt_include_system_instruction = not cli_args.prompt_no_system_instruction
            self.prompt_include_background_knowledge = not cli_args.prompt_no_background_knowledge
            self.prompt_include_candidate_list = not cli_args.prompt_no_candidate_list
            self.prompt_include_output_format = not cli_args.prompt_no_output_format

            self.num_consistency_paths = cli_args.num_consistency_paths
            self.consistency_temperature = cli_args.consistency_temperature
            self.output_csv_llm = cli_args.output_csv_llm

    config_obj = CurrentRunConfig(args)

    os.makedirs(os.path.dirname(config_obj.output_csv_llm), exist_ok=True)

    start_time_total = datetime.now()

    if config_obj.PREPROCESS_MAT_TO_NPY:
        print("\n--- Step 0: Preparing/Re-preparing .npy HRRP data and SC .pkl files ---")
        prepare_npy_data_and_scattering_centers(config_obj)
    else:
        print("\n--- Step 0: Skipping data preprocessing. Using existing files. ---")

    if config_obj.sc_extraction_config["enabled"] and config_obj.fsl_setup["enabled"]:
        if args.num_fsl_tasks == 0 and args.model_name == "dummy_preprocess_check":  # Special case for bash script's preprocess check
            print("Preprocessing check complete (num_fsl_tasks is 0). Skipping LLM run.")
        else:
            experiment_results_row = run_fsl_experiment_main(config_obj)
            if experiment_results_row:
                fieldnames = [
                    'dataset_key', 'model_name', 'api_provider', 'experiment_tag',
                    'n_way', 'k_shot_support', 'q_shot_query', 'num_fsl_tasks',
                    'limit_test_samples', 'temperature_llm', 'max_tokens_completion',
                    'num_consistency_paths',
                    'prompt_sys_instr', 'prompt_bg_know', 'prompt_cand_list', 'prompt_out_fmt',
                    'sc_max_centers', 'sc_amp_prec',
                    'accuracy', 'f1_macro', 'valid_preds_count', 'total_queries_eval', 'total_llm_api_calls'
                ]
                file_exists = os.path.isfile(config_obj.output_csv_llm)
                with open(config_obj.output_csv_llm, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if not file_exists or os.path.getsize(config_obj.output_csv_llm) == 0:
                        writer.writeheader()
                    writer.writerow(experiment_results_row)
                print(f"LLM experiment results appended to {config_obj.output_csv_llm}")
            else:
                print("LLM experiment did not yield results to save to CSV.")

    # --- REMOVED DIRECT BASELINE CALLS FROM HERE ---
    # The bash scripts (run_experiments.sh / ablation.sh) will call
    # src/baseline_evaluator.py directly.
    # if config_obj.RUN_BASELINE_SVM: # This logic is now in the bash script
    #     try:
    #         print(f"\n--- Running SVM Baseline for dataset '{config_obj.dataset_key}' ---")
    #         run_svm_baseline_for_dataset(config_obj.dataset_key, config_obj) # THIS WAS THE PROBLEMATIC CALL
    #     except Exception as e: print(f"Error running SVM baseline for {config_obj.dataset_key}: {e}")

    print(
        f"\nRun for {config_obj.model_name} with tag '{config_obj.experiment_tag}' finished. Total time: {datetime.now() - start_time_total}")


if __name__ == "__main__":
    # The main results directory (e.g., "results/" or "results_ablation/")
    # is now passed via CLI to CurrentRunConfig.
    # We don't need to create RESULTS_BASE_DIR here anymore as it's part of the config.
    main()