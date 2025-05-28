# src/main_experiment.py
# ... (imports remain the same) ...
import os
import numpy as np
import json
from datetime import datetime
import random
import re
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import (
    DEFAULT_DATASET_KEY, AVAILABLE_DATASETS, TARGET_HRRP_LENGTH,
    PREPROCESS_MAT_TO_NPY, PROCESSED_DATA_DIR, TEST_SPLIT_SIZE, RANDOM_STATE,
    SCATTERING_CENTER_EXTRACTION as DEFAULT_SC_EXTRACTION_CONFIG,  # Use as default
    SCATTERING_CENTER_ENCODING as DEFAULT_SC_ENCODING_CONFIG,  # Use as default
    DEFAULT_FSL_TASK_SETUP,
    DEFAULT_LLM_CALLER_PARAMS,
    RESULTS_BASE_DIR, RUN_BASELINE_SVM, BASELINE_SVM_PARAMS,
    DEFAULT_LIMIT_TEST_SAMPLES
)
from data_utils import prepare_npy_data_and_scattering_centers, load_processed_data, build_fsl_tasks
from scattering_center_encoder import encode_all_sc_sets_to_text
from prompt_constructor_sc import PromptConstructorSC
from baseline_evaluator import run_svm_baseline_for_dataset
from api_callers import OpenAICaller, AnthropicCaller, GoogleCaller


def parse_llm_output_for_label(llm_response_text, class_names, open_ended_match=False):  # Added open_ended_match
    if not llm_response_text: return None

    # If not expecting specific format (e.g., include_output_format_instruction=False)
    # or if no candidate list was provided, we do more general matching first.
    if open_ended_match:
        processed_response_early = llm_response_text.lower().strip()
        sorted_class_names_early = sorted(class_names, key=len, reverse=True)
        for cn_early in sorted_class_names_early:
            # Try to find class name as a whole word or at the beginning/end
            # This is a simple heuristic for open-ended matching
            if re.search(r'\b' + re.escape(cn_early.lower()) + r'\b', processed_response_early):
                return cn_early
            if processed_response_early.startswith(cn_early.lower()):
                return cn_early
            if processed_response_early.endswith(cn_early.lower()):
                return cn_early
        # If still no match, fall through to original logic for robustness

    match = re.search(r"预测目标类别：\s*([^\n`]+)`?", llm_response_text, re.IGNORECASE)
    if match:
        extracted_name = match.group(1).strip()
        for cn in class_names:
            if cn.lower() == extracted_name.lower(): return cn

    processed_response = llm_response_text.lower()
    chars_to_remove = ['.', ',', ':', '"', '\'', '：', '`', '[', ']', '『', '』', '【', '】', '(', ')', '（', '）'];
    for char_to_remove in chars_to_remove:
        processed_response = processed_response.replace(char_to_remove, "")

    sorted_class_names = sorted(class_names, key=len, reverse=True)

    for cn in sorted_class_names:
        processed_cn = cn.lower()
        for char_to_remove in chars_to_remove:
            processed_cn = processed_cn.replace(char_to_remove, "")
        if not processed_cn: continue

        if processed_cn in processed_response.split():
            return cn
        if processed_cn in processed_response:
            return cn

    return None


def get_api_caller(provider_name, model_name, api_key, base_url, google_api_endpoint, llm_params):
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


def run_fsl_experiment_main(current_config):  # current_config is now an object
    dataset_key = current_config.dataset_key
    print(
        f"\n{'=' * 40}\nProcessing Dataset: {dataset_key} with Model: {current_config.model_name} (Provider: {current_config.api_provider})\n"
        f"FSL: {current_config.fsl_setup['n_way']}W-{current_config.fsl_setup['k_shot_support']}K-{current_config.fsl_setup['q_shot_query']}Q, Tasks: {current_config.fsl_setup['num_fsl_tasks']}\n"
        f"SC Max Centers: {current_config.sc_extraction_config['max_centers_to_keep']}, SC Amp Precision: {current_config.sc_encoding_config['precision_amp']}\n"
        f"Prompt Ablations: SysInstr={current_config.prompt_include_system_instruction}, BgKnow={current_config.prompt_include_background_knowledge}, CandList={current_config.prompt_include_candidate_list}, OutFmt={current_config.prompt_include_output_format}\n"
        f"Experiment Tag: {current_config.experiment_tag}\n{'=' * 40}")

    # --- Data Loading ---
    # NOTE: prepare_npy_data_and_scattering_centers uses SC config from current_config for extraction
    # So, if max_centers_to_keep is changed by CLI, it should be reflected when data is (re)processed.
    # For ablation on SC quality, ensure PREPROCESS_MAT_TO_NPY is True for that run, or manage SC files separately.

    load_result = load_processed_data(dataset_key, current_config, load_scattering_centers=True)
    if load_result[0] is None and load_result[2] is None:
        print(f"Dataset '{dataset_key}' failed to load. Skipping FSL experiment.");
        return

    _, _, X_meta_test_hrrp, y_meta_test_original, \
        _, X_meta_test_sc_list, label_encoder, class_names_all_dataset = load_result

    if not current_config.sc_extraction_config["enabled"]:  # Use current_config's SC config
        print(f"SC extraction disabled. Skipping FSL(SC) experiment.");
        return

    if X_meta_test_hrrp is None or X_meta_test_hrrp.size == 0:
        print(f"Meta-test data is empty for '{dataset_key}'. Cannot build FSL tasks. Skipping.");
        return

    if current_config.fsl_setup["k_shot_support"] > 0 and \
            (X_meta_test_sc_list is None or (
                    isinstance(X_meta_test_sc_list, list) and not X_meta_test_sc_list and X_meta_test_hrrp.size > 0)):
        print(f"Warning: SC data for meta-test set of '{dataset_key}' is missing or empty, but K>0.");

    data_pool_hrrp = X_meta_test_hrrp
    data_pool_original_labels = y_meta_test_original
    data_pool_sc_list = X_meta_test_sc_list

    if current_config.limit_test_samples is not None and current_config.limit_test_samples > 0 and \
            len(X_meta_test_hrrp) > 0 and current_config.limit_test_samples < len(X_meta_test_hrrp):
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

    # --- Build FSL Tasks ---
    # build_fsl_tasks now uses sc_extraction_config and sc_encoding_config from current_config
    fsl_tasks = build_fsl_tasks(
        data_pool_hrrp, data_pool_original_labels, data_pool_sc_list,
        label_encoder, current_config.fsl_setup,
        current_config.sc_encoding_config,  # Use current config's encoding
        current_config.sc_extraction_config,  # Use current config's extraction
        current_config.RANDOM_STATE
    )
    if not fsl_tasks:
        print(f"Dataset '{dataset_key}' failed to build FSL tasks. Skipping experiment.");
        return

    # --- Initialize LLM Caller ---
    llm_api_caller = get_api_caller(
        current_config.api_provider, current_config.model_name, current_config.api_key,
        current_config.base_url, current_config.google_api_endpoint, current_config.llm_caller_params
    )

    all_query_true_labels_original = []
    all_query_pred_llm_original_labels = []

    fsl_cfg = current_config.fsl_setup
    model_name_sanitized = current_config.model_name.replace("/", "_").replace(".", "_")
    exp_name_parts = [
        "FSL_SC_Proto", dataset_key, model_name_sanitized,
        f"{fsl_cfg['n_way']}W", f"{fsl_cfg['k_shot_support']}K", f"{fsl_cfg['q_shot_query']}Q",
        f"{len(fsl_tasks)}tasks"
    ]
    if current_config.experiment_tag: exp_name_parts.append(current_config.experiment_tag)
    experiment_run_name = "_".join(exp_name_parts)

    current_result_dir = os.path.join(current_config.RESULTS_BASE_DIR, dataset_key, model_name_sanitized,
                                      current_config.experiment_tag if current_config.experiment_tag else "default_run")
    os.makedirs(current_result_dir, exist_ok=True)

    prompts_responses_dir = os.path.join(current_result_dir, "prompts_and_responses_tasks")
    os.makedirs(prompts_responses_dir, exist_ok=True)
    print(f"Prompts and Responses will be saved to: {prompts_responses_dir}")

    print(f"\nStarting LLM predictions for {len(fsl_tasks)} FSL tasks (Run: {experiment_run_name})...")

    total_queries_processed_in_experiment = 0
    for task_idx, task_data in enumerate(
            tqdm(fsl_tasks, desc=f"FSL Task Processing ({model_name_sanitized})", leave=False, unit="task")):
        task_specific_class_names = list(task_data["task_classes"])

        # --- Initialize Prompt Constructor with Ablation Flags ---
        prompt_constructor = PromptConstructorSC(
            dataset_key, task_specific_class_names,
            current_config.sc_encoding_config,  # Use current SC encoding
            include_system_instruction=current_config.prompt_include_system_instruction,
            include_background_knowledge=current_config.prompt_include_background_knowledge,
            include_candidate_list=current_config.prompt_include_candidate_list,
            include_output_format_instruction=current_config.prompt_include_output_format
        )

        prototype_sc_texts_for_prompt = task_data["support_prototypes_sc_texts"]
        prototype_labels_for_prompt = task_data["support_prototypes_labels"]

        few_shot_examples_for_prompt = []
        if fsl_cfg["k_shot_support"] > 0:  # K=0 means 0-shot, no prototypes used in prompt
            if len(prototype_sc_texts_for_prompt) == len(prototype_labels_for_prompt) and \
                    len(prototype_sc_texts_for_prompt) == fsl_cfg['n_way']:
                for sc_text, label in zip(prototype_sc_texts_for_prompt, prototype_labels_for_prompt):
                    few_shot_examples_for_prompt.append((sc_text, label))
            else:
                print(f"  Warning: Task {task_idx + 1} K>0, but prototype data inconsistent. Using 0-shot for prompt.")
                few_shot_examples_for_prompt = []

        task_query_sc_texts = task_data["query_sc_texts"]
        task_query_labels = task_data["query_labels"]

        if not task_query_sc_texts or len(task_query_sc_texts) == 0: continue

        task_query_pred_labels = []
        for query_idx_in_task, query_sc_text_for_prompt in enumerate(task_query_sc_texts):
            prompt_str = prompt_constructor.construct_prompt_with_sc(query_sc_text_for_prompt,
                                                                     few_shot_examples_for_prompt)

            prompt_filename = f"prompt_task{task_idx:03d}_query{query_idx_in_task:03d}.txt"
            prompt_filepath = os.path.join(prompts_responses_dir, prompt_filename)
            try:
                with open(prompt_filepath, "w", encoding="utf-8") as pf:
                    pf.write(f"--- Task {task_idx}, Query {query_idx_in_task} ---\n")
                    pf.write(f"True Label (for this query): {task_query_labels[query_idx_in_task]}\n")
                    pf.write(f"Task Classes: {task_specific_class_names}\n")
                    pf.write("--- Prompt Sent to LLM (using class prototypes): ---\n")
                    pf.write(prompt_str)
            except Exception as e:
                print(f"  Error saving prompt file {prompt_filepath}: {e}")

            llm_response = llm_api_caller.get_completion(prompt_str)
            predicted_label = None

            response_filename = f"response_task{task_idx:03d}_query{query_idx_in_task:03d}.txt"
            response_filepath = os.path.join(prompts_responses_dir, response_filename)
            if llm_response:
                try:
                    with open(response_filepath, "w", encoding="utf-8") as rf:
                        rf.write(llm_response)
                except Exception as e:
                    print(f"  Error saving response file {response_filepath}: {e}")
                # For open-ended matching, pass a flag to parse_llm_output_for_label
                open_ended = not current_config.prompt_include_candidate_list or not current_config.prompt_include_output_format
                predicted_label = parse_llm_output_for_label(llm_response, task_specific_class_names,
                                                             open_ended_match=open_ended)
            else:
                try:
                    with open(response_filepath, "w", encoding="utf-8") as rf:
                        rf.write("LLM_RESPONSE_IS_NONE")
                except Exception as e:
                    print(f"  Error saving empty response file {response_filepath}: {e}")

            task_query_pred_labels.append(predicted_label)
            total_queries_processed_in_experiment += 1

        all_query_true_labels_original.extend(task_query_labels)
        all_query_pred_llm_original_labels.extend(task_query_pred_labels)

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
        f"\nLLM Predictions Complete. Expected Queries: {expected_total_queries}. Actual LLM Calls: {total_queries_processed_in_experiment}.")
    print(
        f"Valid Predictions: {valid_preds_count}/{len(all_query_true_labels_original)}. Failed/Unparsed: {failed_count}.")

    acc_val, f1_macro_val, report_dict = 0.0, 0.0, {}
    # ... (rest of evaluation and summary saving logic remains largely the same) ...
    # Ensure summary includes the ablation flags.
    if valid_preds_count > 0:
        y_true_encoded = label_encoder.transform(y_true_eval)
        y_pred_encoded = label_encoder.transform(y_pred_eval)

        acc_val = accuracy_score(y_true_encoded, y_pred_encoded)
        f1_macro_val = f1_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
        report_labels_encoded = label_encoder.transform(class_names_all_dataset)
        try:
            report_str = classification_report(y_true_encoded, y_pred_encoded, labels=report_labels_encoded,
                                               target_names=class_names_all_dataset, zero_division=0, digits=3)
            report_dict = classification_report(y_true_encoded, y_pred_encoded, labels=report_labels_encoded,
                                                target_names=class_names_all_dataset, output_dict=True, zero_division=0,
                                                digits=3)
        except Exception as e:
            print(f"Classification report generation error: {e}. Trying with present labels.")
            present_labels_encoded = np.unique(np.concatenate((y_true_encoded, y_pred_encoded)))
            present_class_names = list(label_encoder.inverse_transform(present_labels_encoded))
            report_str = classification_report(y_true_encoded, y_pred_encoded, labels=present_labels_encoded,
                                               target_names=present_class_names, zero_division=0, digits=3)
            report_dict = classification_report(y_true_encoded, y_pred_encoded, labels=present_labels_encoded,
                                                target_names=present_class_names, output_dict=True, zero_division=0,
                                                digits=3)

        print("\nLLM FSL Aggregated Classification Report:\n", report_str)
        print(f"Accuracy: {acc_val:.4f}, F1-macro: {f1_macro_val:.4f}")

        if valid_preds_count > 1:
            cm_labels_encoded = label_encoder.transform(class_names_all_dataset)
            cm_class_names_display = class_names_all_dataset
            cm = confusion_matrix(y_true_encoded, y_pred_encoded, labels=cm_labels_encoded)
            plt.figure(figsize=(max(8, len(cm_class_names_display) * 0.6), max(6, len(cm_class_names_display) * 0.45)))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_class_names_display,
                        yticklabels=cm_class_names_display, annot_kws={"size": 8})
            plt.title(f"Confusion Matrix - {experiment_run_name}", fontsize=10);
            plt.xlabel("Predicted");
            plt.ylabel("True")
            plt.xticks(rotation=45, ha="right", fontsize=8);
            plt.yticks(rotation=0, fontsize=8);
            plt.tight_layout()
            cm_path = os.path.join(current_result_dir, "confusion_matrix_aggregated.png")
            plt.savefig(cm_path);
            plt.close()
            print(f"Aggregated confusion matrix saved to: {cm_path}")
    else:
        print("No valid LLM predictions for evaluation.")

    summary = {
        "experiment_run_name": experiment_run_name,
        "dataset": dataset_key,
        "model_name": current_config.model_name,
        "api_provider": current_config.api_provider,
        "fsl_setup": current_config.fsl_setup,
        "prompt_ablation_flags": {
            "system_instruction": current_config.prompt_include_system_instruction,
            "background_knowledge": current_config.prompt_include_background_knowledge,
            "candidate_list": current_config.prompt_include_candidate_list,
            "output_format": current_config.prompt_include_output_format,
        },
        "sc_extraction_config_used": current_config.sc_extraction_config,
        "sc_encoding_config_used": current_config.sc_encoding_config,
        "limit_meta_test_pool_size": current_config.limit_test_samples if current_config.limit_test_samples is not None else "None",
        "llm_caller_params": current_config.llm_caller_params,
        "results_summary": {
            "total_fsl_tasks_generated": len(fsl_tasks),
            "total_queries_expected_in_tasks": expected_total_queries,
            "total_queries_actually_processed_by_llm": total_queries_processed_in_experiment,
            "valid_predictions_from_llm": valid_preds_count,
            "total_query_instances_evaluated": len(all_query_true_labels_original),
            "accuracy": acc_val,
            "f1_macro": f1_macro_val
        },
        "full_classification_report_dict": report_dict
    }
    summary_filename = f"summary_{experiment_run_name}.json"
    summary_path = os.path.join(current_result_dir, summary_filename)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"FSL experiment summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run HRRP FSL experiments with LLMs, including ablations.")
    # --- Core LLM and API args ---
    parser.add_argument("--model_name", type=str, required=True, help="Name of the LLM to use.")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the LLM provider.")
    parser.add_argument("--api_provider", type=str, required=True,
                        choices=["openai", "anthropic", "google", "deepseek_platform", "zhipuai_glm"],
                        help="API provider.")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Base URL for OpenAI-compatible APIs or Anthropic proxy.")
    parser.add_argument("--google_api_endpoint", type=str, default=None, help="Specific API endpoint for Google GenAI.")

    # --- Dataset and Experiment Tag ---
    parser.add_argument("--dataset_key", type=str, default=DEFAULT_DATASET_KEY, help="Dataset key.")
    parser.add_argument("--experiment_tag", type=str, default="", help="Custom tag for results.")

    # --- FSL Parameters ---
    parser.add_argument("--n_way", type=int, default=DEFAULT_FSL_TASK_SETUP["n_way"])
    parser.add_argument("--k_shot_support", type=int, default=DEFAULT_FSL_TASK_SETUP["k_shot_support"])
    parser.add_argument("--q_shot_query", type=int, default=DEFAULT_FSL_TASK_SETUP["q_shot_query"])
    parser.add_argument("--num_fsl_tasks", type=int, default=DEFAULT_FSL_TASK_SETUP["num_fsl_tasks"])

    # --- LLM Caller Parameters ---
    parser.add_argument("--temperature", type=float, default=DEFAULT_LLM_CALLER_PARAMS["temperature"])
    parser.add_argument("--max_tokens_completion", type=int, default=DEFAULT_LLM_CALLER_PARAMS["max_tokens_completion"])

    # --- Data and Preprocessing ---
    parser.add_argument("--limit_test_samples", type=int, default=None,
                        help="Limit meta-test pool size. Overrides DEFAULT_LIMIT_TEST_SAMPLES if provided.")  # Allow None from CLI
    parser.add_argument("--force_data_preprocessing", action='store_true',
                        help="Force data preprocessing and SC extraction even if files exist.")
    parser.add_argument("--skip_svm_baseline", action='store_true', help="Skip SVM baseline.")

    # --- Ablation: Prompt Components ---
    parser.add_argument("--prompt_no_system_instruction", action='store_true',
                        help="Ablation: Do not include system instruction/role definition.")
    parser.add_argument("--prompt_no_background_knowledge", action='store_true',
                        help="Ablation: Do not include SC background knowledge.")
    parser.add_argument("--prompt_no_candidate_list", action='store_true',
                        help="Ablation: Do not include candidate class list (open-ended).")
    parser.add_argument("--prompt_no_output_format", action='store_true',
                        help="Ablation: Do not include output format instructions.")

    # --- Ablation: SC Quality ---
    parser.add_argument("--sc_max_centers_to_keep", type=int, default=None,
                        help="Ablation: Override max_centers_to_keep for SC extraction.")
    parser.add_argument("--sc_encoding_precision_amp", type=int, default=None,
                        help="Ablation: Override SC amplitude encoding precision.")
    # Precision_pos ablation can be added similarly if needed

    args = parser.parse_args()

    # Construct current_config object
    class CurrentRunConfig:
        def __init__(self, cli_args):
            self.dataset_key = cli_args.dataset_key
            self.AVAILABLE_DATASETS = AVAILABLE_DATASETS
            self.TARGET_HRRP_LENGTH = TARGET_HRRP_LENGTH
            # PREPROCESS_MAT_TO_NPY: True if force_data_preprocessing, else use default.
            # This needs careful handling for SC quality ablations. If SC params change, we *must* re-extract.
            self.PREPROCESS_MAT_TO_NPY = True if cli_args.force_data_preprocessing or \
                                                 cli_args.sc_max_centers_to_keep is not None or \
                                                 cli_args.sc_encoding_precision_amp is not None \
                else PREPROCESS_MAT_TO_NPY

            self.PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
            self.TEST_SPLIT_SIZE = TEST_SPLIT_SIZE
            self.RANDOM_STATE = RANDOM_STATE

            # SC Extraction Config - prioritize CLI, then default
            self.sc_extraction_config = {**DEFAULT_SC_EXTRACTION_CONFIG}  # Start with a copy of default
            if cli_args.sc_max_centers_to_keep is not None:
                self.sc_extraction_config["max_centers_to_keep"] = cli_args.sc_max_centers_to_keep

            # SC Encoding Config - prioritize CLI, then default
            self.sc_encoding_config = {**DEFAULT_SC_ENCODING_CONFIG}  # Start with a copy
            if cli_args.sc_encoding_precision_amp is not None:
                self.sc_encoding_config["precision_amp"] = cli_args.sc_encoding_precision_amp
            # Ensure TARGET_HRRP_LENGTH_INFO is in the final encoding config
            self.sc_encoding_config["TARGET_HRRP_LENGTH_INFO"] = self.TARGET_HRRP_LENGTH

            self.fsl_setup = {
                "enabled": True,
                "n_way": cli_args.n_way, "k_shot_support": cli_args.k_shot_support,
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

            self.RESULTS_BASE_DIR = RESULTS_BASE_DIR
            self.RUN_BASELINE_SVM = RUN_BASELINE_SVM if not cli_args.skip_svm_baseline else False
            self.BASELINE_SVM_PARAMS = BASELINE_SVM_PARAMS
            self.experiment_tag = cli_args.experiment_tag

            # Store prompt ablation flags
            self.prompt_include_system_instruction = not cli_args.prompt_no_system_instruction
            self.prompt_include_background_knowledge = not cli_args.prompt_no_background_knowledge
            self.prompt_include_candidate_list = not cli_args.prompt_no_candidate_list
            self.prompt_include_output_format = not cli_args.prompt_no_output_format

    config_obj = CurrentRunConfig(args)

    print(f"Starting HRRP FSL Experiment with LLM: {config_obj.model_name}");
    start_time_total = datetime.now()

    # Data preprocessing logic based on config_obj.PREPROCESS_MAT_TO_NPY
    # This is crucial: if SC parameters for ablation changed, PREPROCESS_MAT_TO_NPY must be true.
    if config_obj.PREPROCESS_MAT_TO_NPY:
        print(
            "\n--- Step 0: Preparing/Re-preparing .npy HRRP data and SC .pkl files (due to CLI flags or defaults) ---")
        # prepare_npy_data_and_scattering_centers now uses sc_extraction_config from config_obj
        prepare_npy_data_and_scattering_centers(config_obj)
    else:
        print(
            "\n--- Step 0: Skipping data preprocessing and SC extraction as per config/CLI. Using existing files. ---")

    if config_obj.sc_extraction_config["enabled"] and config_obj.fsl_setup["enabled"]:
        run_fsl_experiment_main(config_obj)

    if config_obj.RUN_BASELINE_SVM:
        try:
            print(f"\n--- Running SVM Baseline for dataset '{config_obj.dataset_key}' ---")
            run_svm_baseline_for_dataset(config_obj.dataset_key, config_obj)
        except Exception as e:
            print(f"Error running SVM baseline for {config_obj.dataset_key}: {e}")

    print(f"\nAll processes for this run finished. Total time: {datetime.now() - start_time_total}")
    print(f"Results saved in subdirectories under '{config_obj.RESULTS_BASE_DIR}'")


if __name__ == "__main__":
    if not os.path.exists(RESULTS_BASE_DIR): os.makedirs(RESULTS_BASE_DIR)
    main()