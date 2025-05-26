import os
import numpy as np
import json
from datetime import datetime
import random
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import (
    AVAILABLE_DATASETS, TARGET_HRRP_LENGTH, PROCESSED_DATA_DIR, TEST_SPLIT_SIZE, RANDOM_STATE,
    SCATTERING_CENTER_EXTRACTION, SCATTERING_CENTER_ENCODING, 
    FSL_TASK_SETUP,
    OPENAI_API_KEY, OPENAI_PROXY_BASE_URL, LLM_CALLER_PARAMS, LIMIT_TEST_SAMPLES,
    RESULTS_BASE_DIR, RUN_BASELINE_SVM, BASELINE_SVM_PARAMS, PREPROCESS_MAT_TO_NPY,
)
from data_utils import prepare_npy_data_and_scattering_centers, load_processed_data, build_fsl_tasks
from scattering_center_encoder import encode_all_sc_sets_to_text
from gpt_caller import GPTCaller
from prompt_constructor_sc import PromptConstructorSC
from baseline_evaluator import run_svm_baseline_for_dataset

def parse_llm_output_for_label(llm_response_text, class_names):
    if not llm_response_text: return None
    match = re.search(r"预测目标类别：\s*([^\n`]+)`?", llm_response_text, re.IGNORECASE) 
    if match:
        extracted_name = match.group(1).strip()
        for cn in class_names:
            if cn.lower() == extracted_name.lower(): return cn
    
    processed_response = llm_response_text.lower()
    chars_to_remove = ['.',',',':','"','\'','：','`','[',']','『','』','【','】','(',')','（','）'];
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


def run_fsl_experiment_main(dataset_name_key, config):
    print(f"\n{'='*30}\n处理数据集: {dataset_name_key} (FSL Proto N-way K-shot Q-query)\n{'='*30}")

    load_result = load_processed_data(dataset_name_key, config, load_scattering_centers=True)
    if load_result[0] is None and load_result[2] is None : # If both train and test HRRP are None
        print(f"数据集 '{dataset_name_key}' 加载失败或不完整 (无训练和测试HRRP)。跳过FSL实验。"); return
    
    _, _, X_meta_test_hrrp, y_meta_test_original, \
    _, X_meta_test_sc_list, label_encoder, class_names_all_dataset = load_result


    if not config.SCATTERING_CENTER_EXTRACTION["enabled"]:
        print(f"SC提取未启用 for '{dataset_name_key}'，跳过FSL(SC)实验。"); return
    
    if X_meta_test_hrrp is None or X_meta_test_hrrp.size == 0:
        print(f"元测试数据为空 for '{dataset_name_key}'，无法构建FSL评估任务。跳过。"); return
    
    # SCs are needed for prototypes if k_shot_support > 0
    if config.FSL_TASK_SETUP["k_shot_support"] > 0 and \
       (X_meta_test_sc_list is None or (isinstance(X_meta_test_sc_list, list) and not X_meta_test_sc_list and X_meta_test_hrrp.size > 0) ):
        print(f"SC数据 for 元测试集 of '{dataset_name_key}' 未完整加载或为空，但K>0。原型计算可能受影响或失败。"); 
        # build_fsl_tasks will try to handle this by creating empty SC lists for prototypes if needed.

    data_pool_hrrp = X_meta_test_hrrp
    data_pool_original_labels = y_meta_test_original
    data_pool_sc_list = X_meta_test_sc_list
    
    if config.LIMIT_TEST_SAMPLES is not None and config.LIMIT_TEST_SAMPLES > 0 and \
       len(X_meta_test_hrrp) > 0 and config.LIMIT_TEST_SAMPLES < len(X_meta_test_hrrp): # Add check for X_meta_test_hrrp not empty
        num_available = len(X_meta_test_hrrp)
        num_to_sample = min(config.LIMIT_TEST_SAMPLES, num_available)
        print(f"元测试数据池将从 {num_available} 个样本中抽样限制为: {num_to_sample} 个。FSL任务将从此池构建。")
        
        np.random.seed(config.RANDOM_STATE) 
        indices = np.arange(num_available)
        sampled_indices = np.random.choice(indices, num_to_sample, replace=False)
            
        data_pool_hrrp = X_meta_test_hrrp[sampled_indices]
        data_pool_original_labels = y_meta_test_original[sampled_indices]
        if X_meta_test_sc_list: 
             data_pool_sc_list = [X_meta_test_sc_list[i] for i in sampled_indices]
        else: 
             data_pool_sc_list = None 
        print(f"  限制后的数据池大小: {len(data_pool_hrrp)}")

    elif config.LIMIT_TEST_SAMPLES is not None and config.LIMIT_TEST_SAMPLES <=0:
        print(f"LIMIT_TEST_SAMPLES ({config.LIMIT_TEST_SAMPLES}) <= 0, 数据池将为空。FSL任务无法构建。")
        data_pool_hrrp = np.array([])
        data_pool_original_labels = np.array([])
        data_pool_sc_list = []

    fsl_tasks = build_fsl_tasks(
        data_pool_hrrp, data_pool_original_labels, data_pool_sc_list, 
        label_encoder, config.FSL_TASK_SETUP, 
        config.SCATTERING_CENTER_ENCODING, 
        config.SCATTERING_CENTER_EXTRACTION, # MODIFIED: Pass SC_EXTRACTION for max_centers_to_keep
        config.RANDOM_STATE
    )
    if not fsl_tasks:
        print(f"数据集 '{dataset_name_key}' 未能构建FSL任务。跳过实验。"); return

    gpt_caller = GPTCaller(**config.LLM_CALLER_PARAMS, api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_PROXY_BASE_URL)
    
    all_query_true_labels_original = []
    all_query_pred_llm_original_labels = []
    
    fsl_cfg = config.FSL_TASK_SETUP
    exp_name_parts = [
        "FSL_SC_ProtoEval", # MODIFIED: Indicate prototype
        f"{fsl_cfg['n_way']}way",
        f"{fsl_cfg['k_shot_support']}Ksup",
        f"{fsl_cfg['q_shot_query']}Qquery",
        f"{len(fsl_tasks)}tasks", 
        config.SCATTERING_CENTER_ENCODING["format"],
        config.LLM_CALLER_PARAMS["model_name"].replace("/", "_").replace(".", "_")
    ]
    experiment_run_name = "_".join(exp_name_parts)
    current_result_dir = os.path.join(config.RESULTS_BASE_DIR, dataset_name_key, experiment_run_name)
    os.makedirs(current_result_dir, exist_ok=True)
    
    prompts_responses_dir = os.path.join(current_result_dir, "prompts_and_responses")
    os.makedirs(prompts_responses_dir, exist_ok=True)
    print(f"Prompts和Responses将保存到: {prompts_responses_dir}")

    print(f"\n开始对 {len(fsl_tasks)} 个FSL任务进行LLM预测 (run: {experiment_run_name})...")
    
    total_queries_processed_in_experiment = 0
    for task_idx, task_data in enumerate(tqdm(fsl_tasks, desc=f"FSL任务处理 ({dataset_name_key})", leave=False, unit="task")):
        task_specific_class_names = list(task_data["task_classes"]) 
        prompt_constructor = PromptConstructorSC(dataset_name_key, task_specific_class_names, config.SCATTERING_CENTER_ENCODING)

        # MODIFIED: Use prototype SC texts and labels for the prompt
        prototype_sc_texts_for_prompt = task_data["support_prototypes_sc_texts"]
        prototype_labels_for_prompt = task_data["support_prototypes_labels"]
        
        few_shot_examples_for_prompt = []
        if fsl_cfg["k_shot_support"] > 0: # Only add if K > 0 (prototypes are for K>0)
            if len(prototype_sc_texts_for_prompt) == len(prototype_labels_for_prompt) and \
               len(prototype_sc_texts_for_prompt) == fsl_cfg['n_way'] : # Expect N prototypes
                for sc_text, label in zip(prototype_sc_texts_for_prompt, prototype_labels_for_prompt):
                    few_shot_examples_for_prompt.append((sc_text, label))
            else:
                 print(f"  警告: 任务 {task_idx+1} 配置为K>0 shot，但原型SC文本/标签数量 ({len(prototype_sc_texts_for_prompt)}/{fsl_cfg['n_way']}) 不正确。将尝试0-shot。")
                 # Fallback to 0-shot if prototype data is inconsistent
                 few_shot_examples_for_prompt = [] 


        task_query_sc_texts = task_data["query_sc_texts"]
        task_query_labels = task_data["query_labels"] 

        if not task_query_sc_texts or len(task_query_sc_texts) == 0:
            continue

        task_query_pred_labels = []
        for query_idx_in_task, query_sc_text_for_prompt in enumerate(task_query_sc_texts):
            prompt_str = prompt_constructor.construct_prompt_with_sc(query_sc_text_for_prompt, few_shot_examples_for_prompt)
            
            prompt_filename = f"prompt_task{task_idx}_query{query_idx_in_task}.txt"
            prompt_filepath = os.path.join(prompts_responses_dir, prompt_filename)
            try:
                with open(prompt_filepath, "w", encoding="utf-8") as pf:
                    pf.write(f"--- Task {task_idx}, Query {query_idx_in_task} ---\n")
                    pf.write(f"True Label (for this query): {task_query_labels[query_idx_in_task]}\n") 
                    pf.write(f"Task Classes: {task_specific_class_names}\n")
                    pf.write("--- Prompt Sent to LLM (using class prototypes): ---\n")
                    pf.write(prompt_str)
            except Exception as e: print(f"  错误: 无法保存prompt文件 {prompt_filepath}: {e}")

            llm_response = gpt_caller.get_completion(prompt_str)
            predicted_label = None

            response_filename = f"response_task{task_idx}_query{query_idx_in_task}.txt"
            response_filepath = os.path.join(prompts_responses_dir, response_filename)
            if llm_response:
                try:
                    with open(response_filepath, "w", encoding="utf-8") as rf: rf.write(llm_response)
                except Exception as e: print(f"  错误: 无法保存response文件 {response_filepath}: {e}")
                predicted_label = parse_llm_output_for_label(llm_response, task_specific_class_names) 
            else: 
                try:
                    with open(response_filepath, "w", encoding="utf-8") as rf: rf.write("LLM_RESPONSE_IS_NONE")
                except Exception as e: print(f"  错误: 无法保存空的response文件 {response_filepath}: {e}")

            task_query_pred_labels.append(predicted_label)
            total_queries_processed_in_experiment +=1

        all_query_true_labels_original.extend(task_query_labels) 
        all_query_pred_llm_original_labels.extend(task_query_pred_labels)

    valid_preds_count = 0; y_true_eval, y_pred_eval = [], []; failed_count = 0
    for true_l, pred_l in zip(all_query_true_labels_original, all_query_pred_llm_original_labels):
        if pred_l is not None: 
            y_true_eval.append(true_l); y_pred_eval.append(pred_l); valid_preds_count+=1
        else: failed_count+=1
    
    expected_total_queries = sum(len(t['query_labels']) for t in fsl_tasks)
    print(f"\nLLM预测(FSL_SC Proto)完成。总查询样本应处理: {expected_total_queries} (来自 {len(fsl_tasks)} 个任务)。实际处理: {total_queries_processed_in_experiment}。")
    print(f"有效预测: {valid_preds_count}/{len(all_query_true_labels_original)}。失败/无法解析: {failed_count}。")

    acc_val, f1_macro_val, report_dict = 0.0, 0.0, {}
    if valid_preds_count > 0:
        y_true_encoded = label_encoder.transform(y_true_eval) 
        y_pred_encoded = label_encoder.transform(y_pred_eval)
        
        acc_val = accuracy_score(y_true_encoded, y_pred_encoded)
        f1_macro_val = f1_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
        report_labels_encoded = label_encoder.transform(class_names_all_dataset)
        try:
            report_str = classification_report(y_true_encoded, y_pred_encoded, labels=report_labels_encoded, target_names=class_names_all_dataset, zero_division=0, digits=3)
            report_dict = classification_report(y_true_encoded, y_pred_encoded, labels=report_labels_encoded, target_names=class_names_all_dataset, output_dict=True, zero_division=0, digits=3)
        except Exception as e: 
             print(f"生成分类报告时出现异常: {e}. 尝试使用实际出现的标签。")
             present_labels_encoded = np.unique(np.concatenate((y_true_encoded, y_pred_encoded)))
             present_class_names = list(label_encoder.inverse_transform(present_labels_encoded))
             report_str = classification_report(y_true_encoded, y_pred_encoded, labels=present_labels_encoded, target_names=present_class_names, zero_division=0, digits=3)
             report_dict = classification_report(y_true_encoded, y_pred_encoded, labels=present_labels_encoded, target_names=present_class_names, output_dict=True, zero_division=0, digits=3)

        print("\nLLM FSL (SC Proto) 分类报告 (汇总所有任务查询集):\n", report_str)
        print(f"准确率 (Accuracy): {acc_val:.4f}, F1宏平均 (F1-macro): {f1_macro_val:.4f}")
        
        if valid_preds_count > 1 :
            cm_labels_encoded = label_encoder.transform(class_names_all_dataset)
            cm_class_names_display = class_names_all_dataset
            cm = confusion_matrix(y_true_encoded, y_pred_encoded, labels=cm_labels_encoded)
            plt.figure(figsize=(max(8,len(cm_class_names_display)*0.6), max(6,len(cm_class_names_display)*0.45)))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_class_names_display, yticklabels=cm_class_names_display, annot_kws={"size": 8})
            plt.title(f"混淆矩阵 - {experiment_run_name}", fontsize=10); plt.xlabel("预测类别", fontsize=9); plt.ylabel("真实类别", fontsize=9)
            plt.xticks(rotation=45, ha="right", fontsize=8); plt.yticks(rotation=0, fontsize=8); plt.tight_layout()
            plt.savefig(os.path.join(current_result_dir, "confusion_matrix_aggregated.png")); plt.close()
            print(f"混淆矩阵已保存到: {os.path.join(current_result_dir, 'confusion_matrix_aggregated.png')}")
    else: print("无有效LLM预测可供评估。")
    
    summary = {
        "dataset": dataset_name_key, "exp_type": "fsl_sc_prototype_eval", "exp_name": experiment_run_name, 
        "config":{
            "FSL_SETUP": config.FSL_TASK_SETUP, "SC_EXT": config.SCATTERING_CENTER_EXTRACTION, 
            "SC_ENC": config.SCATTERING_CENTER_ENCODING, "LLM": config.LLM_CALLER_PARAMS,
            "LIMIT_META_TEST_POOL_SIZE": config.LIMIT_TEST_SAMPLES if config.LIMIT_TEST_SAMPLES is not None else "None"
        }, 
        "results_summary": {
            "total_fsl_tasks_generated": len(fsl_tasks),
            "total_queries_expected_in_tasks": expected_total_queries,
            "total_queries_actually_processed_by_llm": total_queries_processed_in_experiment,
            "valid_predictions_from_llm": valid_preds_count, 
            "total_query_instances_evaluated": len(all_query_true_labels_original),
            "accuracy": acc_val, "f1_macro": f1_macro_val
        },
        "full_classification_report": report_dict 
    }
    summary_path = os.path.join(current_result_dir, "summary_fsl_aggregated.json")
    with open(summary_path, "w", encoding="utf-8") as f: json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"FSL原型实验总结保存到: {summary_path}")


class CurrentRunConfig:
    AVAILABLE_DATASETS=AVAILABLE_DATASETS; TARGET_HRRP_LENGTH=TARGET_HRRP_LENGTH; PREPROCESS_MAT_TO_NPY=PREPROCESS_MAT_TO_NPY
    PROCESSED_DATA_DIR=PROCESSED_DATA_DIR; TEST_SPLIT_SIZE=TEST_SPLIT_SIZE; RANDOM_STATE=RANDOM_STATE
    SCATTERING_CENTER_EXTRACTION=SCATTERING_CENTER_EXTRACTION; SCATTERING_CENTER_ENCODING=SCATTERING_CENTER_ENCODING
    FSL_TASK_SETUP=FSL_TASK_SETUP 
    OPENAI_API_KEY=OPENAI_API_KEY; OPENAI_PROXY_BASE_URL=OPENAI_PROXY_BASE_URL; LLM_CALLER_PARAMS=LLM_CALLER_PARAMS
    LIMIT_TEST_SAMPLES=LIMIT_TEST_SAMPLES; RESULTS_BASE_DIR=RESULTS_BASE_DIR
    RUN_BASELINE_SVM=RUN_BASELINE_SVM; BASELINE_SVM_PARAMS=BASELINE_SVM_PARAMS

def main():
    config_obj = CurrentRunConfig()
    print("开始HRRP目标识别实验 (FSL原型, N-way K-shot Q-query, 基于散射中心)..."); start_time_total = datetime.now()
    
    if config_obj.PREPROCESS_MAT_TO_NPY or \
       (config_obj.SCATTERING_CENTER_EXTRACTION["enabled"] and \
        not all(os.path.exists(os.path.join(config_obj.PROCESSED_DATA_DIR, dk, "X_train_scatter_centers.pkl")) and \
                os.path.exists(os.path.join(config_obj.PROCESSED_DATA_DIR, dk, "X_test_scatter_centers.pkl"))
                for dk in config_obj.AVAILABLE_DATASETS.keys() if config_obj.AVAILABLE_DATASETS[dk])): 
        print("\n--- 步骤0: 准备 .npy HRRP数据 和 散射中心 .pkl 文件 ---")
        prepare_npy_data_and_scattering_centers(config_obj)
    else:
        print("\n--- 步骤0: 跳过数据预处理和SC提取 (PREPROCESS_MAT_TO_NPY=False 且所需文件已存在) ---")
    
    for dset_key in config_obj.AVAILABLE_DATASETS.keys():
        if not config_obj.AVAILABLE_DATASETS[dset_key]: continue 
        print(f"\n>>>>>> 开始处理数据集: {dset_key} (FSL原型基于SC) <<<<<<")
        if config_obj.SCATTERING_CENTER_EXTRACTION["enabled"] and config_obj.FSL_TASK_SETUP["enabled"]: 
            run_fsl_experiment_main(dset_key, config_obj)
        elif not config_obj.SCATTERING_CENTER_EXTRACTION["enabled"]:
            print(f"跳过FSL(SC Proto)实验 for '{dset_key}'，因为散射中心提取已禁用。")
        elif not config_obj.FSL_TASK_SETUP["enabled"]: 
            print(f"跳过FSL(SC Proto)实验 for '{dset_key}'，因为FSL任务设置已禁用。")
        
        if config_obj.RUN_BASELINE_SVM:
            try: 
                print(f"\n--- 为数据集 '{dset_key}' 运行基线SVM ---")
                run_svm_baseline_for_dataset(dset_key, config_obj)
            except Exception as e: print(f"运行基线SVM时出错 ({dset_key}): {e}")
        print(f">>>>>> 数据集: {dset_key} 处理完毕 <<<<<<")
        
    print(f"\n所有实验流程完成。总耗时: {datetime.now() - start_time_total}")
    print(f"所有结果保存在 '{config_obj.RESULTS_BASE_DIR}' 目录下。")

if __name__ == "__main__":
    if not os.path.exists(RESULTS_BASE_DIR): os.makedirs(RESULTS_BASE_DIR)
    main()
