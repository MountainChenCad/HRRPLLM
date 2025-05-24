import os
import numpy as np
import json
from datetime import datetime
import random
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm

from config import (
    AVAILABLE_DATASETS, TARGET_HRRP_LENGTH, PROCESSED_DATA_DIR, TEST_SPLIT_SIZE, RANDOM_STATE,
    SCATTERING_CENTER_EXTRACTION, SCATTERING_CENTER_ENCODING, 
    FSL_TASK_SETUP, # 使用新的FSL配置
    OPENAI_API_KEY, OPENAI_PROXY_BASE_URL, LLM_CALLER_PARAMS, LIMIT_TEST_SAMPLES,
    RESULTS_BASE_DIR, RUN_BASELINE_SVM, BASELINE_SVM_PARAMS, PREPROCESS_MAT_TO_NPY,
)
from data_utils import prepare_npy_data_and_scattering_centers, load_processed_data
# build_task_support_set 不再需要
from scattering_center_encoder import encode_all_sc_sets_to_text
from dynamic_neighbor_selector import select_k_most_similar_from_support_pool # 新的函数
from gpt_caller import GPTCaller
from prompt_constructor_sc import PromptConstructorSC
from baseline_evaluator import run_svm_baseline_for_dataset
# feature_extractor 和 sc_set_to_feature_vector 主要在 dynamic_neighbor_selector 内部使用

def parse_llm_output_for_label(llm_response_text, class_names):
    # (与之前版本相同)
    if not llm_response_text: return None
    match = re.search(r"预测目标类别：\s*([^\n`]+)`?", llm_response_text, re.IGNORECASE) 
    if match:
        extracted_name = match.group(1).strip()
        for cn in class_names:
            if cn.lower() == extracted_name.lower(): return cn
        return None 
    processed_response = llm_response_text.lower()
    chars_to_remove = ['.',',',':','"','\'','：','`']; [processed_response := processed_response.replace(c,"") for c in chars_to_remove]
    sorted_class_names = sorted(class_names, key=len, reverse=True)
    for cn in sorted_class_names:
        processed_cn = cn.lower(); [processed_cn := processed_cn.replace(c,"") for c in chars_to_remove]
        if processed_cn == "": continue 
        if processed_cn in processed_response: return cn 
    return None

def run_fsl_experiment_main(dataset_name_key, config): # 重命名主实验函数
    print(f"\n{'='*30}\n处理数据集: {dataset_name_key} (FSL动态选择SC)\n{'='*30}")

    load_result = load_processed_data(dataset_name_key, config, load_scattering_centers=True)
    if load_result[0] is None: print(f"无法加载 {dataset_name_key} 数据。"); return
    
    X_meta_train_hrrp, y_meta_train_original, X_meta_test_hrrp, y_meta_test_original, \
    X_meta_train_sc_list, X_meta_test_sc_list, label_encoder, class_names = load_result

    if not config.SCATTERING_CENTER_EXTRACTION["enabled"] or \
       X_meta_train_sc_list is None or X_meta_test_sc_list is None or \
       not X_meta_train_sc_list or not X_meta_test_sc_list:
        print(f"SC数据 for '{dataset_name_key}' 未启用或为空，跳过FSL(SC)实验。"); return

    # --- 支撑集池（元训练集）的SC文本编码 ---
    # 这些文本将用于构建Prompt中的Few-Shot示例
    support_pool_sc_texts_for_prompt = encode_all_sc_sets_to_text(
        X_meta_train_sc_list, config.SCATTERING_CENTER_ENCODING
    )
    
    # --- (可选) 限制元测试集(查询集)的样本数量 ---
    current_X_query_hrrp = X_meta_test_hrrp
    current_X_query_sc_list = X_meta_test_sc_list
    current_y_query_original = y_meta_test_original
    
    if config.LIMIT_TEST_SAMPLES is not None and config.LIMIT_TEST_SAMPLES < len(X_meta_test_hrrp):
        num_available = len(X_meta_test_hrrp)
        num_to_sample = min(config.LIMIT_TEST_SAMPLES, num_available)
        print(f"限制查询样本为: {num_to_sample} (从 {num_available} 个)")
        indices = np.random.choice(num_available, num_to_sample, replace=False)
        current_X_query_hrrp = X_meta_test_hrrp[indices]
        current_X_query_sc_list = [X_meta_test_sc_list[i] for i in indices] if X_meta_test_sc_list else []
        current_y_query_original = y_meta_test_original[indices]

    current_query_sc_texts_for_prompt = encode_all_sc_sets_to_text(
        current_X_query_sc_list, config.SCATTERING_CENTER_ENCODING
    )

    gpt_caller = GPTCaller(**config.LLM_CALLER_PARAMS, api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_PROXY_BASE_URL)
    prompt_constructor = PromptConstructorSC(dataset_name_key, class_names, config.SCATTERING_CENTER_ENCODING)

    y_pred_llm_original_labels = []
    
    k_prompt_str = f"k{config.FSL_TASK_SETUP['k_shots_for_prompt_from_task_support']}Prompt"
    # s_support_str = f"s{config.FSL_TASK_SETUP['samples_per_class_in_task_support_set']}TaskSupport" # 这个配置项已移除
    sim_metric_str = config.FSL_TASK_SETUP["similarity_metric"].replace("_on_", "On")
    sc_format_str = config.SCATTERING_CENTER_ENCODING["format"]
    llm_model_str = config.LLM_CALLER_PARAMS["model_name"].replace("/", "_").replace(".","_")
    experiment_run_name = f"FSL_SC_dynamicSupport_{sim_metric_str}_{k_prompt_str}_{sc_format_str}_{llm_model_str}"
    current_result_dir = os.path.join(config.RESULTS_BASE_DIR, dataset_name_key, experiment_run_name)
    os.makedirs(current_result_dir, exist_ok=True)

    print(f"\n开始对 {len(current_X_query_hrrp)} 个查询样本进行LLM预测 (run: {experiment_run_name})...")
    
    for i in tqdm(range(len(current_X_query_hrrp)), desc=f"LLM预测 ({dataset_name_key}_FSL_SC)", leave=False):
        query_hrrp_sample = current_X_query_hrrp[i]
        query_sc_list_sample = current_X_query_sc_list[i] if current_X_query_sc_list else None
        query_sc_text_for_prompt = current_query_sc_texts_for_prompt[i]
        
        few_shot_examples_for_this_query = []
        if config.FSL_TASK_SETUP["enabled"] and config.FSL_TASK_SETUP["k_shots_for_prompt_from_task_support"] > 0:
            few_shot_examples_for_this_query = select_k_most_similar_from_support_pool(
                query_hrrp_if_needed=query_hrrp_sample,
                query_sc_list_if_needed=query_sc_list_sample,
                support_pool_hrrps=X_meta_train_hrrp,         # 整个元训练HRRP池
                support_pool_sc_lists=X_meta_train_sc_list,  # 整个元训练SC池
                support_pool_labels_original=y_meta_train_original, # 元训练标签
                support_pool_sc_texts_for_prompt=support_pool_sc_texts_for_prompt, # 元训练SC文本
                total_k_shots_for_prompt=config.FSL_TASK_SETUP["k_shots_for_prompt_from_task_support"],
                similarity_metric=config.FSL_TASK_SETUP["similarity_metric"],
                sc_extraction_config=config.SCATTERING_CENTER_EXTRACTION,
                sc_feature_type_for_similarity=config.FSL_TASK_SETUP.get("sc_feature_type_for_similarity")
            )
        
        prompt_str = prompt_constructor.construct_prompt_with_sc(query_sc_text_for_prompt, few_shot_examples_for_this_query)
        
        file_idx_prefix = i 
        prompt_filename = f"prompt_queryIdx{file_idx_prefix}.txt"; response_filename = f"response_queryIdx{file_idx_prefix}.txt"
        with open(os.path.join(current_result_dir, prompt_filename), "w", encoding="utf-8") as pf: pf.write(prompt_str)
        llm_response = gpt_caller.get_completion(prompt_str)
        if llm_response:
            with open(os.path.join(current_result_dir, response_filename), "w", encoding="utf-8") as rf: rf.write(llm_response)
            predicted_label = parse_llm_output_for_label(llm_response, class_names)
            y_pred_llm_original_labels.append(predicted_label)
        else: y_pred_llm_original_labels.append(None)

    # --- 评估 ---
    # (评估逻辑与之前版本相同，确保使用 current_y_query_original 和 y_pred_llm_original_labels)
    valid_preds_count = 0; y_true_eval, y_pred_eval = [], []; failed_count = 0
    for i, (true_l, pred_l) in enumerate(zip(current_y_query_original, y_pred_llm_original_labels)): # 使用 current_y_query_original
        if pred_l is not None: y_true_eval.append(true_l); y_pred_eval.append(pred_l); valid_preds_count+=1
        else: failed_count+=1
    print(f"\nLLM预测(FSL_SC)完成。有效预测: {valid_preds_count}/{len(current_y_query_original)}。失败/无法解析: {failed_count}。")

    acc, f1, report = 0.0, 0.0, {}
    if valid_preds_count > 0:
        y_true_e, y_pred_e = label_encoder.transform(y_true_eval), label_encoder.transform(y_pred_eval)
        acc = accuracy_score(y_true_e,y_pred_e); f1 = f1_score(y_true_e,y_pred_e,average='macro',zero_division=0)
        report_labels_e = label_encoder.transform(label_encoder.classes_)
        try:
            report_s = classification_report(y_true_e,y_pred_e,labels=report_labels_e,target_names=label_encoder.classes_,zero_division=0)
            report = classification_report(y_true_e,y_pred_e,labels=report_labels_e,target_names=label_encoder.classes_,output_dict=True,zero_division=0)
        except Exception as e: report_s=classification_report(y_true_e,y_pred_e,zero_division=0); report=classification_report(y_true_e,y_pred_e,output_dict=True,zero_division=0); print(f"报告出错: {e}")
        print("\nLLM FSL (SC) 分类报告:\n", report_s); print(f"准确率: {acc:.4f}, F1: {f1_macro:.4f}")
        if valid_preds_count > 1 :
            cm=confusion_matrix(y_true_e,y_pred_e,labels=label_encoder.transform(label_encoder.classes_))
            plt.figure(figsize=(max(8,len(class_names)),max(6,int(len(class_names)*0.8)))); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=label_encoder.classes_,yticklabels=label_encoder.classes_)
            plt.title(f"混淆矩阵 - {experiment_run_name}",fontsize=10); plt.xlabel("预测"); plt.ylabel("真实"); plt.xticks(rotation=45,ha="right"); plt.yticks(rotation=0); plt.tight_layout()
            plt.savefig(os.path.join(current_result_dir, "confusion_matrix.png")); plt.close()
    else: print("无有效LLM预测。")
    
    summary = {"dataset":dataset_name_key, "exp_type":"fsl_sc_dynamic_support_selection", "exp_name":experiment_run_name, "config":{"FSL_SETUP":config.FSL_TASK_SETUP, "SC_EXT":config.SCATTERING_CENTER_EXTRACTION, "SC_ENC":config.SCATTERING_CENTER_ENCODING, "LLM":config.LLM_CALLER_PARAMS}, "valid_preds":valid_preds_count, "total_query":len(current_y_query_original), "accuracy":acc, "f1_macro":f1, "report":report}
    with open(os.path.join(current_result_dir, "summary.json"), "w", encoding="utf-8") as f: json.dump(summary,f,indent=4,ensure_ascii=False)
    print(f"FSL实验总结保存到: {os.path.join(current_result_dir, 'summary.json')}")


def main():
    class CurrentRunConfig: 
        AVAILABLE_DATASETS=AVAILABLE_DATASETS; TARGET_HRRP_LENGTH=TARGET_HRRP_LENGTH; PREPROCESS_MAT_TO_NPY=PREPROCESS_MAT_TO_NPY
        PROCESSED_DATA_DIR=PROCESSED_DATA_DIR; TEST_SPLIT_SIZE=TEST_SPLIT_SIZE; RANDOM_STATE=RANDOM_STATE
        SCATTERING_CENTER_EXTRACTION=SCATTERING_CENTER_EXTRACTION; SCATTERING_CENTER_ENCODING=SCATTERING_CENTER_ENCODING
        FSL_TASK_SETUP=FSL_TASK_SETUP 
        OPENAI_API_KEY=OPENAI_API_KEY; OPENAI_PROXY_BASE_URL=OPENAI_PROXY_BASE_URL; LLM_CALLER_PARAMS=LLM_CALLER_PARAMS
        LIMIT_TEST_SAMPLES=LIMIT_TEST_SAMPLES; RESULTS_BASE_DIR=RESULTS_BASE_DIR
        RUN_BASELINE_SVM=RUN_BASELINE_SVM; BASELINE_SVM_PARAMS=BASELINE_SVM_PARAMS
    config_obj = CurrentRunConfig()
    print("开始HRRP目标识别实验 (FSL任务，动态选择，基于散射中心)..."); start_time_total = datetime.now()
    
    if config_obj.PREPROCESS_MAT_TO_NPY or \
       (config_obj.SCATTERING_CENTER_EXTRACTION["enabled"] and \
        not all(os.path.exists(os.path.join(config_obj.PROCESSED_DATA_DIR, dk, "X_train_scatter_centers.pkl")) 
                for dk in config_obj.AVAILABLE_DATASETS.keys())):
        print("\n--- 步骤0: 准备 .npy HRRP数据 和 散射中心 .pkl 文件 ---")
        prepare_npy_data_and_scattering_centers(config_obj)
    
    for dset_key in config_obj.AVAILABLE_DATASETS.keys():
        print(f"\n>>>>>> 开始处理数据集: {dset_key} (FSL基于SC) <<<<<<")
        if config_obj.SCATTERING_CENTER_EXTRACTION["enabled"] and config_obj.FSL_TASK_SETUP["enabled"]: 
            run_fsl_experiment_main(dset_key, config_obj) # 调用新的主实验函数
        elif not config_obj.SCATTERING_CENTER_EXTRACTION["enabled"]:
            print(f"跳过FSL(SC)实验 for '{dset_key}'，因为散射中心提取已禁用。")
        elif not config_obj.FSL_TASK_SETUP["enabled"]: 
            print(f"跳过FSL(SC)实验 for '{dset_key}'，因为FSL任务设置已禁用。")
        
        if config_obj.RUN_BASELINE_SVM:
            try: run_svm_baseline_for_dataset(dset_key, config_obj)
            except Exception as e: print(f"运行基线SVM时出错 ({dset_key}): {e}")
        print(f">>>>>> 数据集: {dset_key} 处理完毕 <<<<<<")
    print(f"\n所有实验流程完成。总耗时: {datetime.now() - start_time_total}")
    print(f"所有结果保存在 '{config_obj.RESULTS_BASE_DIR}' 目录下。")

if __name__ == "__main__":
    if not os.path.exists(RESULTS_BASE_DIR): os.makedirs(RESULTS_BASE_DIR)
    main()