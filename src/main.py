import os
import numpy as np
import json
from datetime import datetime

from config import (
    SIMULATED_DATA_PATH, MEASURED_DATA_PATH, TARGET_HRRP_LENGTH,
    PREPROCESSING, LLM_PARAMS, CLASSIFIER_PARAMS, VISUALIZATION,
    RESULTS_BASE_PATH, OPENAI_API_KEY,
    RUN_SIMULATED_DATA, RUN_MEASURED_DATA,
    RUN_BASELINE_EXPERIMENT, BASELINE_CLASSIFIER_PARAMS
)
from data_loader import load_hrrp_data
from preprocessor import preprocess_hrrps_for_llm, preprocess_hrrps_for_baseline
from llm_embedder import LLMEmbedder
from classifiers import train_and_evaluate_classifier
from visualizer import plot_tsne_visualization

def run_experiment_on_dataset(dataset_name, hrrp_samples, labels, config_snapshot):
    """
    在给定的数据集上运行完整的LLM特征提取和分类实验。
    """
    print(f"\n{'='*20} 在数据集 {dataset_name} 上运行实验 {'='*20}")
    
    # 1. HRRP 预处理和文本转换
    text_sequences = preprocess_hrrps_for_llm(
        hrrp_samples,
        normalize_enabled=config_snapshot['PREPROCESSING']['normalize'],
        normalization_type=config_snapshot['PREPROCESSING']['normalization_type'],
        precision=config_snapshot['PREPROCESSING']['precision'],
        use_space_separator=config_snapshot['PREPROCESSING']['use_space_separator'],
        value_separator=config_snapshot['PREPROCESSING']['value_separator']
    )

    # 2. LLM 特征提取
    if not text_sequences:
        print(f"数据集 {dataset_name} 没有有效的文本序列，跳过LLM嵌入和分类。")
        return

    # 检查API密钥是否有效
    if config_snapshot['LLM_PARAMS']['provider'] == "openai" and \
       (config_snapshot['OPENAI_API_KEY'] is None or config_snapshot['OPENAI_API_KEY'] == "YOUR_API_KEY_HERE"):
        print(f"警告: OpenAI API 密钥未配置或无效。跳过 {dataset_name} 的LLM特征提取。")
        llm_embeddings = np.array([]) # 返回空数组
    else:
        embedder = LLMEmbedder(
            provider=config_snapshot['LLM_PARAMS']['provider'],
            model_name=config_snapshot['LLM_PARAMS']['model_name'],
            api_key=config_snapshot['OPENAI_API_KEY'] if config_snapshot['LLM_PARAMS']['provider'] == "openai" else None,
            batch_size=config_snapshot['LLM_PARAMS']['batch_size']
        )
        llm_embeddings = embedder.get_embeddings(text_sequences)

    if llm_embeddings.size == 0:
        print(f"数据集 {dataset_name} 未能获取LLM嵌入，跳过分类和可视化。")
        return

    # 保存嵌入 (可选)
    exp_name_suffix = f"{'norm' if config_snapshot['PREPROCESSING']['normalize'] else 'raw'}_"
    exp_name_suffix += f"{'space' if config_snapshot['PREPROCESSING']['use_space_separator'] else 'nospace'}_"
    exp_name_suffix += f"{config_snapshot['LLM_PARAMS']['model_name'].split('/')[-1].replace('-', '_')}_" # 简化模型名
    exp_name_suffix += config_snapshot['CLASSIFIER_PARAMS']['type']
    
    embeddings_save_dir = os.path.join(config_snapshot['RESULTS_BASE_PATH'], dataset_name)
    os.makedirs(embeddings_save_dir, exist_ok=True)
    embeddings_filename = os.path.join(embeddings_save_dir, f"llm_embeddings_{exp_name_suffix}.npy")
    np.save(embeddings_filename, llm_embeddings)
    print(f"LLM嵌入已保存到: {embeddings_filename}")

    # 3. 下游分类器训练与评估
    model, accuracy, f1_macro, report_dict, label_encoder = train_and_evaluate_classifier(
        llm_embeddings,
        labels,
        classifier_type=config_snapshot['CLASSIFIER_PARAMS']['type'],
        test_size=config_snapshot['CLASSIFIER_PARAMS']['test_size'],
        random_state=config_snapshot['CLASSIFIER_PARAMS']['random_state'],
        scale_features=True, # LLM嵌入通常也需要标准化
        knn_neighbors=config_snapshot['CLASSIFIER_PARAMS']['knn_neighbors'],
        svm_kernel=config_snapshot['CLASSIFIER_PARAMS']['svm_kernel'],
        svm_c=config_snapshot['CLASSIFIER_PARAMS']['svm_c']
    )
    
    results_summary = {
        "dataset": dataset_name,
        "experiment_name_suffix": exp_name_suffix,
        "llm_model": config_snapshot['LLM_PARAMS']['model_name'],
        "classifier": config_snapshot['CLASSIFIER_PARAMS']['type'],
        "preprocessing_normalize": config_snapshot['PREPROCESSING']['normalize'],
        "preprocessing_space_separator": config_snapshot['PREPROCESSING']['use_space_separator'],
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "classification_report": report_dict,
        "num_samples_total": len(labels),
        "num_features_llm": llm_embeddings.shape[1] if llm_embeddings.ndim > 1 else 0,
        "config_snapshot": config_snapshot # 保存当时的配置快照
    }
    
    results_filename = os.path.join(embeddings_save_dir, f"results_{exp_name_suffix}.json")
    with open(results_filename, 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"LLM实验结果已保存到: {results_filename}")


    # 4. t-SNE 可视化
    if label_encoder and llm_embeddings.size > 0 : # 确保label_encoder已创建且有嵌入
        plot_tsne_visualization(
            llm_embeddings,
            labels, # 传递原始文本标签
            dataset_name,
            f"llm_{exp_name_suffix}",
            config_snapshot['RESULTS_BASE_PATH'],
            label_encoder=label_encoder, # 传递训练好的encoder
            perplexity=config_snapshot['VISUALIZATION']['tsne_perplexity'],
            n_iter=config_snapshot['VISUALIZATION']['tsne_n_iter'],
            random_state=config_snapshot['CLASSIFIER_PARAMS']['random_state']
        )
    else:
        print("跳过t-SNE可视化，因为label_encoder未初始化或没有LLM嵌入。")


def run_baseline_experiment(dataset_name, hrrp_samples, labels, config_snapshot):
    """
    在给定的数据集上运行基于原始HRRP（可能归一化）的基线分类器实验。
    """
    print(f"\n{'='*20} 在数据集 {dataset_name} 上运行基线实验 {'='*20}")

    # 1. HRRP 预处理 (仅归一化)
    baseline_features = preprocess_hrrps_for_baseline(
        hrrp_samples,
        normalize_enabled=config_snapshot['PREPROCESSING']['normalize'], # 使用与LLM实验相同的归一化设置
        normalization_type=config_snapshot['PREPROCESSING']['normalization_type']
    )

    if baseline_features.size == 0:
        print(f"数据集 {dataset_name} 未能生成基线特征，跳过基线分类。")
        return

    # 2. 基线分类器训练与评估
    model, accuracy, f1_macro, report_dict, label_encoder = train_and_evaluate_classifier(
        baseline_features,
        labels,
        classifier_type=config_snapshot['BASELINE_CLASSIFIER_PARAMS']['type'],
        test_size=config_snapshot['CLASSIFIER_PARAMS']['test_size'],
        random_state=config_snapshot['CLASSIFIER_PARAMS']['random_state'],
        scale_features=True, # 原始HRRP数据通常也需要标准化
        svm_kernel=config_snapshot['BASELINE_CLASSIFIER_PARAMS']['svm_kernel'],
        svm_c=config_snapshot['BASELINE_CLASSIFIER_PARAMS']['svm_c']
    )

    exp_name_suffix = f"baseline_{'norm' if config_snapshot['PREPROCESSING']['normalize'] else 'raw'}_"
    exp_name_suffix += config_snapshot['BASELINE_CLASSIFIER_PARAMS']['type']

    results_summary = {
        "dataset": dataset_name,
        "experiment_name_suffix": exp_name_suffix,
        "classifier": config_snapshot['BASELINE_CLASSIFIER_PARAMS']['type'],
        "preprocessing_normalize_for_baseline": config_snapshot['PREPROCESSING']['normalize'],
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "classification_report": report_dict,
        "num_samples_total": len(labels),
        "num_features_baseline": baseline_features.shape[1] if baseline_features.ndim > 1 else 0,
        "config_snapshot": {
            "BASELINE_CLASSIFIER_PARAMS": config_snapshot['BASELINE_CLASSIFIER_PARAMS'],
            "PREPROCESSING_normalize_for_baseline": config_snapshot['PREPROCESSING']['normalize'],
            "CLASSIFIER_PARAMS_test_size": config_snapshot['CLASSIFIER_PARAMS']['test_size'],
            "CLASSIFIER_PARAMS_random_state": config_snapshot['CLASSIFIER_PARAMS']['random_state'],
        }
    }
    
    results_save_dir = os.path.join(config_snapshot['RESULTS_BASE_PATH'], dataset_name)
    os.makedirs(results_save_dir, exist_ok=True)
    results_filename = os.path.join(results_save_dir, f"results_{exp_name_suffix}.json")
    with open(results_filename, 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"基线实验结果已保存到: {results_filename}")

    # 可选: 基线特征的t-SNE可视化
    if label_encoder and baseline_features.size > 0:
        plot_tsne_visualization(
            baseline_features,
            labels,
            dataset_name,
            exp_name_suffix, # 使用基线实验的名称后缀
            config_snapshot['RESULTS_BASE_PATH'],
            label_encoder=label_encoder,
            perplexity=config_snapshot['VISUALIZATION']['tsne_perplexity'],
            n_iter=config_snapshot['VISUALIZATION']['tsne_n_iter'],
            random_state=config_snapshot['CLASSIFIER_PARAMS']['random_state']
        )


def main():
    print("开始HRRP目标识别实验...")
    start_time = datetime.now()

    # 创建一个当前配置的快照，传递给实验函数，确保一致性
    current_config_snapshot = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "LLM_PARAMS": LLM_PARAMS.copy(),
        "PREPROCESSING": PREPROCESSING.copy(),
        "TARGET_HRRP_LENGTH": TARGET_HRRP_LENGTH,
        "CLASSIFIER_PARAMS": CLASSIFIER_PARAMS.copy(),
        "BASELINE_CLASSIFIER_PARAMS": BASELINE_CLASSIFIER_PARAMS.copy(),
        "VISUALIZATION": VISUALIZATION.copy(),
        "RESULTS_BASE_PATH": RESULTS_BASE_PATH,
    }
    
    # --- 处理仿真数据集 ---
    if RUN_SIMULATED_DATA:
        sim_hrrps, sim_labels, _ = load_hrrp_data(
            SIMULATED_DATA_PATH, 'CoHH', 
            expected_length=1000, 
            target_length=TARGET_HRRP_LENGTH
        )
        if sim_hrrps:
            run_experiment_on_dataset("simulated", sim_hrrps, sim_labels, current_config_snapshot)
            if RUN_BASELINE_EXPERIMENT:
                run_baseline_experiment("simulated", sim_hrrps, sim_labels, current_config_snapshot)
        else:
            print("未加载到仿真数据，跳过仿真数据实验。")

    # --- 处理实测数据集 ---
    if RUN_MEASURED_DATA:
        meas_hrrps, meas_labels, _ = load_hrrp_data(
            MEASURED_DATA_PATH, 'data', 
            expected_length=500, 
            target_length=TARGET_HRRP_LENGTH
        )
        if meas_hrrps:
            run_experiment_on_dataset("measured", meas_hrrps, meas_labels, current_config_snapshot)
            if RUN_BASELINE_EXPERIMENT:
                run_baseline_experiment("measured", meas_hrrps, meas_labels, current_config_snapshot)
        else:
            print("未加载到实测数据，跳过实测数据实验。")

    end_time = datetime.now()
    print(f"\n所有实验完成。总耗时: {end_time - start_time}")
    print(f"结果和图像保存在 '{RESULTS_BASE_PATH}' 目录中。")
    print("请检查 src/config.py 文件以进行不同的消融实验配置。")

if __name__ == '__main__':
    # 确保结果目录存在
    os.makedirs(RESULTS_BASE_PATH, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_BASE_PATH, "simulated"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_BASE_PATH, "measured"), exist_ok=True)
    
    main()