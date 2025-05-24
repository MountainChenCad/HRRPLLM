import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

def plot_tsne_visualization(features, 
                            labels, 
                            dataset_name, 
                            experiment_name, 
                            results_path,
                            label_encoder=None,
                            perplexity=30, 
                            n_iter=1000, 
                            random_state=42):
    """
    使用t-SNE对特征进行降维可视化并保存图像。
    """
    if features is None or features.shape[0] < 2 : # t-SNE至少需要2个样本
        print("特征不足，无法进行t-SNE可视化。")
        return

    print(f"正在为 {dataset_name} ({experiment_name}) 生成t-SNE可视化...")
    
    if label_encoder is None:
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        target_names = le.classes_
    else:
        encoded_labels = label_encoder.transform(labels) # 假设labels已经是原始文本标签
        target_names = label_encoder.classes_

    # 确保 perplexity 小于样本数
    actual_perplexity = min(perplexity, features.shape[0] - 1)
    if actual_perplexity <=0: # 如果样本数太少，甚至无法满足perplexity=1
        print(f"样本数量 ({features.shape[0]}) 过少，无法进行有效的t-SNE (perplexity={actual_perplexity}).")
        return

    tsne = TSNE(n_components=2, 
                perplexity=actual_perplexity, 
                n_iter=n_iter, 
                random_state=random_state,
                init='pca', # PCA初始化通常更稳定
                learning_rate='auto'
                )
    
    try:
        embeddings_2d = tsne.fit_transform(features)
    except Exception as e:
        print(f"t-SNE计算失败: {e}")
        return

    plt.figure(figsize=(12, 10))
    
    unique_encoded_labels = np.unique(encoded_labels)
    
    # 使用matplotlib默认颜色循环
    colors = plt.cm.get_cmap('viridis', len(unique_encoded_labels))

    for i, encoded_label_val in enumerate(unique_encoded_labels):
        idx = (encoded_labels == encoded_label_val)
        original_label_name = target_names[encoded_label_val] # 获取原始类别名
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], 
                    color=colors(i / len(unique_encoded_labels)), # 使用颜色映射
                    label=original_label_name, alpha=0.7, s=50)

    plt.title(f't-SNE Visualization of HRRP LLM Embeddings\nDataset: {dataset_name} - Exp: {experiment_name}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(loc='best', markerscale=1.5, fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_dir = os.path.join(results_path, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    save_filepath = os.path.join(save_dir, f"{experiment_name}_tsne.png")
    plt.savefig(save_filepath)
    plt.close()
    print(f"t-SNE图像已保存到: {save_filepath}")

if __name__ == '__main__':
    from config import RESULTS_BASE_PATH, VISUALIZATION
    # 简单测试
    rng = np.random.RandomState(42)
    test_features_viz = rng.rand(50, 128) # 50个样本, 128维特征
    test_labels_viz_list = ['ClassA'] * 20 + ['ClassB'] * 15 + ['ClassC'] * 15
    
    le_viz = LabelEncoder()
    le_viz.fit(test_labels_viz_list)

    plot_tsne_visualization(test_features_viz, 
                            test_labels_viz_list, 
                            "test_dataset", 
                            "test_experiment_llm_knn",
                            RESULTS_BASE_PATH,
                            label_encoder=le_viz,
                            perplexity=VISUALIZATION['tsne_perplexity'],
                            n_iter=VISUALIZATION['tsne_n_iter'])
    print("t-SNE可视化测试完成。请检查 'results/test_dataset/' 目录。")