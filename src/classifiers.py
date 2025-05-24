from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

def train_and_evaluate_classifier(features, 
                                  labels, 
                                  classifier_type='knn', 
                                  test_size=0.3, 
                                  random_state=42, 
                                  scale_features=True,
                                  **kwargs):
    """
    训练并评估指定的分类器。

    Args:
        features (np.array): 特征矩阵.
        labels (list or np.array): 标签.
        classifier_type (str): 'knn', 'svm', 'logistic_regression'.
        test_size (float): 测试集比例.
        random_state (int): 随机种子.
        scale_features (bool): 是否对特征进行标准化.
        **kwargs: 传递给分类器的额外参数 (如 knn_neighbors, svm_kernel, svm_c).

    Returns:
        tuple: (model, accuracy, f1_macro, report_dict)
    """
    if features is None or features.size == 0:
        print("错误: 特征为空，无法训练分类器。")
        return None, 0, 0, {}
    if not labels:
        print("错误: 标签为空，无法训练分类器。")
        return None, 0, 0, {}
        
    # 编码标签
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=test_size, random_state=random_state, stratify=encoded_labels
    )

    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # 初始化分类器
    if classifier_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=kwargs.get('knn_neighbors', 5))
    elif classifier_type == 'svm':
        model = SVC(kernel=kwargs.get('svm_kernel', 'rbf'), 
                    C=kwargs.get('svm_c', 1.0), 
                    probability=True, # Needed for some metrics, but slower
                    random_state=random_state)
    elif classifier_type == 'logistic_regression':
        model = LogisticRegression(random_state=random_state, max_iter=1000, solver='liblinear')
    else:
        raise ValueError(f"不支持的分类器类型: {classifier_type}")

    print(f"正在训练 {classifier_type} 分类器...")
    model.fit(X_train, y_train)
    
    print(f"正在评估 {classifier_type} 分类器...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # 获取原始标签名称用于报告
    try:
        target_names = label_encoder.classes_
        report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
        report_str = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    except ValueError as e: # 可能因为某些类别在测试集中未出现
        print(f"生成分类报告时出错: {e}. 可能由于测试集中某些类别缺失。")
        # 尝试不使用target_names生成报告
        unique_test_labels = np.unique(y_test)
        target_names_from_test = label_encoder.inverse_transform(unique_test_labels)

        # 确保y_pred中的标签也在label_encoder中
        y_pred_original_labels = label_encoder.inverse_transform(np.unique(y_pred))
        
        # 创建一个包含所有可能预测标签的映射
        all_possible_labels_encoded = np.unique(np.concatenate((y_test, y_pred)))
        all_possible_labels_original = label_encoder.inverse_transform(all_possible_labels_encoded)

        report_dict = classification_report(y_test, y_pred, labels=all_possible_labels_encoded, target_names=all_possible_labels_original, output_dict=True, zero_division=0)
        report_str = classification_report(y_test, y_pred, labels=all_possible_labels_encoded, target_names=all_possible_labels_original, zero_division=0)


    print("\n分类报告:")
    print(report_str)
    print(f"准确率: {accuracy:.4f}")
    print(f"宏 F1-Score: {f1_macro:.4f}")
    
    return model, accuracy, f1_macro, report_dict, label_encoder

if __name__ == '__main__':
    from config import CLASSIFIER_PARAMS
    # 简单测试
    # 假设有3类，每类10个样本，特征维度为5
    rng = np.random.RandomState(CLASSIFIER_PARAMS['random_state'])
    X_test_features = rng.rand(30, 5) 
    y_test_labels_list = ['TypeA'] * 10 + ['TypeB'] * 10 + ['TypeC'] * 10
    
    print(f"--- 测试 {CLASSIFIER_PARAMS['type']} 分类器 ---")
    model, acc, f1, report, _ = train_and_evaluate_classifier(
        X_test_features, 
        y_test_labels_list,
        classifier_type=CLASSIFIER_PARAMS['type'],
        test_size=CLASSIFIER_PARAMS['test_size'],
        random_state=CLASSIFIER_PARAMS['random_state'],
        knn_neighbors=CLASSIFIER_PARAMS['knn_neighbors'],
        svm_kernel=CLASSIFIER_PARAMS['svm_kernel'],
        svm_c=CLASSIFIER_PARAMS['svm_c']
    )
    if model:
        print(f"模型: {model}")
        print(f"准确率: {acc:.4f}, F1 (Macro): {f1:.4f}")

    # 测试SVM
    print("\n--- 测试 SVM 分类器 ---")
    model_svm, acc_svm, f1_svm, _, _ = train_and_evaluate_classifier(
        X_test_features, y_test_labels_list, classifier_type='svm', svm_kernel='linear'
    )
    if model_svm:
        print(f"SVM 准确率: {acc_svm:.4f}, F1 (Macro): {f1_svm:.4f}")
