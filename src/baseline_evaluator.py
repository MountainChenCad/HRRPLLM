import os
import numpy as np
import json
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle 

from data_utils import load_processed_data 
from dynamic_neighbor_selector import sc_set_to_feature_vector # sc_set_to_feature_vector 仍然有用

def train_and_eval_svm_baseline(X_train, y_train_encoded, X_test, y_test_encoded, 
                                svm_params, random_state, class_names_for_report, labels_for_report_encoded_unique):
    scaler = StandardScaler()
    # 处理 X_train 或 X_test 可能为空或样本数不足的情况
    if X_train is None or X_train.shape[0] < 1: # SVM至少需要一个训练样本
        print("SVM基线：训练数据不足或为空，无法训练模型。")
        return 0.0, 0.0, {}
    
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is None or X_test.shape[0] < 1:
        print("SVM基线：测试数据为空，仅训练模型，不进行评估。")
        # 仍然可以训练模型，但无法评估
        model = SVC(C=svm_params.get("C",1.0), kernel=svm_params.get("kernel","rbf"), random_state=random_state, probability=True)
        model.fit(X_train_scaled, y_train_encoded)
        # print("SVM模型已训练。")
        return 0.0, 0.0, {"note": "Test data empty, model trained but not evaluated."}

    X_test_scaled = scaler.transform(X_test)
        
    model = SVC(C=svm_params.get("C",1.0), kernel=svm_params.get("kernel","rbf"), random_state=random_state, probability=True)
    model.fit(X_train_scaled, y_train_encoded)
    y_pred_encoded = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    f1_macro = f1_score(y_test_encoded, y_pred_encoded, average='macro', zero_division=0)
    
    report_str, report_dict = "", {}
    try:
        # 确保labels参数是有效的，通常是实际出现在y_test_encoded和y_pred_encoded中的所有唯一标签
        # 或者，使用labels_for_report_encoded_unique，它应该是从label_encoder.classes_转换来的
        active_labels = np.unique(np.concatenate((y_test_encoded,y_pred_encoded)))
        active_class_names = [cn for i, cn in enumerate(class_names_for_report) if labels_for_report_encoded_unique[i] in active_labels]
        
        # 如果 active_class_names 与 active_labels 数量不匹配（可能由于 class_names_for_report 不完整），则调整
        if len(active_labels) != len(active_class_names):
             # Fallback: use label_encoder to get names for active_labels if possible
             # This requires label_encoder to be passed or accessed if class_names_for_report is unreliable
             # For now, we proceed with what we have or default.
             pass


        report_str = classification_report(y_test_encoded, y_pred_encoded, labels=labels_for_report_encoded_unique, target_names=class_names_for_report, zero_division=0, digits=3)
        report_dict = classification_report(y_test_encoded, y_pred_encoded, labels=labels_for_report_encoded_unique, target_names=class_names_for_report, output_dict=True, zero_division=0, digits=3)
    except ValueError as e:
        print(f"SVM基线分类报告生成错误: {e}。尝试不指定target_names。")
        try:
            report_str = classification_report(y_test_encoded, y_pred_encoded, labels=labels_for_report_encoded_unique, zero_division=0, digits=3)
            report_dict = classification_report(y_test_encoded, y_pred_encoded, labels=labels_for_report_encoded_unique, output_dict=True, zero_division=0, digits=3)
        except Exception as e_inner:
            print(f"SVM基线分类报告再次生成错误: {e_inner}。使用默认报告。")
            report_str = classification_report(y_test_encoded, y_pred_encoded, zero_division=0, digits=3)
            report_dict = classification_report(y_test_encoded, y_pred_encoded, output_dict=True, zero_division=0, digits=3)
            
    print("\nSVM 基线分类报告:\n", report_str); print(f"SVM 基线准确率: {accuracy:.4f}, F1宏平均: {f1_macro:.4f}")
    return accuracy, f1_macro, report_dict

def run_svm_baseline_for_dataset(dataset_name_key, config):
    print(f"--- 为数据集 '{dataset_name_key}' 运行SVM基线 ---")
    feature_type = config.BASELINE_SVM_PARAMS.get("feature_type", "raw_hrrp")
    print(f"基线SVM特征类型: {feature_type}")
    
    # load_scattering_centers 应基于 feature_type
    load_sc_flag = True if feature_type == "scattering_centers" else False
    load_res = load_processed_data(dataset_name_key, config, load_scattering_centers=load_sc_flag)
    
    if load_res[0] is None: 
        print(f"无法加载数据 for '{dataset_name_key}'，跳过SVM基线。"); return
    X_tr_h, y_tr_o, X_te_h, y_te_o, X_tr_sc_l, X_te_sc_l, le, c_names = load_res

    X_tr_svm, X_te_svm = None, None

    if feature_type == "raw_hrrp": 
        X_tr_svm, X_te_svm = X_tr_h, X_te_h
        if X_tr_h is None or X_tr_h.size == 0:
             print(f"原始HRRP训练数据为空 for '{dataset_name_key}'，无法运行SVM基线。"); return
    elif feature_type == "scattering_centers":
        if not config.SCATTERING_CENTER_EXTRACTION["enabled"]:
            print(f"SC提取已禁用，但基线请求SC特征。跳过SC SVM for '{dataset_name_key}'。"); return
        if not X_tr_sc_l or (X_te_h.size > 0 and not X_te_sc_l and X_te_o.size > 0) : # 如果有测试HRRP，也应该有测试SC
            print(f"SC列表数据不完整 for '{dataset_name_key}' (训练SC: {'有' if X_tr_sc_l else '无'}, 测试SC: {'有' if X_te_sc_l else '无'})，无法运行SC SVM。"); return
        
        max_c = config.SCATTERING_CENTER_EXTRACTION["max_centers_to_keep"]
        # FSL_TASK_SETUP 中的 sc_feature_type_for_similarity 可能与基线SVM所需的不同，这里应有自己的配置或默认
        # 为了简单，我们假设基线SVM使用与FSL相似的扁平化SC特征，或者可以为其定义一个特定类型。
        # 使用 config.BASELINE_SVM_PARAMS.get("sc_feature_type_for_svm", "pos_amp_flat")
        sc_f_type_svm = config.BASELINE_SVM_PARAMS.get("sc_feature_type_for_svm", "pos_amp_flat") 
        print(f"基线SVM使用的SC特征类型: {sc_f_type_svm}")

        if X_tr_sc_l:
            X_tr_svm = np.array([sc_set_to_feature_vector(s, max_c, sc_f_type_svm) for s in X_tr_sc_l])
        else:
             print(f"训练SC列表为空 for '{dataset_name_key}'，无法运行SC SVM。"); return

        if X_te_sc_l: # 只有当测试SC列表存在时才转换
            X_te_svm = np.array([sc_set_to_feature_vector(s, max_c, sc_f_type_svm) for s in X_te_sc_l])
        elif X_te_o.size > 0 : # 如果有测试标签，说明应该有测试数据，但SC列表缺失
             print(f"测试SC列表为空但存在测试标签 for '{dataset_name_key}'。SVM将仅训练或评估不完整。")
             X_te_svm = np.array([]) # 置为空数组，让下游处理
        else: # 没有测试标签，可能就没有测试数据
            X_te_svm = np.array([])

    else: 
        print(f"未知基线特征类型: {feature_type}。跳过SVM。"); return

    if y_tr_o is None or y_tr_o.size == 0: 
        print(f"训练标签为空 for '{dataset_name_key}'，跳过SVM。"); return
    
    y_tr_e = le.transform(y_tr_o)
    y_te_e = le.transform(y_te_o) if y_te_o is not None and y_te_o.size > 0 else np.array([])
    
    labels_report_e_unique = le.transform(le.classes_) # 所有可能的标签编码
    
    acc, f1, report = train_and_eval_svm_baseline(X_tr_svm,y_tr_e,X_te_svm,y_te_e,config.BASELINE_SVM_PARAMS,config.RANDOM_STATE,c_names,labels_report_e_unique)
    
    # 实验命名和结果保存
    svm_kernel = config.BASELINE_SVM_PARAMS.get('kernel','rbf')
    svm_C = config.BASELINE_SVM_PARAMS.get('C',1.0)
    exp_name = f"baseline_svm_{feature_type}_{svm_kernel}_C{svm_C}"
    summary = {"dataset":dataset_name_key, "exp_type":f"svm_on_{feature_type}", "exp_name":exp_name, 
               "params":{"C": svm_C, "kernel": svm_kernel, "feature_details": sc_f_type_svm if feature_type == "scattering_centers" else "raw_hrrp"}, 
               "accuracy":acc, "f1_macro":f1, "classification_report":report}
    
    res_dir = os.path.join(config.RESULTS_BASE_DIR,dataset_name_key,exp_name); 
    os.makedirs(res_dir,exist_ok=True)
    summary_path = os.path.join(res_dir, "summary_baseline_svm.json")
    with open(summary_path, "w", encoding="utf-8") as f: json.dump(summary,f,indent=4,ensure_ascii=False)
    print(f"SVM基线 ({feature_type}) 总结已保存到: {summary_path}")

if __name__ == "__main__": 
    print("baseline_evaluator.py通常由main_experiment.py调用。")
    # 可以添加一个简单的本地测试，如果需要的话，但需要mock config和数据加载