import os
import numpy as np
import json
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle 

from data_utils import load_processed_data 
from dynamic_neighbor_selector import sc_set_to_feature_vector # 从这里导入

def train_and_eval_svm_baseline(X_train, y_train_encoded, X_test, y_test_encoded, 
                                svm_params, random_state, class_names_for_report, labels_for_report_encoded_unique):
    scaler = StandardScaler()
    if X_train is None or X_train.shape[0] < 2:
        X_train_scaled, X_test_scaled = X_train, X_test if X_test is not None and X_test.shape[0] > 0 else np.array([])
    else:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) if X_test is not None and X_test.shape[0] > 0 else np.array([])
    model = SVC(C=svm_params.get("C",1.0), kernel=svm_params.get("kernel","rbf"), random_state=random_state, probability=True)
    if X_train_scaled is None or X_train_scaled.shape[0]==0 or X_test_scaled is None or X_test_scaled.shape[0]==0:
        return 0.0,0.0,{}
    model.fit(X_train_scaled, y_train_encoded)
    y_pred_encoded = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    f1_macro = f1_score(y_test_encoded, y_pred_encoded, average='macro', zero_division=0)
    try:
        report_str = classification_report(y_test_encoded, y_pred_encoded, labels=labels_for_report_encoded_unique, target_names=class_names_for_report, zero_division=0)
        report_dict = classification_report(y_test_encoded, y_pred_encoded, labels=labels_for_report_encoded_unique, target_names=class_names_for_report, output_dict=True, zero_division=0)
    except ValueError as e:
        report_str = classification_report(y_test_encoded, y_pred_encoded, zero_division=0); report_dict = classification_report(y_test_encoded, y_pred_encoded, output_dict=True, zero_division=0)
    # print("\nSVM 基线分类报告:\n", report_str); print(f"SVM 基线准确率: {accuracy:.4f}, F1: {f1_macro:.4f}")
    return accuracy, f1_macro, report_dict

def run_svm_baseline_for_dataset(dataset_name_key, config):
    print(f"\n--- 为数据集 '{dataset_name_key}' 运行SVM基线 ---")
    feature_type = config.BASELINE_SVM_PARAMS.get("feature_type", "raw_hrrp")
    print(f"基线SVM特征类型: {feature_type}")
    load_sc = True if feature_type == "scattering_centers" else False
    load_res = load_processed_data(dataset_name_key, config, load_scattering_centers=load_sc)
    if load_res[0] is None: print(f"无法加载数据 for '{dataset_name_key}'，跳过SVM。"); return
    X_tr_h, y_tr_o, X_te_h, y_te_o, X_tr_sc_l, X_te_sc_l, le, c_names = load_res

    if feature_type == "raw_hrrp": X_tr_svm, X_te_svm = X_tr_h, X_te_h
    elif feature_type == "scattering_centers":
        if not X_tr_sc_l or not X_te_sc_l: print(f"SC数据为空 for '{dataset_name_key}'，无法运行SC SVM。"); return
        max_c = config.SCATTERING_CENTER_EXTRACTION["max_centers_to_keep"]
        sc_f_type = config.FSL_TASK_SETUP.get("sc_feature_type_for_similarity", "pos_amp_flat") # 复用FSL的配置
        X_tr_svm = np.array([sc_set_to_feature_vector(s, max_c, sc_f_type) for s in X_tr_sc_l])
        X_te_svm = np.array([sc_set_to_feature_vector(s, max_c, sc_f_type) for s in X_te_sc_l])
    else: print(f"未知基线特征类型: {feature_type}。"); return
    if not y_tr_o.size or not y_te_o.size: print(f"标签为空 for '{dataset_name_key}'，跳过SVM。"); return

    y_tr_e, y_te_e = le.transform(y_tr_o), le.transform(y_te_o)
    labels_report_e = le.transform(le.classes_)
    acc, f1, report = train_and_eval_svm_baseline(X_tr_svm,y_tr_e,X_te_svm,y_te_e,config.BASELINE_SVM_PARAMS,config.RANDOM_STATE,c_names,labels_report_e)
    exp_name = f"baseline_svm_{feature_type}_{config.BASELINE_SVM_PARAMS['kernel']}_C{config.BASELINE_SVM_PARAMS['C']}"
    summary = {"dataset":dataset_name_key, "exp_type":f"svm_on_{feature_type}", "exp_name":exp_name, "params":config.BASELINE_SVM_PARAMS, "acc":acc, "f1":f1, "report":report}
    res_dir = os.path.join(config.RESULTS_BASE_DIR,dataset_name_key,exp_name); os.makedirs(res_dir,exist_ok=True)
    with open(os.path.join(res_dir, "summary_baseline.json"), "w", encoding="utf-8") as f: json.dump(summary,f,indent=4,ensure_ascii=False)
    print(f"SVM基线 ({feature_type}) 总结已保存到: {os.path.join(res_dir, 'summary_baseline.json')}")

if __name__ == "__main__": print("baseline_evaluator.py通常由main_experiment.py调用。")
