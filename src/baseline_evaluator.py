# src/baseline_evaluator.py
import os
import numpy as np
import json
import argparse
import csv
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier  # Added
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle

from data_utils import load_processed_data
from dynamic_neighbor_selector import sc_set_to_feature_vector

# Import default configs for when this script is run standalone for defaults
try:
    from config import (
        DEFAULT_DATASET_KEY, AVAILABLE_DATASETS, TARGET_HRRP_LENGTH,
        PREPROCESS_MAT_TO_NPY, PROCESSED_DATA_DIR, TEST_SPLIT_SIZE, RANDOM_STATE,
        SCATTERING_CENTER_EXTRACTION as DEFAULT_SC_EXTRACTION_CONFIG,
        RESULTS_BASE_DIR as DEFAULT_RESULTS_BASE_DIR,  # Not used for CSV output, but for context
        BASELINE_SVM_PARAMS as DEFAULT_SVM_PARAMS,
        BASELINE_RF_PARAMS as DEFAULT_RF_PARAMS
    )
except ImportError:  # Fallback if config.py is not directly runnable or found
    DEFAULT_DATASET_KEY = "simulated"  # Fallback
    DEFAULT_SVM_PARAMS = {"C": 1.0, "kernel": "rbf", "feature_type": "scattering_centers",
                          "sc_feature_type_for_svm": "pos_amp_flat"}
    DEFAULT_RF_PARAMS = {"n_estimators": 100, "feature_type": "scattering_centers",
                         "sc_feature_type_for_rf": "pos_amp_flat"}
    DEFAULT_SC_EXTRACTION_CONFIG = {"enabled": True, "max_centers_to_keep": 10}  # Simplified for fallback
    RANDOM_STATE = 42


def train_and_eval_classifier(
        model_type, X_train, y_train_encoded, X_test, y_test_encoded,
        model_params, random_state, class_names_for_report, labels_for_report_encoded_unique
):
    scaler = StandardScaler()
    if X_train is None or X_train.shape[0] < 1:
        print(f"{model_type} Baseline: Training data insufficient or empty. Cannot train.")
        return 0.0, 0.0, {}

    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is None or X_test.shape[0] < 1:
        print(f"{model_type} Baseline: Test data empty. Model trained but not evaluated.")
        # Train the model anyway
        if model_type == "SVM":
            model = SVC(**{k: v for k, v in model_params.items() if k in SVC().get_params()}, random_state=random_state,
                        probability=True)
        elif model_type == "RF":
            model = RandomForestClassifier(
                **{k: v for k, v in model_params.items() if k in RandomForestClassifier().get_params()},
                random_state=random_state)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        model.fit(X_train_scaled, y_train_encoded)
        return 0.0, 0.0, {"note": f"{model_type} model trained but not evaluated due to empty test set."}

    X_test_scaled = scaler.transform(X_test)

    if model_type == "SVM":
        # Filter params for SVC
        valid_svm_params = {k: v for k, v in model_params.items() if k in SVC().get_params()}
        model = SVC(**valid_svm_params, random_state=random_state, probability=True)
    elif model_type == "RF":
        # Filter params for RandomForestClassifier
        valid_rf_params = {k: v for k, v in model_params.items() if k in RandomForestClassifier().get_params()}
        model = RandomForestClassifier(**valid_rf_params, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X_train_scaled, y_train_encoded)
    y_pred_encoded = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    f1_macro = f1_score(y_test_encoded, y_pred_encoded, average='macro', zero_division=0)

    report_str, report_dict_out = "", {}
    try:
        report_str = classification_report(y_test_encoded, y_pred_encoded, labels=labels_for_report_encoded_unique,
                                           target_names=class_names_for_report, zero_division=0, digits=3)
        report_dict_out = classification_report(y_test_encoded, y_pred_encoded, labels=labels_for_report_encoded_unique,
                                                target_names=class_names_for_report, output_dict=True, zero_division=0,
                                                digits=3)
    except Exception as e:  # Broad exception for report generation
        print(f"{model_type} Baseline: Classification report generation error: {e}. Using simpler report.")
        try:
            report_str = classification_report(y_test_encoded, y_pred_encoded, zero_division=0, digits=3)
            report_dict_out = classification_report(y_test_encoded, y_pred_encoded, output_dict=True, zero_division=0,
                                                    digits=3)
        except Exception:  # Final fallback
            report_dict_out = {"accuracy": accuracy, "macro avg": {"f1-score": f1_macro}}  # Minimal dict
            report_str = f"Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}"

    print(f"\n{model_type} Baseline Classification Report:\n", report_str)
    print(f"{model_type} Baseline Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}")
    return accuracy, f1_macro, report_dict_out


def run_baseline_for_dataset(dataset_key, model_type, feature_type, baseline_params, sc_extraction_config,
                             config_obj_for_paths, output_csv_file=None):
    print(f"\n--- Running {model_type} Baseline for Dataset '{dataset_key}' on Feature Type '{feature_type}' ---")

    load_sc_flag = True if feature_type == "scattering_centers" else False
    # Pass the full config_obj_for_paths (which is CurrentRunConfig instance) to load_processed_data
    load_res = load_processed_data(dataset_key, config_obj_for_paths, load_scattering_centers=load_sc_flag)

    if load_res[0] is None and load_res[2] is None:
        print(f"Cannot load data for '{dataset_key}'. Skipping {model_type} baseline.");
        return
    X_tr_h, y_tr_o, X_te_h, y_te_o, X_tr_sc_l, X_te_sc_l, le, c_names = load_res

    X_tr_clf, X_te_clf = None, None

    if feature_type == "raw_hrrp":
        X_tr_clf, X_te_clf = X_tr_h, X_te_h
        if X_tr_h is None or X_tr_h.size == 0:
            print(f"Raw HRRP training data empty for '{dataset_key}'. Cannot run {model_type} baseline.");
            return
    elif feature_type == "scattering_centers":
        if not sc_extraction_config["enabled"]:
            print(
                f"SC extraction disabled, but baseline requested SC features. Skipping SC {model_type} for '{dataset_key}'.");
            return
        if not X_tr_sc_l or (
                X_te_h is not None and X_te_h.size > 0 and not X_te_sc_l and (y_te_o is not None and y_te_o.size > 0)):
            print(f"SC list data incomplete for '{dataset_key}'. Cannot run SC {model_type}.");
            return

        max_c = sc_extraction_config["max_centers_to_keep"]
        sc_f_type_clf = baseline_params.get("sc_feature_type_for_svm",
                                            baseline_params.get("sc_feature_type_for_rf", "pos_amp_flat"))
        print(f"Baseline {model_type} using SC feature type: {sc_f_type_clf}")

        if X_tr_sc_l:
            X_tr_clf = np.array([sc_set_to_feature_vector(s, max_c, sc_f_type_clf) for s in X_tr_sc_l])
        else:
            print(f"Training SC list empty for '{dataset_key}'. Cannot run SC {model_type}.");
            return

        if X_te_sc_l:
            X_te_clf = np.array([sc_set_to_feature_vector(s, max_c, sc_f_type_clf) for s in X_te_sc_l])
        elif y_te_o is not None and y_te_o.size > 0:
            print(
                f"Test SC list empty but test labels exist for '{dataset_key}'. {model_type} will be trained only or eval incomplete.")
            X_te_clf = np.array([])
        else:
            X_te_clf = np.array([])
    else:
        print(f"Unknown baseline feature type: {feature_type}. Skipping {model_type}.");
        return

    if y_tr_o is None or y_tr_o.size == 0:
        print(f"Training labels empty for '{dataset_key}'. Skipping {model_type}.");
        return

    y_tr_e = le.transform(y_tr_o)
    y_te_e = le.transform(y_te_o) if y_te_o is not None and y_te_o.size > 0 else np.array([])

    labels_report_e_unique = le.transform(le.classes_)

    # Prepare model-specific params from the generic baseline_params
    current_model_params = {}
    if model_type == "SVM":
        current_model_params = {k: v for k, v in baseline_params.items() if k in ["C", "kernel"]}
    elif model_type == "RF":
        current_model_params = {k: v for k, v in baseline_params.items() if
                                k in ["n_estimators", "max_depth", "min_samples_split",
                                      "min_samples_leaf"]}  # Add more RF params if needed

    acc, f1, report_dict = train_and_eval_classifier(
        model_type, X_tr_clf, y_tr_e, X_te_clf, y_te_e,
        current_model_params, RANDOM_STATE, c_names, labels_report_e_unique
    )

    print(f"{model_type} ({feature_type}) on {dataset_key}: Accuracy={acc:.4f}, F1-Macro={f1:.4f}")

    if output_csv_file:
        file_exists = os.path.isfile(output_csv_file)
        with open(output_csv_file, 'a', newline='') as csvfile:
            fieldnames = ['dataset_key', 'model_type', 'feature_type', 'accuracy', 'f1_macro', 'report_dict_json']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'dataset_key': dataset_key,
                'model_type': model_type,
                'feature_type': feature_type,
                'accuracy': f"{acc:.4f}",
                'f1_macro': f"{f1:.4f}",
                'report_dict_json': json.dumps(report_dict)  # Store full report as JSON string
            })
        print(f"Baseline results for {model_type} ({feature_type}) appended to {output_csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline ML model evaluations.")
    parser.add_argument("--dataset_key", type=str, default=DEFAULT_DATASET_KEY,
                        help="Dataset key from config.AVAILABLE_DATASETS.")
    parser.add_argument("--model_type", type=str, required=True, choices=["SVM", "RF"],
                        help="Type of model to run (SVM or RF).")
    parser.add_argument("--feature_type", type=str, default="scattering_centers",
                        choices=["raw_hrrp", "scattering_centers"], help="Feature type to use.")
    parser.add_argument("--output_csv", type=str, default="results/baseline_performance.csv",
                        help="Path to CSV file to append results.")
    # Add CLI args for specific model params if you want to override defaults from config.py
    # For example: --svm_c, --svm_kernel, --rf_n_estimators

    args = parser.parse_args()

    # Create a minimal config-like object for paths and SC settings for load_processed_data
    # This assumes that the SC files were generated using the SC settings from the main config.py
    # If SC settings are ablated for baselines, this part needs to be more sophisticated.
    class MinimalConfigForBaselines:
        AVAILABLE_DATASETS = AVAILABLE_DATASETS
        PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
        TARGET_HRRP_LENGTH = TARGET_HRRP_LENGTH  # Needed by load_processed_data indirectly
        RANDOM_STATE = RANDOM_STATE  # Needed by load_processed_data indirectly
        # These SC configs are what load_processed_data checks
        sc_extraction_config = DEFAULT_SC_EXTRACTION_CONFIG
        # The following are not directly used by load_processed_data but good to have for consistency
        # if other utils are called that might expect them.
        TEST_SPLIT_SIZE = TEST_SPLIT_SIZE
        PREPROCESS_MAT_TO_NPY = PREPROCESS_MAT_TO_NPY

    config_for_paths = MinimalConfigForBaselines()

    baseline_params_to_use = {}
    sc_extraction_to_use = DEFAULT_SC_EXTRACTION_CONFIG  # Use default SC extraction from main config

    if args.model_type == "SVM":
        baseline_params_to_use = {**DEFAULT_SVM_PARAMS}  # Start with defaults
        # Override with CLI args if provided, e.g.:
        # if args.svm_c: baseline_params_to_use['C'] = args.svm_c
    elif args.model_type == "RF":
        baseline_params_to_use = {**DEFAULT_RF_PARAMS}
        # if args.rf_n_estimators: baseline_params_to_use['n_estimators'] = args.rf_n_estimators

    # Ensure feature_type from CLI is consistent with what's in baseline_params_to_use
    # The feature_type for SC vector construction is handled inside run_baseline_for_dataset

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    run_baseline_for_dataset(
        args.dataset_key,
        args.model_type,
        args.feature_type,  # This determines X_tr_clf, X_te_clf
        baseline_params_to_use,  # Contains C, kernel or n_estimators, etc.
        sc_extraction_to_use,  # Contains max_centers_to_keep, enabled status
        config_for_paths,  # For load_processed_data
        args.output_csv
    )


if __name__ == "__main__":
    main()