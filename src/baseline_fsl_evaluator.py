# src/baseline_fsl_evaluator.py
import os
import numpy as np
import json
import argparse
import csv
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import pickle

from data_utils import load_processed_data, build_fsl_tasks
from dynamic_neighbor_selector import sc_set_to_feature_vector

# Import configs
try:
    from config import (
        DEFAULT_DATASET_KEY, AVAILABLE_DATASETS, TARGET_HRRP_LENGTH,
        PROCESSED_DATA_DIR, TEST_SPLIT_SIZE, RANDOM_STATE,
        SCATTERING_CENTER_EXTRACTION as DEFAULT_SC_EXTRACTION_CONFIG,
        SCATTERING_CENTER_ENCODING as DEFAULT_SC_ENCODING_CONFIG,
        DEFAULT_FSL_TASK_SETUP,
        BASELINE_SVM_PARAMS as DEFAULT_SVM_PARAMS,
        BASELINE_RF_PARAMS as DEFAULT_RF_PARAMS
    )
except ImportError:
    DEFAULT_DATASET_KEY = "simulated"
    DEFAULT_SVM_PARAMS = {"C": 1.0, "kernel": "rbf"}
    DEFAULT_RF_PARAMS = {"n_estimators": 100}
    DEFAULT_SC_EXTRACTION_CONFIG = {"enabled": True, "max_centers_to_keep": 10}
    DEFAULT_SC_ENCODING_CONFIG = {"format": "list_of_dicts", "precision_pos": 0, "precision_amp": 3}
    DEFAULT_FSL_TASK_SETUP = {"n_way": 3, "k_shot_support": 1, "q_shot_query": 1, "num_fsl_tasks": 30}
    RANDOM_STATE = 42


def train_and_eval_fsl_task(model_type, support_features, support_labels,
                            query_features, query_labels, model_params, random_state):
    """Train model on support set and evaluate on query set for one FSL task"""

    if support_features is None or len(support_features) == 0:
        return None, None

    # Scale features
    scaler = StandardScaler()
    support_features_scaled = scaler.fit_transform(support_features)

    if query_features is None or len(query_features) == 0:
        return None, None

    query_features_scaled = scaler.transform(query_features)

    # Create and train model
    if model_type == "SVM":
        valid_params = {k: v for k, v in model_params.items() if k in SVC().get_params()}
        model = SVC(**valid_params, random_state=random_state, probability=True)
    elif model_type == "RF":
        valid_params = {k: v for k, v in model_params.items() if k in RandomForestClassifier().get_params()}
        model = RandomForestClassifier(**valid_params, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    try:
        model.fit(support_features_scaled, support_labels)
        predictions = model.predict(query_features_scaled)

        # Calculate metrics for this task
        acc = accuracy_score(query_labels, predictions)
        f1 = f1_score(query_labels, predictions, average='macro', zero_division=0)

        return acc, f1
    except Exception as e:
        print(f"Error in training/evaluation: {e}")
        return None, None


def run_fsl_baseline(dataset_key, model_type, feature_type, model_params,
                     fsl_config, sc_extraction_config, sc_encoding_config,
                     config_obj, output_csv_file=None):
    """Run FSL baseline evaluation"""

    print(f"\n--- Running FSL {model_type} Baseline ---")
    print(f"Dataset: {dataset_key}, Feature: {feature_type}")
    print(f"FSL Setup: {fsl_config['n_way']}-way, {fsl_config['k_shot_support']}-shot")

    # Load data
    load_sc_flag = (feature_type == "scattering_centers")
    load_res = load_processed_data(dataset_key, config_obj, load_scattering_centers=load_sc_flag)

    if load_res[0] is None and load_res[2] is None:
        print(f"Cannot load data for '{dataset_key}'.")
        return

    _, _, X_test_hrrp, y_test_original, _, X_test_sc_list, label_encoder, class_names = load_res

    if X_test_hrrp is None or X_test_hrrp.size == 0:
        print(f"Test data is empty for '{dataset_key}'.")
        return

    # Build FSL tasks
    print(f"Building {fsl_config['num_fsl_tasks']} FSL tasks...")
    fsl_tasks = build_fsl_tasks(
        X_test_hrrp, y_test_original, X_test_sc_list,
        label_encoder, fsl_config, sc_encoding_config, sc_extraction_config,
        config_obj.RANDOM_STATE
    )

    if not fsl_tasks:
        print(f"Failed to build FSL tasks for '{dataset_key}'.")
        return

    print(f"Successfully built {len(fsl_tasks)} FSL tasks.")

    # Run baseline on each task
    task_accuracies = []
    task_f1_scores = []

    for task_idx, task in enumerate(tqdm(fsl_tasks, desc=f"Running {model_type} on FSL tasks")):
        # Prepare features based on feature_type
        if feature_type == "raw_hrrp":
            support_features = task["support_hrrp_actual"]
            query_features = task["query_hrrp"]
        elif feature_type == "scattering_centers":
            if not sc_extraction_config["enabled"]:
                print("SC extraction disabled but SC features requested.")
                return

            # Convert SC lists to feature vectors
            max_centers = sc_extraction_config["max_centers_to_keep"]
            sc_feature_type = model_params.get("sc_feature_type_for_svm",
                                               model_params.get("sc_feature_type_for_rf", "pos_amp_flat"))

            support_features = np.array([
                sc_set_to_feature_vector(sc_list, max_centers, sc_feature_type)
                for sc_list in task["support_sc_list_actual"]
            ])

            query_features = np.array([
                sc_set_to_feature_vector(sc_list, max_centers, sc_feature_type)
                for sc_list in task["query_sc_list"]
            ])
        else:
            print(f"Unknown feature type: {feature_type}")
            return

        support_labels = task["support_labels_actual"]
        query_labels = task["query_labels"]

        # Train and evaluate on this task
        acc, f1 = train_and_eval_fsl_task(
            model_type, support_features, support_labels,
            query_features, query_labels, model_params, config_obj.RANDOM_STATE
        )

        if acc is not None:
            task_accuracies.append(acc)
            task_f1_scores.append(f1)

    # Calculate overall metrics
    if task_accuracies:
        mean_acc = np.mean(task_accuracies)
        std_acc = np.std(task_accuracies)
        mean_f1 = np.mean(task_f1_scores)
        std_f1 = np.std(task_f1_scores)

        print(f"\n{model_type} FSL Results:")
        print(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"  Valid tasks: {len(task_accuracies)}/{len(fsl_tasks)}")

        # Save results
        if output_csv_file:
            file_exists = os.path.isfile(output_csv_file)
            with open(output_csv_file, 'a', newline='') as csvfile:
                fieldnames = ['dataset_key', 'model_type', 'feature_type',
                              'n_way', 'k_shot', 'q_shot', 'num_tasks',
                              'mean_accuracy', 'std_accuracy', 'mean_f1', 'std_f1',
                              'valid_tasks', 'total_tasks']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    'dataset_key': dataset_key,
                    'model_type': model_type,
                    'feature_type': feature_type,
                    'n_way': fsl_config['n_way'],
                    'k_shot': fsl_config['k_shot_support'],
                    'q_shot': fsl_config['q_shot_query'],
                    'num_tasks': fsl_config['num_fsl_tasks'],
                    'mean_accuracy': f"{mean_acc:.4f}",
                    'std_accuracy': f"{std_acc:.4f}",
                    'mean_f1': f"{mean_f1:.4f}",
                    'std_f1': f"{std_f1:.4f}",
                    'valid_tasks': len(task_accuracies),
                    'total_tasks': len(fsl_tasks)
                })
            print(f"Results saved to {output_csv_file}")
    else:
        print("No valid results obtained.")


def main():
    parser = argparse.ArgumentParser(description="Run FSL baseline evaluations.")
    parser.add_argument("--dataset_key", type=str, default=DEFAULT_DATASET_KEY)
    parser.add_argument("--model_type", type=str, required=True, choices=["SVM", "RF"])
    parser.add_argument("--feature_type", type=str, default="scattering_centers",
                        choices=["raw_hrrp", "scattering_centers"])
    parser.add_argument("--n_way", type=int, default=DEFAULT_FSL_TASK_SETUP["n_way"])
    parser.add_argument("--k_shot", type=int, default=DEFAULT_FSL_TASK_SETUP["k_shot_support"])
    parser.add_argument("--q_shot", type=int, default=DEFAULT_FSL_TASK_SETUP["q_shot_query"])
    parser.add_argument("--num_tasks", type=int, default=DEFAULT_FSL_TASK_SETUP["num_fsl_tasks"])
    parser.add_argument("--output_csv", type=str, default="results/fsl_baseline_results.csv")

    args = parser.parse_args()

    # Create config object
    class ConfigForFSLBaseline:
        AVAILABLE_DATASETS = AVAILABLE_DATASETS
        PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
        TARGET_HRRP_LENGTH = TARGET_HRRP_LENGTH
        RANDOM_STATE = RANDOM_STATE
        sc_extraction_config = DEFAULT_SC_EXTRACTION_CONFIG
        SCATTERING_CENTER_EXTRACTION = DEFAULT_SC_EXTRACTION_CONFIG
        TEST_SPLIT_SIZE = TEST_SPLIT_SIZE

    config = ConfigForFSLBaseline()

    # Setup FSL config
    fsl_config = {
        "enabled": True,
        "n_way": args.n_way,
        "k_shot_support": args.k_shot,
        "q_shot_query": args.q_shot,
        "num_fsl_tasks": args.num_tasks,
        "sc_feature_type_for_prototype": "pos_amp_flat"
    }

    # Get model params
    if args.model_type == "SVM":
        model_params = DEFAULT_SVM_PARAMS.copy()
    else:
        model_params = DEFAULT_RF_PARAMS.copy()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Run FSL baseline
    run_fsl_baseline(
        args.dataset_key,
        args.model_type,
        args.feature_type,
        model_params,
        fsl_config,
        DEFAULT_SC_EXTRACTION_CONFIG,
        DEFAULT_SC_ENCODING_CONFIG,
        config,
        args.output_csv
    )


if __name__ == "__main__":
    main()