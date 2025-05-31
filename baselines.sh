#!/bin/bash
# test_fsl_baseline.sh

echo "=== STARTING FSL BASELINE TEST ==="
echo "Current time: $(date)"
echo "Working directory: $(pwd)"

PYTHON_EXECUTABLE="python3"
FSL_BASELINE_SCRIPT="src/baseline_fsl_evaluator.py"
DATASET="simulated"

# FSL settings (matching LLM experiments)
N_WAY=3
K_SHOT=1  # 1-shot learning
Q_SHOT=1
NUM_TASKS=30

# Results directory
RESULTS_DIR="results/fsl_baseline"
mkdir -p "$RESULTS_DIR"
RESULTS_CSV="$RESULTS_DIR/fsl_baseline_results.csv"

echo ""
echo "FSL Configuration:"
echo "  Dataset: $DATASET"
echo "  N-way: $N_WAY"
echo "  K-shot: $K_SHOT (training samples per class)"
echo "  Q-shot: $Q_SHOT (query samples per class)"
echo "  Number of tasks: $NUM_TASKS"
echo ""

# Test 1: SVM with raw HRRP
echo "--- Running FSL SVM with raw HRRP features ---"
"$PYTHON_EXECUTABLE" "$FSL_BASELINE_SCRIPT" \
    --dataset_key "$DATASET" \
    --model_type "SVM" \
    --feature_type "raw_hrrp" \
    --n_way "$N_WAY" \
    --k_shot "$K_SHOT" \
    --q_shot "$Q_SHOT" \
    --num_tasks "$NUM_TASKS" \
    --output_csv "$RESULTS_CSV"

echo ""
echo "--- Running FSL SVM with scattering centers ---"
"$PYTHON_EXECUTABLE" "$FSL_BASELINE_SCRIPT" \
    --dataset_key "$DATASET" \
    --model_type "SVM" \
    --feature_type "scattering_centers" \
    --n_way "$N_WAY" \
    --k_shot "$K_SHOT" \
    --q_shot "$Q_SHOT" \
    --num_tasks "$NUM_TASKS" \
    --output_csv "$RESULTS_CSV"

echo ""
echo "--- Running FSL RF with scattering centers ---"
"$PYTHON_EXECUTABLE" "$FSL_BASELINE_SCRIPT" \
    --dataset_key "$DATASET" \
    --model_type "RF" \
    --feature_type "scattering_centers" \
    --n_way "$N_WAY" \
    --k_shot "$K_SHOT" \
    --q_shot "$Q_SHOT" \
    --num_tasks "$NUM_TASKS" \
    --output_csv "$RESULTS_CSV"

echo ""
echo "=== FSL BASELINE TEST COMPLETED ==="

if [ -f "$RESULTS_CSV" ]; then
    echo ""
    echo "Results summary:"
    cat "$RESULTS_CSV" | column -t -s ','
fi