import os

# --- LLM API 配置 ---
# 重要: 请将 "YOUR_API_KEY_HERE" 替换为您的真实 OpenAI API 密钥
# 或者，如果您使用其他LLM提供商或本地模型，请相应修改 llm_embedder.py
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
LLM_PARAMS = {
    "provider": "openai",  # 'openai' 或 'huggingface_local' (huggingface_local 示例)
    "model_name": "text-embedding-ada-002",  # OpenAI 嵌入模型
    # "model_name": "sentence-transformers/all-MiniLM-L6-v2", # Hugging Face 本地模型示例
    "batch_size": 16, # 对于某些API或本地模型，批处理可以提速
}

# --- 数据集路径 ---
BASE_DATA_PATH = "datasets"
SIMULATED_DATA_PATH = os.path.join(BASE_DATA_PATH, "simulated_hrrp")
MEASURED_DATA_PATH = os.path.join(BASE_DATA_PATH, "measured_hrrp")

# --- 结果保存路径 ---
RESULTS_BASE_PATH = "results"

# --- HRRP 预处理配置 ---
# 仿真数据原始长度1000，实测数据原始长度500
# 我们将实测数据补零到与仿真数据一致的长度，或统一到一个新长度
TARGET_HRRP_LENGTH = 1000 # 所有HRRP将被处理到的目标长度

PREPROCESSING = {
    "normalize": True,             # 是否进行幅度归一化 (True/False for ablation)
    "normalization_type": "max",   # "max" 或 "energy"
    "precision": 3,                # 数值转换为文本时的精度 (小数点后位数)
    "use_space_separator": True,   # 数字间是否使用空格分隔 (True/False for ablation)
                                   # True: "7 5 3", False: "753"
    "value_separator": ", ",       # HRRP值在文本序列中的分隔符
}

# --- 分类器配置 ---
CLASSIFIER_PARAMS = {
    "type": "knn",  # 'knn', 'svm', 'logistic_regression' (for ablation)
    "knn_neighbors": 5,
    "svm_kernel": "rbf",
    "svm_c": 1.0,
    "test_size": 0.3, # 训练集测试集划分比例
    "random_state": 42
}

# --- 基线方法配置 ---
BASELINE_CLASSIFIER_PARAMS = {
    "type": "svm", # 使用原始HRRP（可能归一化后）作为特征的基线
    "svm_kernel": "rbf",
    "svm_c": 1.0,
}


# --- 可视化配置 ---
VISUALIZATION = {
    "tsne_perplexity": 30,
    "tsne_n_iter": 1000,
}

# --- 实验运行控制 ---
RUN_SIMULATED_DATA = True
RUN_MEASURED_DATA = True
RUN_BASELINE_EXPERIMENT = True # 是否运行基于原始HRRP的基线分类器

# 消融实验可以通过多次修改这里的配置并运行main.py来实现
# 例如, 改变 PREPROCESSING['normalize'] 或 PREPROCESSING['use_space_separator']
# 或 CLASSIFIER_PARAMS['type'] 或 LLM_PARAMS['model_name']