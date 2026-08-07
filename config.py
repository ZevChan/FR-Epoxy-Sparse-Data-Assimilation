"""
config.py — 受控数据增量消融实验 共享配置
==============================================
Frozen protocol: 配置由文献训练集确定并冻结
Adaptive protocol: Before/After 各自独立优化但规则相同
"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
import numpy as np

# ===== 路径 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROJECT_DIR = DATA_DIR  # 兼容旧引用
WITH_DATA_DIR = os.path.join(DATA_DIR, "BasicData")
WITHOUT_DATA_DIR = os.path.join(DATA_DIR, "BasicData_wo")

OUTPUT_DIR = os.path.join(BASE_DIR, "Results")

# ===== 目标变量 =====
# 注意：pHRR 已移除，不作为预测目标
TARGETS = ["LOI", "UL94_Rating", "THR", "TSP", "Flexural_Strength"]

DISPLAY_NAMES = {
    "LOI": "LOI",
    "pHRR": "pHRR",
    "THR": "THR",
    "TSP": "TSP",
    "Flexural_Strength": "Flexural Strength",
    "UL94_Rating": "UL-94",
}

UNITS = {
    "LOI": "%",
    "pHRR": "kW/m²",
    "THR": "MJ/m²",
    "TSP": "m²",
    "Flexural_Strength": "MPa",
}

# ===== 实验设计 =====
# seed 42 已固定为方法开发种子：其固定文献留出集参与了方法开发，
# 不再进入正式统计。正式论文使用全新 EVALUATION_SEEDS。
DEVELOPMENT_SEED = 42
PRIMARY_SEED = DEVELOPMENT_SEED
DEFAULT_SEED = PRIMARY_SEED
TEST_SIZE = 0.20

# 正式评价种子（全新 10 个，不含 42）
EVALUATION_SEEDS = [7, 13, 19, 29, 37, 43, 53, 61, 71, 79]  # 全量 10 seeds

# ===== 配置选择参数 =====
N_TRIALS = 50
N_CV_FOLDS = 5
MIN_DELTA = 0.001
PATIENCE = 10
N_BOOTSTRAP = 5000

# ===== K 扫描（最终方法：全整数逐 1 扫描）=====
# K=1..MAX_K 全整数扫描（固定默认 XGBoost 参数 + 5 折 CV，无 patience），
# 因为特征进入顺序会引发性能突变（如 LOI K=94），粗网格会漏掉精确位置。
MAX_K = 1000
# 扫描用固定基础超参数（不调优）
SCAN_BASE_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0,
    "reg_lambda": 1,
}
# 候选 K 集合（只删除完全重复的 K，不压缩相邻 K）
TOP_K_CANDIDATES = 1        # 只对扫描 CV 最高的 1 个 K 做 HPO
N_TOP_CANDIDATE_K = 10      # CV 最高前 N 个 K（备用）
CV_TOLERANCE = 0.002        # CV_max - CV_k <= tol 的 K 也进入候选
K_NEIGHBOR_RADIUS = 2       # 候选 K 附近 ±radius 整数点
PATIENCE_K = 50             # K 递增扫描：连续 N 个 K 无性能提升则停止搜索（不进入 HPO 前的早停）
MIN_DELTA_K = 0.0005        # K 扫描中视为“性能提升”的最小 CV 增量
JOINT_HPO_TRIALS = 50       # 单 K 的 Optuna HPO trials
HPO_PATIENCE = 10            # HPO 早停：连续 N 个 trial 无性能提升则停止优化
MAX_ALTERNATING_ROUNDS = 4  # K—超参数交替搜索最大轮数（收敛判断，非人为固定2轮）
SENTINEL_TOLERANCE = 0.002  # 高K哨兵若超过 K<=1000 最佳 CV 超过该容差则报错扩展扫描

# ── 实验种子 ──
# 正式运行使用 EVALUATION_SEEDS（10 个新种子）
# 开发/调试时可用 SEEDS=[DEVELOPMENT_SEED]
SEEDS = EVALUATION_SEEDS

# ===== 强制保留特征 =====
# 阻燃剂添加量、固化剂添加量、Other Material 添加量、固化温度/时间/压力
FORCED_FEATURE_GROUPS = {
    "flame_retardant_amount": ["Flame_retardant_AdditionAmount(wt%)"],
    "curing_agent_amount": ["Curing_agent_AdditionAmount(wt%)"],
    "other_material_amount": [
        "Other_Material_AdditionAmount_1 (wt%)",
        "Other_Material_AdditionAmount_2 (wt%)",
        "Other_Material_AdditionAmount_3 (wt%)",
    ],
    "curing_strategy": [
        "Curing_Tem1", "Curing_Time1", "Curing_Tem2", "Curing_Time2",
        "Curing_Tem3", "Curing_Time3", "Curing_Tem4", "Curing_Time4",
        "Curing_Tem5", "Curing_Time5", "Curing_Tem6", "Curing_Time6",
        "Curing_Tem7", "Curing_Time8", "Curing_Tem9", "Curing_Time9",
        "Curing_Tem10", "Curing_Time10", "Curing_Pressure",
    ],
}
# 拍平为列表（顺序固定，保证可复现）
FORCED_FEATURES = [c for grp in FORCED_FEATURE_GROUPS.values() for c in grp]

# ===== HPO 搜索空间 =====
HPO_SPACE = {
    "n_estimators": (50, 500),
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha": (0, 10),
    "reg_lambda": (0, 10),
}

# ===== 配色 =====
COLOR_BEFORE = "#5DA5DA"
COLOR_AFTER = "#C91511"
COLOR_LINE = "#8C92AC"

# ===== UMAP =====
SMILES_COLS = [
    "EPOXY STRUCTURE",
    "Flame_retardant",
    "Curing_agent ",
    "Other_Material_1",
    "Other_Material_2",
]

# ===== 输出子目录 =====
for sub in ["SampleSizes", "Frozen", "Adaptive", "Robustness",
            "SHAP_Stability", "Distribution"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)
