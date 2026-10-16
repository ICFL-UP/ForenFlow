"""Central configuration: paths, feature schema, training hyperparameters."""

from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
KAGGLE_DIR = DATA_DIR / "kaggle"
MODEL_DIR = BASE_DIR / "models" / "saved"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
PLOT_DIR = BASE_DIR / "outputs" / "plots"
RESULTS_DIR = BASE_DIR / "outputs" / "results"

for d in [RAW_DIR, PROCESSED_DIR, KAGGLE_DIR, MODEL_DIR, REPORT_DIR, PLOT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LABEL_COLUMN = "AttackLabel"
RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "PacketSize",
    "ResponseTime",
    "Protocol",
    "SrcPort",
    "DstPort",
    "PacketRate",
    "FlowDuration",
    "NumPackets",
    "PayloadSize",
    "FlagCount",
    "Entropy",
    "BytesSent",
    "BytesReceived",
    "FlowRate",
    "ActiveTime",
    "IdleTime",
    "ByteRatio",
    "TimeGap",
]
NUM_FEATURES = len(FEATURE_COLUMNS)

# Flow extraction 
FLOW_WINDOW_SECONDS = 5
MIN_PACKETS_PER_FLOW = 3

# Sliding window 
SEQUENCE_LENGTH = 5
SEQUENCE_STRIDE = 1

# LSTM Baseline
LSTM_BASE_HIDDEN = 64
LSTM_BASE_LAYERS = 2
LSTM_BASE_DROPOUT = 0.2

# LSTM Improved 
LSTM_IMP_HIDDEN_1 = 128
LSTM_IMP_HIDDEN_2 = 64
LSTM_IMP_DROPOUT = 0.3

# Transformer
TF_D_MODEL = NUM_FEATURES
TF_NHEAD = 2
TF_NUM_LAYERS = 2
TF_DIM_FF = 64
TF_DROPOUT = 0.1

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 15
EARLY_STOP_PATIENCE = 5
TEST_SPLIT = 0.2               # 70 / 10 / 20 train / validation / test
VAL_SPLIT = 0.1

# Youden J thresholds written by run_all.py and read by services/ml.py
YOUDEN_RESULTS_FILE = RESULTS_DIR / "youden_thresholds.json"

# SHAP (global attributions with GradientExplainer)
SHAP_BACKGROUND_SAMPLES = 100
SHAP_MAX_DISPLAY = 10
