from pathlib import Path

# ---- labels ----
EXERCISE_LABELS = ["squat", "pushup", "lunge", "idle"]
NUM_CLASSES     = len(EXERCISE_LABELS)

# ---- model ----
SEQUENCE_LEN = 30
INPUT_DIM    = 33 * 4   # 33 landmarks x (x, y, z, visibility)
HIDDEN_DIM   = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3

# ---- training ----
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-3

# ---- paths ----
ROOT           = Path(__file__).resolve().parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
EXPORT_DIR     = ROOT / "models" / "exported"

for p in [DATA_RAW, DATA_PROCESSED, CHECKPOINT_DIR, EXPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)
