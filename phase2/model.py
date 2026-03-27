"""
Phase 2: model.py
PyTorch LSTM 모델 정의 + 학습 루프 + YOLO 사람 감지 래퍼

사용법:
    python phase2/model.py
"""

import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    EXERCISE_LABELS, NUM_CLASSES,
    SEQUENCE_LEN, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
    BATCH_SIZE, EPOCHS, LR,
    DATA_RAW, CHECKPOINT_DIR,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
except ImportError as e:
    print(f"[ERROR] 패키지 미설치: {e}")
    print("pip install torch scikit-learn")
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---- Dataset ----

class ExerciseDataset(Dataset):
    """
    data/raw/*.json -> (sequence_tensor, label_int) 쌍
    슬라이딩 윈도우(stride = SEQUENCE_LEN // 2)로 샘플 생성
    """
    def __init__(self, data_dir: Path = DATA_RAW, seq_len: int = SEQUENCE_LEN):
        self.seq_len = seq_len
        self.samples = []

        self.label_enc = LabelEncoder()
        self.label_enc.fit(EXERCISE_LABELS)

        for path in data_dir.glob("*.json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            label = data.get("label", "")
            if label not in EXERCISE_LABELS:
                continue

            label_idx = int(self.label_enc.transform([label]).tolist()[0])
            lm_seqs   = [np.array(fr["landmarks"]).flatten() for fr in data["frames"]]

            stride = max(1, seq_len // 2)
            for start in range(0, len(lm_seqs) - seq_len, stride):
                seq = np.stack(lm_seqs[start : start + seq_len]).astype(np.float32)
                self.samples.append((seq, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)


# ---- Model ----

class ExerciseLSTM(nn.Module):
    """
    양방향 LSTM 운동 분류기
    in  : (batch, SEQUENCE_LEN, INPUT_DIM)
    out : (batch, NUM_CLASSES)
    """
    def __init__(
        self,
        input_dim:   int = INPUT_DIM,
        hidden_dim:  int = HIDDEN_DIM,
        num_layers:  int = NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout:     float = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # 마지막 타임스텝


# ---- Training ----

def train():
    dataset = ExerciseDataset()
    if len(dataset) == 0:
        print("[WARNING] 학습 데이터 없음. Phase 1 로 데이터를 먼저 수집하세요.")
        return None

    indices  = list(range(len(dataset)))
    tr_idx, va_idx = train_test_split(indices, test_size=0.2, random_state=42)

    tr_loader = DataLoader(
        torch.utils.data.Subset(dataset, tr_idx),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    va_loader = DataLoader(
        torch.utils.data.Subset(dataset, va_idx),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    model     = ExerciseLSTM().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    best_acc  = 0.0

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        tr_loss = 0.0
        for seqs, labels in tr_loader:
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(seqs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()

        # validate
        model.eval()
        correct = 0
        with torch.no_grad():
            for seqs, labels in va_loader:
                seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
                correct += (model(seqs).argmax(1) == labels).sum().item()

        acc = correct / len(va_idx)
        scheduler.step()
        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={tr_loss/len(tr_loader):.4f}  val_acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": acc,
                    "label_classes": list(dataset.label_enc.classes_),
                },
                CHECKPOINT_DIR / "best_model.pth",
            )
            print(f"  -> checkpoint saved  (val_acc={acc:.3f})")

    print(f"\nDone. best val_acc={best_acc:.3f}")
    return model


# ---- YOLO person detector (optional) ----

class PersonDetector:
    """YOLO 로 프레임에서 사람 영역만 크롭 -> MediaPipe 정확도 향상"""
    def __init__(self, weights: str = "yolov8n.pt"):
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights)
        except ImportError:
            print("[WARNING] ultralytics 미설치. pip install ultralytics")
            self.model = None

    def crop(self, frame: np.ndarray, conf: float = 0.5):
        """사람 크롭 리스트 반환: [(crop_img, (x1,y1,x2,y2)), ...]"""
        if self.model is None:
            return []
        results = self.model(frame, classes=[0], conf=conf, verbose=False)
        crops = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crops.append((frame[y1:y2, x1:x2], (x1, y1, x2, y2)))
        return crops


# ---- Real-time classifier ----

class ExerciseClassifier:
    """학습된 체크포인트로 실시간 운동 분류"""
    def __init__(self, ckpt_path: Optional[str] = None):
        path = Path(ckpt_path) if ckpt_path else CHECKPOINT_DIR / "best_model.pth"
        ckpt = torch.load(str(path), map_location=DEVICE, weights_only=False)

        self.label_classes = ckpt["label_classes"]
        num_classes        = len(self.label_classes)

        self.model = ExerciseLSTM(num_classes=num_classes).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.buffer = []

    def update(self, landmarks: list) -> Optional[tuple]:
        """
        새 프레임 관절 데이터 추가.
        버퍼가 찼을 때 (label, confidence) 반환, 아직 부족하면 None.
        """
        flat = np.array(landmarks).flatten().astype(np.float32)
        self.buffer.append(flat)

        if len(self.buffer) > SEQUENCE_LEN:
            self.buffer.pop(0)
        if len(self.buffer) < SEQUENCE_LEN:
            return None

        seq = torch.from_numpy(np.stack(self.buffer)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs   = torch.softmax(self.model(seq), dim=1)[0]
            idx     = int(probs.argmax().item())
        return self.label_classes[idx], float(probs[idx].item())


if __name__ == "__main__":
    train()