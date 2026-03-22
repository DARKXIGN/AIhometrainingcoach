"""
Phase 4: convert_tflite.py
PyTorch -> ONNX -> TensorFlow Lite 변환

필요 패키지:
    pip install onnx onnx-tf tensorflow

실행 (trainer/ 폴더에서):
    python phase4/convert_tflite.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    EXERCISE_LABELS,
    SEQUENCE_LEN,
    INPUT_DIM,
    CHECKPOINT_DIR,
    EXPORT_DIR,
)


# ---- Step 1: PyTorch -> ONNX ----

def export_onnx() -> str:
    try:
        import torch
    except ImportError:
        print("[ERROR] torch 미설치: pip install torch")
        sys.exit(1)

    # torch 가 확인된 후에만 model import
    from phase2.model import ExerciseLSTM  # type: ignore[import]

    ckpt_path = CHECKPOINT_DIR / "best_model.pth"
    if not ckpt_path.exists():
        print("[ERROR] 체크포인트 없음. Phase 2 학습을 먼저 완료하세요.")
        sys.exit(1)

    ckpt        = torch.load(str(ckpt_path), map_location="cpu")
    num_classes = len(ckpt["label_classes"])

    model = ExerciseLSTM(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ★ torch.onnx.export 의 args 는 반드시 tuple 이어야 함
    dummy     = torch.randn(1, SEQUENCE_LEN, INPUT_DIM)
    onnx_path = str(EXPORT_DIR / "exercise_model.onnx")

    torch.onnx.export(
        model,
        (dummy,),          # tuple 로 감싸야 함
        onnx_path,
        opset_version=14,
        input_names=["pose_sequence"],
        output_names=["logits"],
        dynamic_axes={"pose_sequence": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"[OK] ONNX 저장: {onnx_path}")
    return onnx_path


# ---- Step 2: ONNX -> TFLite ----

def export_tflite(onnx_path: str) -> str:
    try:
        import onnx
        from onnx_tf.backend import prepare  # type: ignore[import]
        import tensorflow as tf              # type: ignore[import]
    except ImportError as e:
        print(f"[ERROR] {e}\npip install onnx onnx-tf tensorflow")
        sys.exit(1)

    saved_dir = str(EXPORT_DIR / "tf_saved_model")
    tf_rep    = prepare(onnx.load(onnx_path))
    tf_rep.export_graph(saved_dir)

    # Float32
    converter   = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    tflite_path = str(EXPORT_DIR / "exercise_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(converter.convert())
    print(f"[OK] TFLite 저장: {tflite_path}")

    # INT8 양자화
    conv_q = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    conv_q.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_q.target_spec.supported_types = [tf.int8]

    def rep_data():
        for _ in range(100):
            yield [np.random.randn(1, SEQUENCE_LEN, INPUT_DIM).astype(np.float32)]

    conv_q.representative_dataset = rep_data
    quant_path = str(EXPORT_DIR / "exercise_model_int8.tflite")
    with open(quant_path, "wb") as f:
        f.write(conv_q.convert())
    print(f"[OK] INT8 TFLite 저장: {quant_path}")
    return tflite_path


# ---- Step 3: TFLite 추론 테스트 ----

class TFLiteRunner:
    def __init__(self, model_path: str):
        try:
            import tensorflow as tf  # type: ignore[import]
        except ImportError:
            print("[ERROR] tensorflow 미설치: pip install tensorflow")
            sys.exit(1)

        self.interp = tf.lite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()
        self.out = self.interp.get_output_details()

    def predict(self, seq: np.ndarray) -> tuple:
        """seq: (SEQUENCE_LEN, INPUT_DIM) float32"""
        self.interp.set_tensor(self.inp[0]["index"], seq[np.newaxis].astype(np.float32))
        self.interp.invoke()
        logits = self.interp.get_tensor(self.out[0]["index"])[0]
        probs  = np.exp(logits) / np.sum(np.exp(logits))
        idx    = int(np.argmax(probs))
        return EXERCISE_LABELS[idx], float(probs[idx])

    def benchmark(self, n: int = 100) -> None:
        import time
        dummy = np.random.randn(SEQUENCE_LEN, INPUT_DIM).astype(np.float32)
        t0    = time.perf_counter()
        for _ in range(n):
            self.predict(dummy)
        ms = (time.perf_counter() - t0) / n * 1000
        print(f"평균 추론: {ms:.2f} ms / frame")


# ---- 전체 파이프라인 ----

def run_pipeline() -> None:
    print("=== Phase 4: 변환 파이프라인 시작 ===\n")
    onnx_path   = export_onnx()
    tflite_path = export_tflite(onnx_path)

    print("\n=== 추론 속도 테스트 ===")
    TFLiteRunner(tflite_path).benchmark(50)
    print(f"\n완료. 모바일 앱에 포함할 파일: {tflite_path}")


if __name__ == "__main__":
    run_pipeline()
