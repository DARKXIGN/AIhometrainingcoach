"""
Phase 3-B: gradio_demo.py
영상 / 이미지 업로드 -> 운동 자세 분석
HuggingFace Spaces 무료 배포 가능

실행 (trainer/ 폴더에서):
    python phase3/gradio_demo.py
"""

import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EXERCISE_LABELS, CHECKPOINT_DIR

try:
    import cv2
    import mediapipe as mp                          # type: ignore[import]
    import gradio as gr
    _mp_pose    = mp.solutions.pose                 # type: ignore[attr-defined]
    _mp_drawing = mp.solutions.drawing_utils        # type: ignore[attr-defined]
except ImportError as e:
    print(f"[ERROR] {e}\npip install opencv-python mediapipe gradio")
    sys.exit(1)


# ---- 모델 (앱 시작 시 1회 로드) ----
classifier: Optional[object] = None
_ckpt = CHECKPOINT_DIR / "best_model.pth"
if _ckpt.exists():
    try:
        from phase2.model import ExerciseClassifier  # type: ignore[import]
        classifier = ExerciseClassifier(str(_ckpt))
        print("모델 로드 완료")
    except Exception as exc:
        print(f"모델 로드 실패 (포즈 분석만 가능): {exc}")


# ---- 각도 계산 ----
def _calc_angle(a, b, c) -> float:   # type: ignore[type-arg]
    ba    = np.array([a.x - b.x, a.y - b.y], dtype=float)
    bc    = np.array([c.x - b.x, c.y - b.y], dtype=float)
    cos_v = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0))))


# ---- 영상 분석 ----
def analyze_video(video_path: Optional[str]) -> Tuple[Optional[str], str]:
    if video_path is None:
        return None, "영상을 업로드해주세요."

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc   = cv2.VideoWriter.fourcc(*"mp4v")  # ★ VideoWriter_fourcc -> VideoWriter.fourcc
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    preds: list      = []
    angles_all: list = []

    with _mp_pose.Pose(min_detection_confidence=0.6) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                _mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    _mp_pose.POSE_CONNECTIONS,
                    _mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3),
                )

                lk = _calc_angle(lm[23], lm[25], lm[27])
                angles_all.append({"left_knee": lk})

                if classifier is not None:
                    lm_list = [[l.x, l.y, l.z, l.visibility] for l in lm]
                    pred = classifier.update(lm_list)  # type: ignore[union-attr]
                    if pred:
                        preds.append(pred[0])
                        cv2.putText(frame, f"{pred[0]}  {pred[1]:.0%}",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 200), 2)
            writer.write(frame)

    cap.release()
    writer.release()
    return out_path, _build_summary(preds, angles_all)


# ---- 이미지 분석 ----
def analyze_image(image: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], str]:
    if image is None:
        return None, "이미지를 업로드해주세요."

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image

    with _mp_pose.Pose(min_detection_confidence=0.6) as pose:
        results = pose.process(rgb)

    if not results.pose_landmarks:
        return image, "포즈를 감지할 수 없습니다."

    lm  = results.pose_landmarks.landmark
    out = image.copy()
    _mp_drawing.draw_landmarks(out, results.pose_landmarks, _mp_pose.POSE_CONNECTIONS)

    JOINTS = [
        ("left_knee",  23, 25, 27),
        ("right_knee", 24, 26, 28),
        ("left_hip",   11, 23, 25),
        ("right_hip",  12, 24, 26),
    ]
    lines = ["### 관절 각도\n"]
    for name, i1, i2, i3 in JOINTS:
        lines.append(f"- {name}: {_calc_angle(lm[i1], lm[i2], lm[i3]):.1f} deg")

    return out, "\n".join(lines)


def _build_summary(preds: list, angles_all: list) -> str:
    if not preds and not angles_all:
        return "포즈를 감지할 수 없었습니다."
    lines = ["## 분석 요약\n"]
    if preds:
        from collections import Counter
        top = Counter(preds).most_common(1)[0]
        lines.append(f"**주 동작**: {top[0]}  ({top[1]} / {len(preds)} frames)")
    if angles_all:
        vals = [a["left_knee"] for a in angles_all]
        lines.append(f"\n**왼쪽 무릎 각도** 평균: {np.mean(vals):.1f} deg  최소: {np.min(vals):.1f} deg")
    return "\n".join(lines)


# ---- Gradio UI ----
with gr.Blocks(title="AI 트레이닝 코치") as demo:   # ★ theme 제거 (버전 호환 문제)
    gr.Markdown("# 🏋️ AI 홈 트레이닝 코치")
    gr.Markdown("운동 영상 또는 이미지를 업로드하면 자세를 분석합니다.")

    with gr.Tab("영상 분석"):
        with gr.Row():
            v_in  = gr.Video(label="운동 영상 업로드", height=300)
            v_out = gr.Video(label="분석 결과",        height=300)
        v_txt = gr.Markdown()
        gr.Button("분석 시작", variant="primary").click(
            analyze_video, inputs=v_in, outputs=[v_out, v_txt]
        )

    with gr.Tab("이미지 분석"):
        with gr.Row():
            i_in  = gr.Image(label="운동 이미지 업로드", type="numpy")
            i_out = gr.Image(label="포즈 오버레이")
        i_txt = gr.Markdown()
        gr.Button("분석 시작", variant="primary").click(
            analyze_image, inputs=i_in, outputs=[i_out, i_txt]
        )

if __name__ == "__main__":
    demo.launch(share=True, server_port=7860)
