"""
Phase 3-A: dashboard.py
Streamlit 실시간 대시보드

실행 (trainer/ 폴더에서):
    streamlit run phase3/dashboard.py
"""

import sys
from pathlib import Path
from collections import deque
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EXERCISE_LABELS, CHECKPOINT_DIR

try:
    import cv2
    import mediapipe as mp                          # type: ignore[import]
    import numpy as np
    import streamlit as st
    _mp_pose    = mp.solutions.pose                 # type: ignore[attr-defined]
    _mp_drawing = mp.solutions.drawing_utils        # type: ignore[attr-defined]
except ImportError as e:
    print(f"[ERROR] {e}\npip install opencv-python mediapipe streamlit")
    sys.exit(1)


# ---- 페이지 설정 ----
st.set_page_config(page_title="AI 트레이닝 코치", page_icon="🏋️", layout="wide")
st.title("🏋️ AI 홈 트레이닝 코치")
st.caption("웹캠 기반 실시간 운동 자세 분석")

# ---- 사이드바 ----
with st.sidebar:
    st.header("설정")
    target_exercise = st.selectbox("목표 운동", EXERCISE_LABELS)
    target_reps     = st.slider("목표 횟수", 5, 50, 15)
    knee_threshold  = st.slider("무릎 각도 기준", 60, 130, 90)
    run_toggle      = st.toggle("분석 시작", value=False)
    st.divider()
    st.caption("q 키 또는 토글 OFF 로 중지")

# ---- 레이아웃 ----
col_vid, col_stat = st.columns([3, 2])
with col_vid:
    video_slot = st.empty()
with col_stat:
    st.subheader("실시간 분석")
    label_slot = st.empty()
    conf_slot  = st.empty()
    rep_slot   = st.empty()
    chart_slot = st.empty()

progress_slot = st.progress(0, text="대기 중...")


# ---- 횟수 카운터 ----
class RepCounter:
    def __init__(self, down_thresh: float = 90.0) -> None:
        self.down_thresh = down_thresh
        self.count       = 0
        self._down       = False

    def update(self, angle: float) -> int:
        if angle < self.down_thresh and not self._down:
            self._down = True
        elif angle >= self.down_thresh + 15 and self._down:
            self._down = False
            self.count += 1
        return self.count


# ---- 각도 계산 ----
def _calc_angle(a, b, c) -> float:   # type: ignore[type-arg]
    ba    = np.array([a.x - b.x, a.y - b.y], dtype=float)
    bc    = np.array([c.x - b.x, c.y - b.y], dtype=float)
    cos_v = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0))))


# ---- 메인 루프 ----
def main() -> None:
    if not run_toggle:
        video_slot.info("사이드바에서 '분석 시작'을 켜주세요.")
        return

    # 모델 로드 시도
    classifier: Optional[object] = None
    ckpt_path = CHECKPOINT_DIR / "best_model.pth"
    if ckpt_path.exists():
        try:
            from phase2.model import ExerciseClassifier  # type: ignore[import]
            classifier = ExerciseClassifier(str(ckpt_path))
            st.sidebar.success("모델 로드 완료")
        except Exception as exc:
            st.sidebar.warning(f"모델 로드 실패: {exc}")
    else:
        st.sidebar.warning("학습 모델 없음 — 포즈 분석만 수행")

    counter    = RepCounter(down_thresh=float(knee_threshold))
    angle_hist = {
        "left_knee":  deque(maxlen=60),
        "right_knee": deque(maxlen=60),
    }

    cap = cv2.VideoCapture(0)

    with _mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as pose:
        while run_toggle and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                _mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    _mp_pose.POSE_CONNECTIONS,
                    _mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3),
                    _mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1),
                )

                lk = _calc_angle(lm[23], lm[25], lm[27])
                rk = _calc_angle(lm[24], lm[26], lm[28])
                angle_hist["left_knee"].append(lk)
                angle_hist["right_knee"].append(rk)

                reps = counter.update(lk)
                rep_slot.metric("운동 횟수", f"{reps} / {target_reps}")
                progress_slot.progress(
                    min(reps / max(target_reps, 1), 1.0),
                    text=f"진행률 {min(reps/max(target_reps,1), 1.0):.0%}",
                )

                if classifier is not None:
                    lm_list = [[l.x, l.y, l.z, l.visibility] for l in lm]
                    pred = classifier.update(lm_list)  # type: ignore[union-attr]
                    if pred:
                        label_slot.metric("현재 동작", pred[0])
                        conf_slot.progress(pred[1], text=f"신뢰도 {pred[1]:.0%}")

                import pandas as pd
                df = pd.DataFrame({k: list(v) for k, v in angle_hist.items()})
                if not df.empty:
                    chart_slot.line_chart(df, height=200)

            video_slot.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )

    cap.release()


main()
