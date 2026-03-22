"""
Phase 1: pose_extractor.py
OpenCV + MediaPipe 로 웹캠에서 관절 좌표를 추출하고 JSON 으로 저장합니다.

실행 (trainer/ 폴더에서):
    python phase1/pose_extractor.py --label squat --frames 300
    python phase1/pose_extractor.py            # 라벨 없음 = 미리보기
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, EXERCISE_LABELS

try:
    import cv2
    import mediapipe as mp                          # type: ignore[import]
    _mp_pose    = mp.solutions.pose                 # type: ignore[attr-defined]
    _mp_drawing = mp.solutions.drawing_utils        # type: ignore[attr-defined]
except ImportError as e:
    print(f"[ERROR] 패키지 미설치: {e}")
    print("pip install opencv-python mediapipe")
    sys.exit(1)


ANGLE_JOINTS = {
    "left_elbow":  (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_knee":   (23, 25, 27),
    "right_knee":  (24, 26, 28),
    "left_hip":    (11, 23, 25),
    "right_hip":   (12, 24, 26),
}


@dataclass
class PoseFrame:
    landmarks: list
    angles:    dict
    timestamp_ms: int


def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba    = a - b
    bc    = c - b
    cos_v = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0))))


def get_angles(landmarks) -> dict:  # type: ignore[type-arg]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    return {
        name: calc_angle(pts[i1], pts[i2], pts[i3])
        for name, (i1, i2, i3) in ANGLE_JOINTS.items()
    }


def draw_overlay(
    frame: np.ndarray,
    results,                # type: ignore[type-arg]
    angles: dict,           # type: ignore[type-arg]
    label: Optional[str],
    idx: int,
    total: int,
) -> None:
    _mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        _mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=_mp_drawing.DrawingSpec(
            color=(0, 255, 128), thickness=2, circle_radius=3
        ),
        connection_drawing_spec=_mp_drawing.DrawingSpec(
            color=(255, 255, 255), thickness=1
        ),
    )
    y = 28
    for name, angle in angles.items():
        cv2.putText(frame, f"{name}: {angle:.1f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 1, cv2.LINE_AA)
        y += 20

    if label:
        cv2.putText(
            frame, f"[REC] {label}  {idx}/{total}",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2, cv2.LINE_AA,
        )


def run(label: Optional[str] = None, max_frames: int = 300, camera_id: int = 0) -> None:
    cap         = cv2.VideoCapture(camera_id)
    frames_data = []
    idx         = 0

    with _mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                angles = get_angles(results.pose_landmarks.landmark)
                draw_overlay(frame, results, angles, label, idx, max_frames)

                if label and idx < max_frames:
                    lm_list = [
                        [lm.x, lm.y, lm.z, lm.visibility]
                        for lm in results.pose_landmarks.landmark
                    ]
                    frames_data.append(asdict(PoseFrame(
                        landmarks=lm_list,
                        angles=angles,
                        timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC)),
                    )))
                    idx += 1
                    if idx >= max_frames:
                        print(f"수집 완료: {max_frames} frames")
                        break

            cv2.imshow("Pose Extractor  (q: quit)", frame)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    if label and frames_data:
        existing = list(DATA_RAW.glob(f"{label}_*.json"))
        out_path = DATA_RAW / f"{label}_{len(existing)}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"label": label, "frames": frames_data}, f)
        print(f"저장 완료: {out_path}  ({len(frames_data)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label", default=None,
        choices=EXERCISE_LABELS,           # ★ None 제거 (str 타입 충돌 해결)
        help="저장할 운동 이름 (없으면 미리보기 모드)",
    )
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run(label=args.label, max_frames=args.frames, camera_id=args.camera)
