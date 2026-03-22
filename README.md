# AI Home Training Coach

A computer vision project that goes from a webcam to a mobile app — using MediaPipe, PyTorch, Streamlit, Gradio, and TensorFlow Lite.

## Requirements

- Python 3.11 (3.14 is not supported by mediapipe)
- Webcam

## Project Structure

```
trainer/
├── config.py                 <- Global constants (paths, hyperparameters)
├── requirements.txt
├── pyrightconfig.json        <- IDE type-checking config
├── data/
│   ├── raw/                  <- Collected JSON files from Phase 1
│   └── processed/
├── models/
│   ├── checkpoints/          <- best_model.pth (saved after training)
│   └── exported/             <- .onnx / .tflite (exported models)
├── phase1/
│   └── pose_extractor.py     <- Webcam -> joint landmarks -> JSON
├── phase2/
│   └── model.py              <- Dataset / LSTM model / training loop
├── phase3/
│   ├── dashboard.py          <- Streamlit real-time dashboard
│   └── gradio_demo.py        <- Gradio public demo
└── phase4/
    └── convert_tflite.py     <- PyTorch -> ONNX -> TFLite conversion
```

## Setup

### 1. Create a Python 3.11 virtual environment

```powershell
cd C:\Users\A\Desktop\AIproject\hometraining

py -3.11 -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

> **Phase 4 only** (mobile conversion) requires additional packages:
> ```powershell
> pip install onnx onnx-tf tensorflow
> ```

---

## Running the Project

> **All scripts must be run from the `trainer/` root directory.**
> Running from inside a phase subfolder will cause import errors.

### Phase 1 — Collect training data

```powershell
python phase1/pose_extractor.py --label squat  --frames 300
python phase1/pose_extractor.py --label pushup --frames 300
python phase1/pose_extractor.py --label lunge  --frames 300
python phase1/pose_extractor.py --label idle   --frames 200
```

- Omitting `--label` runs a live preview without saving.
- Collect at least **200–300 frames per exercise** for good accuracy.
- Make sure your **full body is visible** in the camera frame.

### Phase 2 — Train the model

```powershell
python phase2/model.py
```

- Automatically loads all JSON files from `data/raw/`
- Saves the best checkpoint to `models/checkpoints/best_model.pth`

### Phase 3-A — Real-time dashboard (Streamlit)

```powershell
streamlit run phase3/dashboard.py
```

- Open your browser at `http://localhost:8501`
- Toggle "Start Analysis" in the sidebar to begin

### Phase 3-B — Public demo (Gradio)

```powershell
python phase3/gradio_demo.py
```

- With `share=True` enabled, a public URL is generated via HuggingFace tunnel
- Upload a video or image to analyze posture

### Phase 4 — Mobile conversion (TFLite)

```powershell
python phase4/convert_tflite.py
```

- Converts `best_model.pth` -> `exercise_model.onnx` -> `exercise_model.tflite`
- Output is saved to `models/exported/`
- Copy the `.tflite` file to your Flutter app's `android/app/src/main/assets/`

---

## Supported Exercises

| Label    | Exercise |
|----------|----------|
| `squat`  | Squat    |
| `pushup` | Push-up  |
| `lunge`  | Lunge    |
| `idle`   | Resting / no exercise |

To add a new exercise, add its name to `EXERCISE_LABELS` in `config.py` and collect data for it.

---

## Tips

- **Start with one exercise first** (e.g. squat only) to verify the full pipeline before adding more.
- **Good lighting** and a **clear background** significantly improve landmark detection.
- Collect data in the same environment where you plan to use the app.
- Phase 4 requires Phase 2 to be completed first.
