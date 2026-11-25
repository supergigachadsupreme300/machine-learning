# Toan-model — Hand Gesture Classifier (Landmark-based)

A model pipeline that **takes a hand gesture (landmarks extracted from a
picture) and returns the gesture label (answer)**.

Training data is landmark CSVs (MediaPipe hand landmarks) under `data/raw/`.
Collect data with the provided UI or `collect_dataset.py`, then train.

## Data layout

```
Toan-model/data/
  raw/
    <gesture_id>/
      <gesture_id>.csv   # columns: timestamp, gesture, then 63 landmark coords
  split/                 # optional, created by split_dataset.py (image-based)
    train/ val/ test/
```

Each CSV row is one captured hand: `timestamp, gesture, wrist_x, wrist_y, ...`.

## Quick start

```bash
cd Toan-model
pip install -r requirements.txt

# Train on collected landmark data
python train.py

# (optional) cross-validation
python train.py --cv

# Predict from a landmark vector (JSON list of 63 floats)
python predict.py --landmarks "[0.1, 0.2, ...]"

# Predict from a CSV row
python predict.py --csv path/to/sample.csv

# Predict from an image (requires mediapipe)
python predict.py path/to/gesture.jpg

# Evaluate
python evaluate.py
```

## Inference from Python

```python
from predict import predict_from_landmarks
result = predict_from_landmarks(landmarks_list_63)
print(result["label"], result["confidence"])
```

## Outputs

- `models/best.joblib` — trained model bundle (model + classes + config).
- `models/classes.json` — ordered class names.
- `reports/training_report.json` — metrics.
- `reports/confusion_matrix.png` — test confusion matrix.
- `reports/evaluation.json` — evaluation metrics.

## Model options

`--model-name`: `random_forest` (default), `svm`, `knn`, `gradient_boosting`,
`mlp`, `logistic`.
