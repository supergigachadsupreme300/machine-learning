# Quy-model — Gesture Classifier (Decision Tree)

Classical ML model: flattens gesture images into feature vectors and trains a
Decision Tree (or Random Forest).

## Quick start

```bash
cd Quy-model
pip install -r requirements.txt

# Train (Decision Tree)
python train.py

# Predict
python predict.py path/to/gesture.jpg

# Evaluate
python evaluate.py

# Live camera (display only)
python run_camera.py

# Live camera with gesture actions
python run_camera.py --execute
```

## Outputs

- `models/best.joblib` — trained model bundle
- `models/classes.json` — class names
- `reports/evaluation.json` — metrics
