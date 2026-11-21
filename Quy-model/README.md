# Quy-model — Gesture Classifier (Decision Tree / Random Forest)

Classical ML model: flattens gesture images into feature vectors and trains a
Decision Tree (or Random Forest).

## Quick start

```bash
cd Quy-model
pip install -r requirements.txt

# Train (decision tree by default)
python train.py

# Use random forest instead
python train.py --model-name random_forest

# Predict
python predict.py path/to/gesture.jpg

# Evaluate
python evaluate.py
```

## Outputs

- `models/best.joblib` — trained model bundle
- `models/classes.json` — class names
- `reports/evaluation.json` — metrics
