# SpeakSafe – Trained Models

## Files

| File | Description |
|------|-------------|
| svm_model.pkl     | SVM classifier with RBF kernel          |
| scaler.pkl        | StandardScaler fitted on training features |
| `label_map.json`   | Class index mapping: `{"human": 0, "ai": 1}` |
| `training_report.txt` | Accuracy, AUC, and full classification report from last training run |

## Generating the model

```bash
# 1. Add audio files to data/human/ and data/ai/
# 2. Run training
python scripts/train.py --data-dir data/ --out models/speaksafe_v1.pkl
```

## Note

Model files are excluded from version control (see .gitignore).
Use Git LFS or a model registry (e.g. HuggingFace Hub, MLflow) to track versions.
