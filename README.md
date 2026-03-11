# Digit Recognizer Project

## 1. Overview

This project trains machine learning models to classify handwritten digits.
The system supports two groups of models:

1. Classical machine learning baseline models
2. A neural network model (MLP)

The training pipeline is organized into **two phases**:

**Phase 1 — Model Selection**
Multiple models are trained on a train/validation split and evaluated.
The best performing model architecture is selected.

**Phase 2 — Final Training**
Only the selected best model is retrained using the full training dataset.

This design avoids data leakage and ensures the final model uses all available training data.

All results are saved inside the `output` directory.

---

## 2. Project Structure

```
DIGIT-RECOGNIZER/
│
├── dataset/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│
├── output/
│
├── src/
│   ├── models/
│   │   ├── base_models.py
│   │   └── mlp_model.py
│   │
│   ├── utils.py
│   └── run.py
│
├── requirements.txt
└── README.md
```

**dataset**
Contains the input data such as `train.csv` and `test.csv`.

**src/models**
Contains model implementations.

**src/utils.py**
Helper functions for logging, saving results, and preparing output directories.

**src/run.py**
Main training script implementing the full pipeline.

**output**
Stores experiment results, trained models, and submissions.

---

## 3. Output Structure

Each run creates a folder inside `output`.

Example:

```
output/
  all/
    exp1/
      metrics.json
      metrics.txt
      leaderboard.csv
      meta.json

  final/
    final1/
      model.joblib
      submission.csv
```

Description of files:

**metrics.json**
Machine readable evaluation metrics.

**metrics.txt**
Human readable evaluation summary.

**leaderboard.csv**
Accuracy comparison between all tested models.

**meta.json**
Metadata describing the experiment configuration.

**model.joblib**
Serialized final trained model.

**submission.csv**
Prediction file for Kaggle evaluation.

---

## 4. Installation

Create a Python environment and install dependencies.

```
pip install -r requirements.txt
```

---

## 5. Phase 1 — Model Selection

Train models and evaluate them on a validation split.

### Train only the MLP model

```
python src/run.py --model mlp
```

### Train baseline models only

```
python src/run.py --model base
```

### Train all models

```
python src/run.py --model all --run-name exp1
```

This step will:

1. Split the dataset into train/validation
2. Train the selected models
3. Evaluate accuracy
4. Select the best model architecture

Example output:

```
output/all/exp1/
```

This folder contains evaluation results but **not the final trained model**.

---

## 6. Phase 2 — Final Training

Retrain the selected best model on the **full training dataset**.

```
python src/run.py \
  --fit-full-train \
  --selection-dir output/all/exp1 \
  --run-name final1
```

Output:

```
output/final/final1/model.joblib
```

Only the best model is retrained and saved.

---

## 7. Generate Submission

You can generate predictions for the test set using the final model.

```
python src/run.py \
  --fit-full-train \
  --selection-dir output/all/exp1 \
  --make-submission \
  --run-name final1
```

This will produce:

```
output/final/final1/submission.csv
```

---

## 8. Predict Using an Existing Model

If a trained model already exists, predictions can be generated without retraining.

```
python src/run.py \
  --predict-only \
  --model-path output/final/final1/model.joblib
```

---

## 9. Leaderboard File

The system produces a `leaderboard.csv` file showing model performance.

Example:

```
model,accuracy
logreg,0.91
dt,0.86
mlp,0.97
```

The model with the highest accuracy is selected for the final training stage.

---

## 10. Typical Workflow

Recommended workflow:

### Step 1 — Model selection

```
python src/run.py --model all --run-name exp1
```

### Step 2 — Final training

```
python src/run.py --fit-full-train --selection-dir output/all/exp1 --run-name final1
```

### Step 3 — Generate submission

```
python src/run.py --predict-only --model-path output/final/final1/model.joblib --run-name submit1
```

---

## 11. Notes

Each run automatically creates a timestamped experiment directory unless a custom `--run-name` is provided.

Example:

```
python src/run.py --model all --run-name experiment_01
```

This makes experiments easy to track and reproduce.
