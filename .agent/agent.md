# Agent: TrainEvaluator

## Purpose
Train and evaluate a regression model on tabular data. Log all epoch results, save train/val/test datasets, and generate comparison plots including R² and MSE.

## Input
- Data folder: `./data/`
  - `train.csv`
  - `val.csv`
  - `test.csv`

## Output
- Model logs: `./logs/train_log.csv`
- Metrics plots: `./plots/train_vs_test.png`
- Evaluation report: `./reports/eval.md`
- Trained model: `./models/final_model.pt`

## Workflow
1. Load and preprocess data.
2. Train the model (recording per-epoch train/validation scores).
3. Evaluate the model on validation and test sets.
4. Generate and save comparison plot with predicted vs actual results.
5. Save trained model to `./models/final_model.pt`.

## Runtime
- Python >= 3.10
- Dependencies: `torch`, `sklearn`, `matplotlib`, `pandas`, `numpy`
- Device: GPU (optional, fallback to CPU)

## Notes
- The plot should display both R² and MSE on the image (legend or title).
- Save model in PyTorch format (`.pt`).
- If R² > 0.85 on validation set, include a "high_performance" flag in the report.
