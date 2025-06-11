# Code Tunnel

This repository contains utilities for experimenting with Gaussian Process Regression (GPR) on high-dimensional sparse data.

The main script `gpr_sparse_pipeline.py` can generate synthetic sparse datasets
and evaluate several GPR approximations using cross validation. It supports
Exact GPR, variational methods, and a simple Deep Kernel Learning (DKL) model.
Training functions automatically retry with longer optimization if convergence
issues are detected. Control the retry behaviour with the `--retries` and
`--max-mult` options.
When a CUDA GPU is available the script uses it automatically. CPU workloads
are parallelised across roughly 30% of the available cores.
Progress for each cross-validation fold is displayed along with the time taken
so you can monitor long experiments.
Usage example:

```bash
python gpr_sparse_pipeline.py --model all --folds 5
```

To specifically run the DKL variant use:

```bash
python gpr_sparse_pipeline.py --model dkl --test
```

Use `--test` for a quick run on smaller data.
To adjust the retry behaviour you can pass `--retries` and `--max-mult`:

```bash
python gpr_sparse_pipeline.py --model svgp --retries 3 --max-mult 2.5
```

After each model is evaluated, a scatter plot of squared error versus
predicted sigma is saved. Training points are plotted in blue and test points
in red.
Additionally, a prediction comparison plot is generated showing true values on
the x-axis and predictions on the y-axis. The title of this plot displays the
test set R² score and MAE.

## TrainEvaluator Usage

The script `train_evaluator.py` trains a simple neural network on CSV data located in `./data/`.
It records per-epoch metrics to `./logs/train_log.csv`, saves the trained model to
`./models/final_model.pt`, and produces a scatter plot comparing predictions with true
values in `./plots/train_vs_test.png`. After preprocessing, the arrays are saved in
`./data/processed/` so experiments can be reproduced. After evaluation, a report is
written to `./reports/eval.md`. The report includes a `high_performance` flag when the
validation R² exceeds 0.85.

Run it with:

```bash
python train_evaluator.py
```
