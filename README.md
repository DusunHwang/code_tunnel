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

## Hybrid CCM Example

The script `hybrid_ccm_predictor.py` demonstrates a basic hybrid color
correction matrix model. When run without arguments it loads Lab values from
the ``ColorChecker N Ohta`` dataset and trains separate CCMs for fluorescent and
non-fluorescent patches.

```bash
python hybrid_ccm_predictor.py
```
