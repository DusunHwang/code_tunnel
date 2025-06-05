
# Code Tunnel

This repository contains utilities for experimenting with Gaussian Process Regression (GPR) on high-dimensional sparse data.

The main script `gpr_sparse_pipeline.py` can generate synthetic sparse datasets
and evaluate several GPR approximations using cross validation. It supports
Exact GPR, variational methods, and a simple Deep Kernel Learning (DKL) model.
Usage example:

```bash
python gpr_sparse_pipeline.py --model all --folds 5
```

To specifically run the DKL variant use:

```bash
python gpr_sparse_pipeline.py --model dkl --test
```

Use `--test` for a quick run on smaller data.
