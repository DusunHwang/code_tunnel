# -*- coding: utf-8 -*-
"""Pipeline for comparing GPR models on sparse data."""

import argparse
import numpy as np
from typing import Iterable, Tuple
import matplotlib.pyplot as plt
import warnings
import inspect
import time

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.exceptions import ConvergenceWarning

try:
    import torch
    import gpytorch
except ImportError:  # fallback in case gpytorch is not installed
    gpytorch = None


def generate_sparse_data(
    n_samples=3000, n_features=300, nnz=10, noise_std=0.1, random_state=0
):
    rng = np.random.default_rng(random_state)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    for i in range(n_samples):
        idx = rng.choice(n_features, size=nnz, replace=False)
        X[i, idx] = rng.normal(size=nnz)
    true_w = rng.normal(size=n_features)
    y = X.dot(true_w) + rng.normal(scale=noise_std, size=n_samples)
    return X, y


def train_exact_gpr(
    X_train,
    y_train,
    kernel="rbf",
    n_restarts=1,
    retries=2,
    max_mult=3.0,
):
    if kernel == "rbf":
        kern = RBF(length_scale=1.0)
    elif kernel == "matern32":
        kern = Matern(nu=1.5)
    else:
        kern = RBF(length_scale=1.0)

    best_model = None
    best_ll = -np.inf
    base = max(1, n_restarts)
    restarts = base
    for _ in range(retries + 1):
        model = GaussianProcessRegressor(
            kernel=kern, normalize_y=True, n_restarts_optimizer=restarts
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X_train, y_train)
        ll = model.log_marginal_likelihood()
        if ll > best_ll:
            best_ll = ll
            best_model = model
        conv_warn = any(isinstance(msg.message, ConvergenceWarning) for msg in w)
        if not conv_warn or restarts >= base * max_mult:
            break
        restarts = int(np.ceil(restarts * 1.5))

    return best_model


def train_sor(X_train, y_train, subset_size=200, kernel="rbf"):
    """Subset-of-Regressors using a random subset of training data."""
    idx = np.random.choice(len(X_train), min(subset_size, len(X_train)), replace=False)
    return train_exact_gpr(X_train[idx], y_train[idx], kernel=kernel)


def train_nystrom_gpr(X_train, y_train, n_components=100, gamma=1.0):
    """Approximate GPR using Nystroem features and ridge regression."""
    transformer = Nystroem(gamma=gamma, n_components=n_components, random_state=0)
    X_trans = transformer.fit_transform(X_train)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_trans, y_train)
    residual_var = np.var(y_train - ridge.predict(X_trans))
    return transformer, ridge, residual_var


class GPyTorchSVGP(gpytorch.models.ApproximateGP if gpytorch else object):
    def __init__(self, inducing_points):
        if not gpytorch:
            raise ImportError("gpytorch is required for SVGP")
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _train_svgp_once(X_train, y_train, num_inducing, device, training_iter, lr):
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    inducing_points = X_train_t[:num_inducing]
    model = GPyTorchSVGP(inducing_points).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=lr
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_train_t.size(0))
    loss_val = None
    for _ in range(training_iter):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = -mll(output, y_train_t)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
    model.eval()
    likelihood.eval()
    return (model, likelihood, loss_val)


def train_svgp(
    X_train,
    y_train,
    num_inducing=50,
    device="cpu",
    training_iter=200,
    retries=2,
    max_mult=3.0,
    lr=0.1,
):
    if not gpytorch:
        raise ImportError("gpytorch is required for SVGP")

    base_iter = training_iter
    best_model = None
    best_loss = float("inf")
    for _ in range(retries + 1):
        model, likelihood, loss_val = _train_svgp_once(
            X_train, y_train, num_inducing, device, training_iter, lr
        )
        if loss_val < best_loss:
            best_loss = loss_val
            best_model = (model, likelihood)
        if training_iter >= base_iter * max_mult:
            break
        training_iter = int(training_iter * 1.5)

    return best_model


def predict_svgp(model, likelihood, X_test, device="cpu"):
    model.eval()
    likelihood.eval()
    X_test_t = torch.from_numpy(X_test).float().to(device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(X_test_t))
    return preds.mean.cpu().numpy(), preds.variance.cpu().numpy()


class DKLGP(gpytorch.models.ExactGP if gpytorch else object):
    """Deep Kernel Learning model with a simple MLP feature extractor."""

    def __init__(self, train_x, train_y, likelihood, feature_dim=32):
        if not gpytorch:
            raise ImportError("gpytorch is required for DKL")
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(train_x.size(-1), 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, feature_dim),
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        projected = self.feature_extractor(x)
        mean_x = self.mean_module(projected)
        covar_x = self.covar_module(projected)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _train_dkl_once(X_train, y_train, feature_dim, device, training_iter, lr):
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = DKLGP(X_train_t, y_train_t, likelihood, feature_dim=feature_dim).to(device)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    loss_val = None
    for _ in range(training_iter):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = -mll(output, y_train_t)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
    model.eval()
    likelihood.eval()
    return model, likelihood, loss_val


def train_dkl(
    X_train,
    y_train,
    feature_dim=32,
    device="cpu",
    training_iter=150,
    retries=2,
    max_mult=3.0,
    lr=0.01,
):
    if not gpytorch:
        raise ImportError("gpytorch is required for DKL")

    base_iter = training_iter
    best_model = None
    best_loss = float("inf")
    for _ in range(retries + 1):
        model, likelihood, loss_val = _train_dkl_once(
            X_train, y_train, feature_dim, device, training_iter, lr
        )
        if loss_val < best_loss:
            best_loss = loss_val
            best_model = (model, likelihood)
        if training_iter >= base_iter * max_mult:
            break
        training_iter = int(training_iter * 1.5)

    return best_model


def predict_dkl(model, likelihood, X_test, device="cpu"):
    model.eval()
    likelihood.eval()
    X_test_t = torch.from_numpy(X_test).float().to(device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(X_test_t))
    return preds.mean.cpu().numpy(), preds.variance.cpu().numpy()


def cross_validate_model(
    train_fn, X, y, n_splits=5, progress_name=None, **train_kwargs
):
    """Run K-fold cross validation and compute error metrics."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []
    errors_train: Iterable[float] = []
    sigmas_train: Iterable[float] = []
    errors_test: Iterable[float] = []
    sigmas_test: Iterable[float] = []
    for i, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        if progress_name:
            print(f"[{progress_name}] Fold {i}/{n_splits} training...")
        start_time = time.time()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model_info = train_fn(X_train, y_train, **train_kwargs)
        if progress_name:
            duration = time.time() - start_time
            print(f"[{progress_name}] Fold {i}/{n_splits} finished in {duration:.2f}s")

        if train_fn is train_svgp:
            model, likelihood = model_info
            y_pred_test, var_pred_test = predict_svgp(model, likelihood, X_test)
            y_pred_train, var_pred_train = predict_svgp(model, likelihood, X_train)
        elif train_fn is train_dkl:
            model, likelihood = model_info
            y_pred_test, var_pred_test = predict_dkl(model, likelihood, X_test)
            y_pred_train, var_pred_train = predict_dkl(model, likelihood, X_train)
        elif train_fn is train_nystrom_gpr:
            transformer, ridge, var_train = model_info
            y_pred_test = ridge.predict(transformer.transform(X_test))
            var_pred_test = np.ones_like(y_pred_test) * var_train
            y_pred_train = ridge.predict(transformer.transform(X_train))
            var_pred_train = np.ones_like(y_pred_train) * var_train
        else:
            model = model_info
            if isinstance(model, GaussianProcessRegressor):
                y_pred_test, std_pred_test = model.predict(X_test, return_std=True)
                var_pred_test = std_pred_test**2
                y_pred_train, std_pred_train = model.predict(X_train, return_std=True)
                var_pred_train = std_pred_train**2
            else:
                y_pred_test = model.predict(X_test)
                var_pred_test = np.var(y_train - model.predict(X_train)) * np.ones_like(
                    y_pred_test
                )
                y_pred_train = model.predict(X_train)
                var_pred_train = np.var(y_train - y_pred_train) * np.ones_like(
                    y_pred_train
                )

        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        metrics.append(
            {
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "sigma_mean": var_pred_test.mean() ** 0.5,
            }
        )
        errors_test.extend((y_test - y_pred_test) ** 2)
        sigmas_test.extend(var_pred_test)
        errors_train.extend((y_train - y_pred_train) ** 2)
        sigmas_train.extend(var_pred_train)

    agg = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
    agg["corr_mse_sigma"] = np.corrcoef(errors_test, sigmas_test)[0, 1]
    return (
        agg,
        np.array(errors_train),
        np.array(sigmas_train),
        np.array(errors_test),
        np.array(sigmas_test),
    )


def plot_error_sigma_scatter(err_train, sig_train, err_test, sig_test, name):
    """Save scatter plot of squared error vs predictive sigma."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        np.sqrt(sig_train), err_train, alpha=0.5, s=10, label="Train", color="blue"
    )
    ax.scatter(np.sqrt(sig_test), err_test, alpha=0.5, s=10, label="Test", color="red")
    ax.set_title(f"{name} Sigma vs MSE")
    ax.set_xlabel("Predicted sigma")
    ax.set_ylabel("Squared error")
    ax.legend()
    fig.tight_layout()
    fname = f"{name.replace(' ', '_').lower()}_mse_sigma.png"
    plt.savefig(fname)
    plt.close(fig)
    print(f"Saved scatter plot to {fname}")


def evaluate(model, X_test, y_test, predictive_variance=None):
    y_pred = model.predict(X_test) if predictive_variance is None else None
    if predictive_variance is not None:
        y_pred = predictive_variance[0]
        variances = predictive_variance[1]
    else:
        variances = getattr(model, "sigma_", np.var(y_pred - y_test)) * np.ones_like(
            y_pred
        )
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    sigma = np.sqrt(variances)
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "sigma_mean": sigma.mean(),
    }


def main():
    parser = argparse.ArgumentParser(description="GPR sparse data pipeline")
    parser.add_argument(
        "--model",
        choices=["exact", "sor", "nystrom", "svgp", "dkl", "all"],
        default="all",
    )
    parser.add_argument("--folds", type=int, default=5, help="number of CV folds")
    parser.add_argument("--test", action="store_true", help="run a quick test")
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="maximum number of additional training attempts",
    )
    parser.add_argument(
        "--max-mult",
        type=float,
        default=3.0,
        help="maximum multiplier for training iterations",
    )
    args = parser.parse_args()

    n_samples = 500 if args.test else 3000
    X, y = generate_sparse_data(n_samples=n_samples)

    def run(name, fn, **kw):
        params = inspect.signature(fn).parameters
        if "retries" in params:
            kw.setdefault("retries", args.retries)
        if "max_mult" in params:
            kw.setdefault("max_mult", args.max_mult)
        print(f"Running {name}...")
        start_total = time.time()
        res, err_tr, sig_tr, err_te, sig_te = cross_validate_model(
            fn,
            X,
            y,
            n_splits=args.folds,
            progress_name=name,
            **kw,
        )
        total_time = time.time() - start_total
        print(f"{name} finished in {total_time:.2f}s")
        print(f"{name} CV results", res)
        plot_error_sigma_scatter(err_tr, sig_tr, err_te, sig_te, name)

    if args.model in ("exact", "all"):
        run("Exact GPR", train_exact_gpr)

    if args.model in ("sor", "all"):
        run("Subset of Regressors", train_sor)

    if args.model in ("nystrom", "all"):
        run("Nystroem GPR", train_nystrom_gpr)

    if args.model in ("svgp", "all"):
        if gpytorch:
            run("SVGP", train_svgp)
        else:
            print("gpytorch not available; skipping SVGP")

    if args.model in ("dkl", "all"):
        if gpytorch:
            run("DKL", train_dkl)
        else:
            print("gpytorch not available; skipping DKL")


if __name__ == "__main__":
    main()
