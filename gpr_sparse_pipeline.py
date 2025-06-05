# -*- coding: utf-8 -*-
"""Pipeline for comparing GPR models on sparse data."""

import argparse
import numpy as np
from typing import Iterable, Tuple

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge

try:
    import torch
    import gpytorch
except ImportError:  # fallback in case gpytorch is not installed
    gpytorch = None


def generate_sparse_data(n_samples=3000, n_features=300, nnz=10, noise_std=0.1, random_state=0):
    rng = np.random.default_rng(random_state)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    for i in range(n_samples):
        idx = rng.choice(n_features, size=nnz, replace=False)
        X[i, idx] = rng.normal(size=nnz)
    true_w = rng.normal(size=n_features)
    y = X.dot(true_w) + rng.normal(scale=noise_std, size=n_samples)
    return X, y


def train_exact_gpr(X_train, y_train, kernel="rbf"):
    if kernel == "rbf":
        kern = RBF(length_scale=1.0)
    elif kernel == "matern32":
        kern = Matern(nu=1.5)
    else:
        kern = RBF(length_scale=1.0)
    model = GaussianProcessRegressor(kernel=kern, normalize_y=True)
    model.fit(X_train, y_train)
    return model


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
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_svgp(X_train, y_train, num_inducing=50, device="cpu"):
    if not gpytorch:
        raise ImportError("gpytorch is required for SVGP")
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    inducing_points = X_train_t[:num_inducing]
    model = GPyTorchSVGP(inducing_points).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=0.1)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_train_t.size(0))
    training_iter = 200
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = -mll(output, y_train_t)
        loss.backward()
        optimizer.step()
    model.eval()
    likelihood.eval()
    return model, likelihood


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
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        projected = self.feature_extractor(x)
        mean_x = self.mean_module(projected)
        covar_x = self.covar_module(projected)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_dkl(X_train, y_train, feature_dim=32, device="cpu", training_iter=150):
    if not gpytorch:
        raise ImportError("gpytorch is required for DKL")
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = DKLGP(X_train_t, y_train_t, likelihood, feature_dim=feature_dim).to(device)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(training_iter):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = -mll(output, y_train_t)
        loss.backward()
        optimizer.step()
    model.eval()
    likelihood.eval()
    return model, likelihood


def predict_dkl(model, likelihood, X_test, device="cpu"):
    model.eval()
    likelihood.eval()
    X_test_t = torch.from_numpy(X_test).float().to(device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(X_test_t))
    return preds.mean.cpu().numpy(), preds.variance.cpu().numpy()


def cross_validate_model(train_fn, X, y, n_splits=5, **train_kwargs):
    """Run K-fold cross validation and compute error metrics."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []
    errors: Iterable[float] = []
    sigmas: Iterable[float] = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model_info = train_fn(X_train, y_train, **train_kwargs)

        if train_fn is train_svgp:
            model, likelihood = model_info
            y_pred, var_pred = predict_svgp(model, likelihood, X_test)
        elif train_fn is train_dkl:
            model, likelihood = model_info
            y_pred, var_pred = predict_dkl(model, likelihood, X_test)
        elif train_fn is train_nystrom_gpr:
            transformer, ridge, var_train = model_info
            y_pred = ridge.predict(transformer.transform(X_test))
            var_pred = np.ones_like(y_pred) * var_train
        else:
            model = model_info
            if isinstance(model, GaussianProcessRegressor):
                y_pred, std_pred = model.predict(X_test, return_std=True)
                var_pred = std_pred ** 2
            else:
                y_pred = model.predict(X_test)
                var_pred = np.var(y_train - model.predict(X_train)) * np.ones_like(y_pred)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics.append({"mse": mse, "mae": mae, "r2": r2, "sigma_mean": var_pred.mean() ** 0.5})
        errors.extend((y_test - y_pred) ** 2)
        sigmas.extend(var_pred)

    agg = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
    agg["corr_mse_sigma"] = np.corrcoef(errors, sigmas)[0, 1]
    return agg


def evaluate(model, X_test, y_test, predictive_variance=None):
    y_pred = model.predict(X_test) if predictive_variance is None else None
    if predictive_variance is not None:
        y_pred = predictive_variance[0]
        variances = predictive_variance[1]
    else:
        variances = getattr(model, "sigma_", np.var(y_pred - y_test)) * np.ones_like(y_pred)
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
    args = parser.parse_args()

    n_samples = 500 if args.test else 3000
    X, y = generate_sparse_data(n_samples=n_samples)

    def run(name, fn, **kw):
        res = cross_validate_model(fn, X, y, n_splits=args.folds, **kw)
        print(f"{name} CV results", res)

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
