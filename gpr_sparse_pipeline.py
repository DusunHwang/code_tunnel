# -*- coding: utf-8 -*-
"""Pipeline for comparing GPR models on sparse data."""

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

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
    parser.add_argument("--test", action="store_true", help="run a quick test")
    args = parser.parse_args()

    X, y = generate_sparse_data(n_samples=500 if args.test else 3000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Exact GPR...")
    gpr = train_exact_gpr(X_train, y_train)
    results = evaluate(gpr, X_test, y_test)
    print("Exact GPR results", results)

    if gpytorch:
        print("Training SVGP...")
        svgp_model, likelihood = train_svgp(X_train, y_train, num_inducing=50)
        mean, var = predict_svgp(svgp_model, likelihood, X_test)
        svgp_results = evaluate(gpr, X_test, y_test, predictive_variance=(mean, var))
        print("SVGP results", svgp_results)
    else:
        print("gpytorch not available; skipping SVGP")


if __name__ == "__main__":
    main()
