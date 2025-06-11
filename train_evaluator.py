import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_data(data_dir="./data"):
    """Load train/val/test CSV files from ``data_dir``."""
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def preprocess(train_df, val_df, test_df, target="target"):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df.drop(columns=[target]))
    y_train = train_df[target].values
    X_val = scaler.transform(val_df.drop(columns=[target]))
    y_val = val_df[target].values
    X_test = scaler.transform(test_df.drop(columns=[target]))
    y_test = test_df[target].values
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_processed_datasets(X_train, y_train, X_val, y_val, X_test, y_test, out_dir="./data/processed"):
    """Save preprocessed arrays so experiments are reproducible."""
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "train.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(out_dir, "val.npz"), X=X_val, y=y_val)
    np.savez(os.path.join(out_dir, "test.npz"), X=X_test, y=y_test)


class RegNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(X_train, y_train, X_val, y_val, device="cpu", epochs=50, batch_size=32):
    model = RegNet(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred_train = model(torch.from_numpy(X_train).float().to(device)).cpu().numpy()
            pred_val = model(torch.from_numpy(X_val).float().to(device)).cpu().numpy()
        train_mse = mean_squared_error(y_train, pred_train)
        val_mse = mean_squared_error(y_val, pred_val)
        train_r2 = r2_score(y_train, pred_train)
        val_r2 = r2_score(y_val, pred_val)
        logs.append([epoch, train_mse, train_r2, val_mse, val_r2])
        print(f"Epoch {epoch}: train_mse={train_mse:.4f} val_mse={val_mse:.4f}")
    return model, np.array(logs)


def evaluate(model, X, y, device="cpu"):
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X).float().to(device)).cpu().numpy()
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    return preds, mse, r2


def plot_results(y_train, pred_train, y_test, pred_test, mse, r2, out_path="./plots/train_vs_test.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.scatter(y_train, pred_train, color="blue", label="Train", s=10, alpha=0.5)
    plt.scatter(y_test, pred_test, color="red", label="Test", s=10, alpha=0.5)
    lims = [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())]
    plt.plot(lims, lims, "k--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"R2={r2:.3f} MSE={mse:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_logs(logs, path="./logs/train_log.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(logs, columns=["epoch", "train_mse", "train_r2", "val_mse", "val_r2"])
    df.to_csv(path, index=False)


def save_report(val_mse, val_r2, test_mse, test_r2, path="./reports/eval.md"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    high_perf = val_r2 > 0.85
    with open(path, "w") as f:
        f.write(f"## Evaluation Report\n")
        f.write(f"Validation MSE: {val_mse:.4f}\n")
        f.write(f"Validation R2: {val_r2:.4f}\n")
        f.write(f"Test MSE: {test_mse:.4f}\n")
        f.write(f"Test R2: {test_r2:.4f}\n")
        if high_perf:
            f.write("high_performance: true\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_df, val_df, test_df = load_data()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess(train_df, val_df, test_df)
    save_processed_datasets(X_train, y_train, X_val, y_val, X_test, y_test)
    model, logs = train_model(X_train, y_train, X_val, y_val, device=device)
    save_logs(logs)
    pred_train, train_mse, train_r2 = evaluate(model, X_train, y_train, device=device)
    pred_val, val_mse, val_r2 = evaluate(model, X_val, y_val, device=device)
    pred_test, test_mse, test_r2 = evaluate(model, X_test, y_test, device=device)
    plot_results(y_train, pred_train, y_test, pred_test, test_mse, test_r2)
    save_report(val_mse, val_r2, test_mse, test_r2)
    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), "./models/final_model.pt")


if __name__ == "__main__":
    main()
