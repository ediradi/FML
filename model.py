"""MLP für Adult in PyTorch: Modell, Training, Auswertung.

Wird zentral (Task 1), für die lokalen Baselines und später in den Flower-Clients
verwendet, damit überall exakt dasselbe Modell und dieselbe Trainingsschleife läuft.
"""
from __future__ import annotations

import copy
import random
import warnings

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cpu")  # Modell ist klein, CPU reicht


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class MLP(nn.Module):
    """in_dim -> 64 -> 32 -> 1, ReLU, Dropout. Ausgabe = Logit (Sigmoid in der Loss-Funktion)."""

    def __init__(self, in_dim: int = 89, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.1):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --- Gewichte als numpy-Liste (Format, das Flower austauscht) -----------------------
def get_weights(model: nn.Module) -> list[np.ndarray]:
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, weights: list[np.ndarray]) -> None:
    keys = model.state_dict().keys()
    model.load_state_dict({k: torch.tensor(w) for k, w in zip(keys, weights)}, strict=True)


# --- Training ----------------------------------------------------------------------
def _loader(X, y, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


def train(model: nn.Module, X, y, epochs: int = 1, lr: float = 1e-3, batch_size: int = 64,
          optimizer: str = "adam", proximal_mu: float = 0.0, seed: int = 0) -> float:
    """Trainiert `epochs` Epochen, gibt den mittleren Loss der letzten Epoche zurück.

    proximal_mu > 0 aktiviert den FedProx-Term  mu/2 * ||w - w_global||^2,
    wobei w_global die Gewichte zu Beginn des Aufrufs sind (= globales Modell vom Server).
    """
    model.to(DEVICE).train()
    opt = (torch.optim.Adam(model.parameters(), lr=lr) if optimizer == "adam"
           else torch.optim.SGD(model.parameters(), lr=lr))
    loss_fn = nn.BCEWithLogitsLoss()
    global_params = [p.detach().clone() for p in model.parameters()] if proximal_mu > 0 else None
    loader = _loader(X, y, batch_size, shuffle=True, seed=seed)
    for _ in range(epochs):
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            if global_params is not None:
                prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
                loss = loss + proximal_mu / 2 * prox
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
    return total / max(n, 1)


@torch.no_grad()
def predict_proba(model: nn.Module, X, batch_size: int = 4096) -> np.ndarray:
    model.to(DEVICE).eval()
    out = [torch.sigmoid(model(xb.to(DEVICE))).cpu()
           for (xb,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                                   batch_size=batch_size)]
    return torch.cat(out).numpy()


def evaluate(model: nn.Module, X, y) -> dict:
    """Loss, Accuracy, Balanced Accuracy, F1 (Klasse >50K), ROC-AUC."""
    p = predict_proba(model, X)
    yhat = (p >= 0.5).astype(int)
    eps = 1e-7
    loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    with warnings.catch_warnings():  # einklassige Client-Daten (non-IID) erzeugen sonst Warnungen
        warnings.simplefilter("ignore")
        return {
            "loss": loss,
            "accuracy": accuracy_score(y, yhat),
            "balanced_acc": balanced_accuracy_score(y, yhat),
            "f1": f1_score(y, yhat, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        }


def fit(X_train, y_train, X_val, y_val, max_epochs: int = 50, patience: int = 5,
        lr: float = 1e-3, batch_size: int = 64, hidden=(64, 32), dropout: float = 0.1,
        seed: int = 42, verbose: bool = False):
    """Training mit Early Stopping auf dem Val-Loss. Gibt (bestes Modell, Verlauf) zurück.
    Für zentrales Training und für die lokalen Baselines."""
    set_seed(seed)
    model = MLP(X_train.shape[1], hidden, dropout)
    best, best_loss, wait, history = None, np.inf, 0, []
    for epoch in range(1, max_epochs + 1):
        tr_loss = train(model, X_train, y_train, epochs=1, lr=lr,
                        batch_size=batch_size, seed=seed + epoch)
        val = evaluate(model, X_val, y_val)
        history.append({"epoch": epoch, "train_loss": tr_loss, **{f"val_{k}": v for k, v in val.items()}})
        if verbose:
            print(f"Epoche {epoch:3d}  train_loss {tr_loss:.4f}  val_loss {val['loss']:.4f}  "
                  f"val_acc {val['accuracy']:.4f}")
        if val["loss"] < best_loss - 1e-4:
            best, best_loss, wait = copy.deepcopy(model), val["loss"], 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best, history
