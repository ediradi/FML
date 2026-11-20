"""Aufteilung der Trainingsdaten auf FL-Clients (horizontale Partitionierung).

- iid_partition:        zufällig, gleich groß, gleiche Labelverteilung (stratifiziert)
- dirichlet_partition:  Label-Skew über Dirichlet(alpha); kleines alpha = stärker non-IID
- attribute_partition:  Feature-/Domain-Skew, z. B. Clients = Altersgruppen oder workclass
- quantity_partition:   Mengen-Skew, Clients mit sehr unterschiedlich vielen Daten
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def iid_partition(y: np.ndarray, n_clients: int, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(n_clients)]
    for c in np.unique(y):  # stratifiziert pro Klasse
        idx = rng.permutation(np.where(y == c)[0])
        for k, chunk in enumerate(np.array_split(idx, n_clients)):
            parts[k].extend(chunk)
    return [rng.permutation(np.array(p)) for p in parts]


def dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float = 0.5,
                        seed: int = 42, min_size: int = 50) -> list[np.ndarray]:
    """Standardverfahren (z. B. Hsu et al. 2019): pro Klasse Anteile ~ Dir(alpha)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    for _ in range(1000):
        parts = [[] for _ in range(n_clients)]
        for c in classes:
            idx = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk)
        if min(len(p) for p in parts) >= min_size:
            return [rng.permutation(np.array(p)) for p in parts]
    raise RuntimeError("Keine Partition mit min_size gefunden – alpha erhöhen oder min_size senken")


def attribute_partition(df: pd.DataFrame, by: str = "age", n_clients: int = 5,
                        seed: int = 42) -> list[np.ndarray]:
    """Clients nach einem Merkmal: numerisch -> Quantile, kategorisch -> häufigste Werte
    (Rest in den letzten Client). Erzeugt realistischen Feature- und Label-Skew."""
    rng = np.random.default_rng(seed)
    col = df[by]
    if pd.api.types.is_numeric_dtype(col):
        groups = pd.qcut(col, q=n_clients, labels=False, duplicates="drop").to_numpy()
    else:
        top = col.value_counts().index[: n_clients - 1]
        groups = col.map({v: i for i, v in enumerate(top)}).fillna(n_clients - 1).to_numpy(int)
    return [rng.permutation(np.where(groups == k)[0]) for k in range(int(groups.max()) + 1)]


def quantity_partition(y: np.ndarray, n_clients: int, alpha: float = 0.5,
                       seed: int = 42, min_size: int = 50) -> list[np.ndarray]:
    """Größen ~ Dir(alpha), Labelverteilung bleibt pro Client etwa gleich (zufällige Zuteilung)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    p = rng.dirichlet(alpha * np.ones(n_clients))
    sizes = np.maximum((p * (n - min_size * n_clients)).astype(int), 0) + min_size
    sizes[-1] = n - sizes[:-1].sum()
    idx = rng.permutation(n)
    return [np.sort(c) for c in np.split(idx, np.cumsum(sizes)[:-1])]


def describe(parts: list[np.ndarray], y: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "client": range(len(parts)),
        "n": [len(p) for p in parts],
        "pos_rate": [float(y[p].mean()) for p in parts],
    })


if __name__ == "__main__":
    from data import load_raw
    tr, _ = load_raw()
    y = tr["income"].to_numpy()
    print("IID, 5 Clients\n", describe(iid_partition(y, 5), y), "\n")
    for a in (1.0, 0.5, 0.1):
        print(f"Dirichlet alpha={a}, 5 Clients\n", describe(dirichlet_partition(y, 5, a), y), "\n")
    print("Nach Alter (Quintile)\n", describe(attribute_partition(tr, "age", 5), y), "\n")
    print("Nach workclass\n", describe(attribute_partition(tr, "workclass", 4), y))
