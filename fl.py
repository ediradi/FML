"""Federated Learning mit Flower (Message API, flwr >= 1.20) für das Adult-MLP.

Ablauf einer Simulation:
  1. Server initialisiert das MLP (fester Seed) und schickt die Gewichte an die Clients
  2. Jeder Client trainiert `local_epochs` Epochen auf seiner Partition (85 % davon,
     15 % bleiben als lokale Validierung) und schickt die Gewichte zurück
  3. Strategie (FedAvg, FedProx, FedAvgM, FedMedian) aggregiert zum globalen Modell
  4. Server wertet das globale Modell nach jeder Runde auf Val- und Testset aus

Aufruf (z. B. im Notebook):
    from fl import run_fl
    hist = run_fl(partition="iid", n_clients=10, num_rounds=20, local_epochs=1)
    hist  # DataFrame mit Metriken pro Runde
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAvgM, FedMedian, FedProx
from flwr.simulation import run_simulation

from data import load_dataset
from model import MLP, evaluate, set_seed, train
from partition import attribute_partition, dirichlet_partition, iid_partition, quantity_partition

DATA_SEED = 42  # fester Seed für Train/Val-Split (gleich wie in Notebook 01)


# --- Daten: einmal pro Prozess laden ------------------------------------------------
@lru_cache(maxsize=1)
def _data():
    return load_dataset(val_size=0.15, seed=DATA_SEED)


@lru_cache(maxsize=32)
def get_partitions(partition: str, n_clients: int, alpha: float = 0.5,
                   by: str = "age", seed: int = 42) -> tuple[np.ndarray, ...]:
    """Indizes (bezogen auf X_train) je Client. Gleiche Argumente -> gleiche Aufteilung."""
    d = _data()
    y = d["y_train"]
    if partition == "iid":
        parts = iid_partition(y, n_clients, seed=seed)
    elif partition == "dirichlet":
        parts = dirichlet_partition(y, n_clients, alpha=alpha, seed=seed)
    elif partition == "attribute":
        parts = attribute_partition(d["df_train"], by=by, n_clients=n_clients, seed=seed)
    elif partition == "quantity":
        parts = quantity_partition(y, n_clients, alpha=alpha, seed=seed)
    else:
        raise ValueError(f"Unbekannte Partitionierung: {partition}")
    return tuple(parts)


def client_split(idx: np.ndarray, client_id: int, seed: int = 42):
    """Lokale Aufteilung eines Clients in Train (85 %) und Val (15 %), wie in Notebook 01."""
    rng = np.random.default_rng(seed + client_id)
    idx = rng.permutation(idx)
    n_val = max(int(0.15 * len(idx)), 1)
    return idx[n_val:], idx[:n_val]


# --- ClientApp ----------------------------------------------------------------------
def make_client_app(partition: str, n_clients: int, alpha: float, by: str,
                    part_seed: int, hidden: tuple, dropout: float) -> ClientApp:
    app = ClientApp()

    def _local(context: Context):
        cid = int(context.node_config["partition-id"])
        idx = get_partitions(partition, n_clients, alpha, by, part_seed)[cid]
        tr_idx, va_idx = client_split(idx, cid, part_seed)
        d = _data()
        return cid, d["X_train"], d["y_train"], tr_idx, va_idx

    @app.train()
    def client_train(msg: Message, context: Context) -> Message:
        cid, X, y, tr_idx, _ = _local(context)
        cfg = msg.content["config"]
        rnd = int(cfg["server-round"])
        model = MLP(X.shape[1], hidden, dropout)
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        loss = train(model, X[tr_idx], y[tr_idx],
                     epochs=int(cfg["local-epochs"]), lr=float(cfg["lr"]),
                     batch_size=int(cfg["batch-size"]), optimizer=str(cfg["optimizer"]),
                     proximal_mu=float(cfg.get("proximal-mu", 0.0)),
                     seed=int(cfg["seed"]) * 100_000 + rnd * 1000 + cid)
        content = RecordDict({
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord({"train_loss": loss, "num-examples": len(tr_idx)}),
        })
        return Message(content=content, reply_to=msg)

    @app.evaluate()
    def client_evaluate(msg: Message, context: Context) -> Message:
        cid, X, y, _, va_idx = _local(context)
        model = MLP(X.shape[1], hidden, dropout)
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        m = evaluate(model, X[va_idx], y[va_idx])
        # ROC-AUC weglassen: bei einklassigen Clients undefiniert, Flower verlangt aber
        # bei allen Clients dieselben Schlüssel
        metrics = {k: float(m[k]) for k in ("loss", "accuracy", "balanced_acc", "f1")}
        metrics["num-examples"] = len(va_idx)
        return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)

    return app


# --- ServerApp ----------------------------------------------------------------------
STRATEGIES = {"fedavg": FedAvg, "fedprox": FedProx, "fedavgm": FedAvgM, "fedmedian": FedMedian}


def make_server_app(store: dict, strategy: str, n_clients: int, num_rounds: int,
                    fraction_train: float, fraction_evaluate: float, train_cfg: dict,
                    strategy_kwargs: dict, hidden: tuple, dropout: float,
                    model_seed: int) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        d = _data()
        set_seed(model_seed)
        model = MLP(d["X_train"].shape[1], hidden, dropout)  # gleiches Startmodell für alle

        def central_eval(server_round: int, arrays: ArrayRecord) -> MetricRecord:
            m = MLP(d["X_train"].shape[1], hidden, dropout)
            m.load_state_dict(arrays.to_torch_state_dict())
            out = {}
            for split in ("val", "test"):
                for k, v in evaluate(m, d[f"X_{split}"], d[f"y_{split}"]).items():
                    out[f"{split}_{k}"] = float(v)
            return MetricRecord(out)

        n_min = max(1, int(round(fraction_train * n_clients)))
        strat = STRATEGIES[strategy](
            fraction_train=fraction_train, fraction_evaluate=fraction_evaluate,
            min_train_nodes=n_min, min_evaluate_nodes=1, min_available_nodes=n_clients,
            **strategy_kwargs)
        result = strat.start(
            grid=grid, initial_arrays=ArrayRecord(model.state_dict()),
            num_rounds=num_rounds, train_config=ConfigRecord(train_cfg),
            evaluate_fn=central_eval)
        store["result"] = result
        if store.get("save_to") and result.train_metrics_clientapp:  # nur bei Client-Antworten
            # sofort speichern, noch bevor Ray herunterfährt (unter Windows kann
            # ray.shutdown() mit "access violation" abstürzen)
            _save(store, result, d["X_train"].shape[1], hidden, dropout)

    return app


def _to_frame(result) -> pd.DataFrame:
    rows = {}
    for rnd, mr in result.evaluate_metrics_serverapp.items():
        rows.setdefault(rnd, {}).update(dict(mr))
    for rnd, mr in result.train_metrics_clientapp.items():
        rows.setdefault(rnd, {}).update({f"clients_{k}": v for k, v in dict(mr).items()})
    for rnd, mr in result.evaluate_metrics_clientapp.items():
        rows.setdefault(rnd, {}).update({f"clients_val_{k}": v for k, v in dict(mr).items()})
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "round"
    return df.reset_index()


def _save(store: dict, result, in_dim: int, hidden: tuple, dropout: float) -> None:
    out = Path(store["save_to"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df = _to_frame(result)
    for k, v in {**store["meta"], "runtime_s": round(time.time() - store["t0"], 1)}.items():
        df[k] = v
    df.to_csv(out, index=False)
    m = MLP(in_dim, hidden, dropout)
    m.load_state_dict(result.arrays.to_torch_state_dict())
    torch.save(m.state_dict(), out.with_suffix(".pt"))


def load_model(path, in_dim: int = 89, hidden: tuple = (64, 32), dropout: float = 0.1) -> MLP:
    """Finales globales Modell eines gespeicherten Laufs (CSV- oder .pt-Pfad)."""
    m = MLP(in_dim, tuple(hidden), dropout)
    m.load_state_dict(torch.load(Path(path).with_suffix(".pt")))
    return m


def run_cached(path, force: bool = False, verbose: bool = True, **kw) -> pd.DataFrame:
    """Führt run_fl(**kw) in einem eigenen Python-Prozess aus und speichert nach `path` (.csv).

    - Existiert die CSV schon, wird sie geladen (force=True rechnet neu).
    - Eigener Prozess: Ray und Flower laufen nicht im Jupyter-Kernel. Ein Absturz von Ray
      beim Herunterfahren (bekannt unter Windows) betrifft nur den Hilfsprozess; die
      Ergebnisse sind zu diesem Zeitpunkt bereits gespeichert.
    """
    path = Path(path)
    if path.exists() and not force:
        return pd.read_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    cmd = [sys.executable, str(Path(__file__).resolve()), "--out", str(path),
           "--kwargs", json.dumps(kw)]
    proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent),
                          capture_output=True, text=True)
    if not path.exists():
        print(proc.stdout[-3000:], proc.stderr[-6000:], sep="\n")
        raise RuntimeError(f"Lauf {path.name} fehlgeschlagen (Exit-Code {proc.returncode}), Log oben")
    df = pd.read_csv(path)
    if verbose:
        note = "" if proc.returncode == 0 else f"  (Hilfsprozess Exit-Code {proc.returncode}, Ergebnis gesichert)"
        print(f"{path.stem}: {df.runtime_s.iloc[0]:.0f}s, Test-Acc letzte Runde "
              f"{df.test_accuracy.iloc[-1]:.4f}{note}")
    return df


def run_fl(partition: str = "iid", n_clients: int = 10, num_rounds: int = 20,
           local_epochs: int = 1, lr: float = 1e-3, batch_size: int = 64,
           optimizer: str = "adam", strategy: str = "fedavg", strategy_kwargs: dict | None = None,
           fraction_train: float = 1.0, fraction_evaluate: float = 0.0,
           alpha: float = 0.5, by: str = "age", part_seed: int = 42, seed: int = 42,
           hidden: tuple = (64, 32), dropout: float = 0.1, num_cpus_per_client: float = 1.0,
           return_model: bool = False, quiet: bool = True, save_to: str | None = None):
    """Eine FL-Simulation. Gibt DataFrame (Metriken pro Runde, Runde 0 = Startmodell)
    zurück, mit return_model=True zusätzlich das finale globale Modell.

    strategy: "fedavg" | "fedprox" | "fedavgm" | "fedmedian"
    strategy_kwargs: z. B. {"proximal_mu": 0.1} für FedProx,
                     {"server_momentum": 0.9} für FedAvgM
    fraction_evaluate > 0: Clients werten das globale Modell zusätzlich lokal aus
    """
    strategy_kwargs = dict(strategy_kwargs or {})
    train_cfg = {"local-epochs": local_epochs, "lr": lr, "batch-size": batch_size,
                 "optimizer": optimizer, "seed": seed}
    t0 = time.time()
    meta = dict(partition=partition, n_clients=n_clients, strategy=strategy,
                local_epochs=local_epochs, lr=lr, batch_size=batch_size,
                fraction_train=fraction_train, alpha=alpha, seed=seed, **strategy_kwargs)
    store: dict = {"save_to": save_to, "meta": meta, "t0": t0}
    # quiet: nur Fehler von Flower anzeigen (u. a. Hinweis, dass run_simulation zugunsten
    # von `flwr run` abgekündigt ist – für Notebook-Experimente bewusst weiter verwendet)
    logging.getLogger("flwr").setLevel(logging.ERROR if quiet else logging.INFO)
    run_simulation(
        server_app=make_server_app(store, strategy, n_clients, num_rounds, fraction_train,
                                   fraction_evaluate, train_cfg, strategy_kwargs, hidden,
                                   dropout, seed),
        client_app=make_client_app(partition, n_clients, alpha, by, part_seed, hidden, dropout),
        num_supernodes=n_clients,
        backend_config={"client_resources": {"num_cpus": num_cpus_per_client, "num_gpus": 0.0}},
    )
    if "result" not in store:
        raise RuntimeError("Simulation ohne Ergebnis beendet – Log oben prüfen")
    if not store["result"].train_metrics_clientapp:
        # Flower läuft weiter, auch wenn alle Clients abstürzen -> globales Modell bliebe untrainiert
        raise RuntimeError("Keine Client-Antworten erhalten – Fehler der ClientApps im Log oben prüfen")
    df = _to_frame(store["result"])
    df.attrs.update(meta, runtime_s=round(time.time() - t0, 1))
    if return_model:
        m = MLP(_data()["X_train"].shape[1], hidden, dropout)
        m.load_state_dict(store["result"].arrays.to_torch_state_dict())
        return df, m
    return df


if __name__ == "__main__":
    # Direktaufruf: Funktionen per Import laden, sonst findet Ray sie in den Workern nicht
    import argparse

    import fl

    ap = argparse.ArgumentParser(description="Einzelner FL-Lauf (wird von run_cached genutzt)")
    ap.add_argument("--out", help="Ziel-CSV")
    ap.add_argument("--kwargs", default="{}", help="Argumente für run_fl als JSON")
    a = ap.parse_args()
    kw = json.loads(a.kwargs)
    if "hidden" in kw:
        kw["hidden"] = tuple(kw["hidden"])
    if a.out:
        fl.run_fl(save_to=a.out, **kw)
    else:  # Schnelltest: python fl.py
        h = fl.run_fl(partition="iid", n_clients=5, num_rounds=5, local_epochs=1)
        print(h[["round", "val_accuracy", "test_accuracy", "test_f1"]].round(4))
