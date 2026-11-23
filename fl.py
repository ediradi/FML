"""
Flower simulation for Adult Income.

Task 2:
- IID federated learning
- FedAvg
- Evaluation of the global model after every round
"""

from __future__ import annotations

from collections import OrderedDict

import flwr as fl
import numpy as np

from model import (
    MLP,
    evaluate,
    get_weights,
    set_weights,
    train,
)


# -----------------------------------------------------------
# Client
# -----------------------------------------------------------

class AdultClient(fl.client.NumPyClient):

    def __init__(
        self,
        cid: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        lr: float = 1e-3,
        local_epochs: int = 1,
        batch_size: int = 64,
    ):
        self.cid = cid
        self.X_train = X_train
        self.y_train = y_train

        self.lr = lr
        self.local_epochs = local_epochs
        self.batch_size = batch_size

        self.model = MLP(in_dim=X_train.shape[1])

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):

        set_weights(self.model, parameters)

        train(
            self.model,
            self.X_train,
            self.y_train,
            epochs=config.get("local_epochs", self.local_epochs),
            lr=config.get("lr", self.lr),
            batch_size=config.get("batch_size", self.batch_size),
            seed=int(self.cid),
        )

        return (
            get_weights(self.model),
            len(self.X_train),
            {},
        )

    def evaluate(self, parameters, config):

        set_weights(self.model, parameters)

        metrics = evaluate(
            self.model,
            self.X_train,
            self.y_train,
        )

        return (
            float(metrics["loss"]),
            len(self.X_train),
            {
                "accuracy": float(metrics["accuracy"]),
                "f1": float(metrics["f1"]),
            },
        )


# -----------------------------------------------------------
# Server-side evaluation
# -----------------------------------------------------------

def make_evaluate_fn(
    X_test,
    y_test,
):

    history = []

    def evaluate_fn(server_round, parameters, config):

        model = MLP(in_dim=X_test.shape[1])

        set_weights(model, parameters)

        metrics = evaluate(
            model,
            X_test,
            y_test,
        )

        history.append(
            {
                "round": server_round,
                **metrics,
            }
        )

        print(
            f"Round {server_round:3d} | "
            f"Acc={metrics['accuracy']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"AUC={metrics['roc_auc']:.4f}"
        )

        return metrics["loss"], {
            k: float(v)
            for k, v in metrics.items()
            if k != "loss"
        }

    return evaluate_fn, history


# -----------------------------------------------------------
# Simulation
# -----------------------------------------------------------

def run_fedavg(
    X_train,
    y_train,
    X_test,
    y_test,
    partitions,
    rounds: int = 20,
    local_epochs: int = 1,
    lr: float = 1e-3,
    batch_size: int = 64,
):

    n_clients = len(partitions)

    evaluate_fn, history = make_evaluate_fn(
        X_test,
        y_test,
    )

    def client_fn(context):

        cid = int(context.node_config["partition-id"])

        idx = partitions[cid]

        return AdultClient(
            cid=cid,
            X_train=X_train[idx],
            y_train=y_train[idx],
            lr=lr,
            local_epochs=local_epochs,
            batch_size=batch_size,
        ).to_client()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=lambda rnd: {
            "lr": lr,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
        },
    )

    backend_config = {
        "client_resources": {
            "num_cpus": 1,
        }
    }

    fl.simulation.run_simulation(
        client_fn=client_fn,
        num_supernodes=n_clients,
        config=fl.server.ServerConfig(
            num_rounds=rounds
        ),
        strategy=strategy,
        backend_config=backend_config,
    )

    return history