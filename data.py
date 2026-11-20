"""Laden und Vorverarbeiten des UCI Adult (Census Income) Datensatzes.

Quelle: https://archive.ics.uci.edu/dataset/2/adult  (48.842 Instanzen, 14 Features)
Offizieller Split: 32.561 Train (adult.data) / 16.281 Test (adult.test).

Unterstützte Eingaben (in dieser Reihenfolge gesucht):
  1. data/adult.data + data/adult.test   (Original-Dateien aus adult.zip)
  2. data/adult-all.csv                  (beide Dateien hintereinander, ohne Header)
  3. ucimlrepo.fetch_ucirepo(id=2)       (Download, falls Netz verfügbar)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).parent / "data"
N_TRAIN = 32561  # Zeilen in adult.data

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
# "education" wird verworfen: 1:1-redundant zu "education-num"
CATEGORICAL = ["workclass", "marital-status", "occupation", "relationship",
               "race", "sex", "native-country"]
TARGET = "income"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].str.strip()
    df = df.replace("?", np.nan)
    # adult.test hat Labels mit Punkt (">50K.")
    df[TARGET] = df[TARGET].str.rstrip(".").map({"<=50K": 0, ">50K": 1}).astype(int)
    return df


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gibt (train, test) im offiziellen Split zurück, bereinigt, Label 0/1."""
    data, test = DATA_DIR / "adult.data", DATA_DIR / "adult.test"
    allcsv = DATA_DIR / "adult-all.csv"
    if data.exists() and test.exists():
        tr = pd.read_csv(data, header=None, names=COLUMNS, skipinitialspace=True)
        te = pd.read_csv(test, header=None, names=COLUMNS, skipinitialspace=True, skiprows=1)
    elif allcsv.exists():
        df = pd.read_csv(allcsv, header=None, names=COLUMNS, skipinitialspace=True)
        tr, te = df.iloc[:N_TRAIN], df.iloc[N_TRAIN:]
    else:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=2)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        df.columns = COLUMNS
        tr, te = df.iloc[:N_TRAIN], df.iloc[N_TRAIN:]
    tr, te = _clean(tr.dropna(how="all")), _clean(te.dropna(how="all"))
    assert len(tr) == 32561 and len(te) == 16281, (len(tr), len(te))
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def make_preprocessor() -> ColumnTransformer:
    """Fehlwerte -> häufigste Kategorie, One-Hot für Kategorien, Standardisierung numerisch.
    capital-gain/-loss werden vorher log1p-transformiert (stark schief)."""
    num = Pipeline([("scale", StandardScaler())])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def _log_capital(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[["capital-gain", "capital-loss"]] = np.log1p(df[["capital-gain", "capital-loss"]])
    return df


def load_processed():
    """X_train, y_train, X_test, y_test (float32/int64), Feature-Namen, sowie die
    Roh-DataFrames (für attributbasierte non-IID-Partitionierung).

    Hinweis zu FL: Der Preprocessor wird hier global auf dem Trainingsset gefittet
    (gemeinsames Feature-Schema für alle Clients). In einem strikten FL-Setting
    müssten Kategorien-Vokabular und Skalierungsstatistiken vorab vereinbart bzw.
    föderiert berechnet werden – im Report als Annahme nennen.
    """
    tr, te = load_raw()
    tr_l, te_l = _log_capital(tr), _log_capital(te)
    pre = make_preprocessor()
    X_tr = pre.fit_transform(tr_l).astype(np.float32)
    X_te = pre.transform(te_l).astype(np.float32)
    y_tr = tr[TARGET].to_numpy(np.int64)
    y_te = te[TARGET].to_numpy(np.int64)
    return X_tr, y_tr, X_te, y_te, list(pre.get_feature_names_out()), tr, te


def load_dataset(val_size: float = 0.15, seed: int = 42) -> dict:
    """Train/Val/Test für die Experimente.

    - Test = offizielles adult.test, wird nur für die finale Auswertung verwendet
    - Val  = stratifizierter Anteil von adult.data, für Tuning und Early Stopping
    - Der Preprocessor wird nur auf dem Train-Anteil gefittet (kein Leakage aus Val/Test)

    Rückgabe: dict mit X_train, y_train, X_val, y_val, X_test, y_test (numpy),
    feature_names sowie den Roh-DataFrames df_train, df_val, df_test
    (für attributbasierte Partitionierung, z. B. nach Alter).
    """
    tr, te = load_raw()
    tr, va = train_test_split(tr, test_size=val_size, stratify=tr[TARGET], random_state=seed)
    tr, va = tr.reset_index(drop=True), va.reset_index(drop=True)
    pre = make_preprocessor()
    X = {
        "train": pre.fit_transform(_log_capital(tr)),
        "val": pre.transform(_log_capital(va)),
        "test": pre.transform(_log_capital(te)),
    }
    out = {f"X_{k}": v.astype(np.float32) for k, v in X.items()}
    out.update({f"y_{k}": d[TARGET].to_numpy(np.int64)
                for k, d in (("train", tr), ("val", va), ("test", te))})
    out.update(feature_names=list(pre.get_feature_names_out()),
               df_train=tr, df_val=va, df_test=te)
    return out


if __name__ == "__main__":
    tr, te = load_raw()
    print("Train", tr.shape, "Test", te.shape)
    print("Positivrate Train %.3f, Test %.3f" % (tr[TARGET].mean(), te[TARGET].mean()))
    print("Fehlwerte Train:", tr.isna().sum()[lambda s: s > 0].to_dict())
    print("Fehlwerte Test: ", te.isna().sum()[lambda s: s > 0].to_dict())
    X_tr, y_tr, X_te, y_te, names, *_ = load_processed()
    print("Features nach Encoding:", X_tr.shape[1])
