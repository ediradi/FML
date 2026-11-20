"""Erster Test: Daten laden, prüfen, vorverarbeiten.

Erwartet data/adult.data und data/adult.test (aus adult.zip von
https://archive.ics.uci.edu/dataset/2/adult).

Aufruf:  python test_data.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data import TARGET, load_processed, load_raw

# 1. Rohdaten laden und mit den UCI-Kennzahlen vergleichen
train, test = load_raw()
print(f"Train: {train.shape}, Test: {test.shape}")

checks = {
    "Zeilen Train = 32561": len(train) == 32561,
    "Zeilen Test = 16281": len(test) == 16281,
    "Train >50K = 7841": int(train[TARGET].sum()) == 7841,
    "Test >50K = 3846": int(test[TARGET].sum()) == 3846,
    "Fehlwerte workclass = 1836 / 963": (train["workclass"].isna().sum(), test["workclass"].isna().sum()) == (1836, 963),
}
for name, ok in checks.items():
    print(f"  [{'OK' if ok else 'FEHLER'}] {name}")

print(f"\nAnteil >50K: Train {train[TARGET].mean():.3f}, Test {test[TARGET].mean():.3f}")
print("Fehlwerte Train:", train.isna().sum()[lambda s: s > 0].to_dict())
print("\nErste Zeilen:")
print(train.head())

# 2. Vorverarbeitung
X_tr, y_tr, X_te, y_te, names, *_ = load_processed()
print(f"\nNach Vorverarbeitung: X_train {X_tr.shape}, X_test {X_te.shape}")
print(f"NaN in X: {np.isnan(X_tr).sum() + np.isnan(X_te).sum()}")

# 3. Plausibilitätscheck: logistische Regression sollte ca. 0.85 erreichen
clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
acc = accuracy_score(y_te, clf.predict(X_te))
print(f"\nLogistische Regression, Test-Accuracy: {acc:.4f} (erwartet ca. 0.845)")

assert all(checks.values()), "Daten stimmen nicht mit UCI überein"
print("\nAlles in Ordnung.")
