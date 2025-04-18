# featureselection.py

import numpy as np
import os
import argparse
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def select_top_k_features(X, y, k=5):
    """
    Wählt die Top-k wichtigsten Features basierend auf Mutual Information.
    Rückgabe:
        - X_reduced: reduzierte Feature-Matrix (N, k)
        - indices: Liste der ausgewählten Feature-Indices
    """
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)
    indices = selector.get_support(indices=True)
    return X_new, indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Selection")
    parser.add_argument("--X", type=str, required=True, help="Pfad zu X_features.npy")
    parser.add_argument("--y", type=str, required=True, help="Pfad zu den Labels (z. B. trainLabels.npy)")
    parser.add_argument("--k", type=int, default=5, help="Anzahl der Top-Features")
    parser.add_argument("--output", type=str, default="X_top5.npy", help="Ausgabedatei für reduzierte Features")
    parser.add_argument("--save_indices", type=str, default="selected_features_indices.npy", help="Datei für die Feature-Indizes")

    args = parser.parse_args()

    # 1. Daten laden
    if not os.path.exists(args.X):
        raise FileNotFoundError(f"❌ Feature-Datei nicht gefunden: {args.X}")
    if not os.path.exists(args.y):
        raise FileNotFoundError(f"❌ Label-Datei nicht gefunden: {args.y}")

    X = np.load(args.X)
    y = np.load(args.y)

    print(f"📥 Eingeladen: {X.shape[1]} Features, {X.shape[0]} Samples")

    # 2. NaN-Werte behandeln
    if np.isnan(X).any():
        print("⚠️ Warnung: Feature-Matrix enthält NaNs – sie werden durch 0 ersetzt.")
        X = np.nan_to_num(X, nan=0.0)

    # 3. Feature Selection
    X_reduced, selected_indices = select_top_k_features(X, y, k=args.k)

    print(f"✅ Top-{args.k} Features ausgewählt: Indizes = {selected_indices}")
    print(f"🔢 Neue Shape: {X_reduced.shape}")

    # 4. Speichern
    np.save(args.output, X_reduced)
    np.save(args.save_indices, selected_indices)

    print(f"💾 Reduzierte Feature-Matrix gespeichert unter: {args.output}")
    print(f"💾 Feature-Indizes gespeichert unter: {args.save_indices}")
