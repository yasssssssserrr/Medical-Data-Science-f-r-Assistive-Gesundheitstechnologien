# featureextraction.py

import numpy as np
import os
import argparse
from scipy.stats import skew, kurtosis

# === FUNKTION: Feature-Berechnung ===

def extract_features(X):
    """
    Berechnet 10 statistische Features für jeden Kanal eines Zeitreihen-Samples.
    Input:
        X: np.array mit Form (N, T, C)
    Output:
        Features: np.array mit Form (N, C * 10)
    """
    N, T, C = X.shape
    features = []

    for i in range(N):
        sample_features = []
        for c in range(C):
            signal = X[i, :, c]

            # 10 statistische Features
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            min_val = np.min(signal)
            max_val = np.max(signal)
            energy = np.sum(signal**2)
            skewness = skew(signal)
            kurt = kurtosis(signal)
            median = np.median(signal)
            signal_range = max_val - min_val
            zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)

            # Alle Features zum Sample-Feature-Vektor hinzufügen
            sample_features.extend([
                mean_val, std_val, min_val, max_val, energy,
                skewness, kurt, median, signal_range, zero_crossings
            ])
        features.append(sample_features)

    return np.array(features)


# === HAUPTPROGRAMM ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Extraction aus vorverarbeiteten Daten")
    parser.add_argument("--input", type=str, required=True, help="Pfad zur Datei mit vorverarbeiteten Daten (.npy)")
    parser.add_argument("--output", type=str, default="X_features.npy", help="Pfad zur Zieldatei mit Features")

    args = parser.parse_args()

    # 1. Daten laden
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"❌ Datei nicht gefunden: {args.input}")

    print("📥 Lade vorverarbeitete Daten...")
    X_preprocessed = np.load(args.input)
    print(f"✅ Eingabeform: {X_preprocessed.shape}  (Samples, Zeitpunkte, Kanäle)")

    # 2. Feature Extraction
    print("🧠 Extrahiere Features...")
    X_features = extract_features(X_preprocessed)
    print(f"✅ Feature-Matrix erzeugt: {X_features.shape}  (Samples, 150 Features)")

    # 3. Speichern
    np.save(args.output, X_features)
    print(f"💾 Features gespeichert in: {args.output}")
