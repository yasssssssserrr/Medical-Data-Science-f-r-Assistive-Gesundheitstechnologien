# preprocessing.py

import numpy as np
import os
import argparse
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

# === INTERPOLATION ===

def interpolate_sensor(data, target_length):
    """
    Interpoliert jedes Zeitreihensample auf target_length.
    Input:  data shape (N, T_old, C)
    Output: shape (N, target_length, C)
    """
    N, T_old, C = data.shape
    interpolated = np.zeros((N, target_length, C))

    for i in range(N):
        for c in range(C):
            x_old = np.linspace(0, 1, T_old)
            x_new = np.linspace(0, 1, target_length)
            f = interp1d(x_old, data[i, :, c], kind='linear', fill_value="extrapolate")
            interpolated[i, :, c] = f(x_new)
    return interpolated

# === NORMALISIERUNG ===

def normalisiere_sensor(daten):
    """
    Z-Score Normalisierung pro Kanal (global über alle Zeitpunkte/Samples)
    """
    N, T, C = daten.shape
    reshaped = daten.reshape(-1, C)
    scaler = StandardScaler().fit(reshaped)
    normalisiert = scaler.transform(reshaped).reshape(N, T, C)
    return normalisiert

# === DOWNSAMPLING ===

def downsample(daten, faktor):
    """
    Reduziert Zeitauflösung durch Auswahl jedes n-ten Zeitpunkts
    """
    return daten[:, ::faktor, :]

# === GESAMTPROZESS ===

def verarbeite_alle_sensoren(datapfad, dateien, faktor, target_length):
    """
    Lade alle Sensoren, interpoliere auf target_length, normalisiere, merge, downsample
    """
    sensoren = []

    for name in dateien:
        pfad = os.path.join(datapfad, name)
        if not os.path.exists(pfad):
            raise FileNotFoundError(f"Datei nicht gefunden: {pfad}")
        data = np.load(pfad)  # (N, T, C)
        if data.shape[1] != target_length:
            data = interpolate_sensor(data, target_length)
        sensoren.append(data)

    # Normalisieren & Zusammenführen
    sensoren_norm = [normalisiere_sensor(s) for s in sensoren]
    merged = np.concatenate(sensoren_norm, axis=2)  # → (N, T, 15)

    return downsample(merged, faktor)

# === STARTPUNKT ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing mit Interpolation + Downsampling")
    parser.add_argument("--datadir", type=str, required=True, help="Pfad zum Ordner mit .npy-Dateien")
    parser.add_argument("--factor", type=int, default=2, help="Downsampling-Faktor")
    parser.add_argument("--target_length", type=int, default=800, help="Ziel-Zeitreihenlänge (z. B. 800)")

    args = parser.parse_args()

    SENSOR_FILES = [
        "trainAccelerometer.npy",
        "trainGravity.npy",
        "trainGyroscope.npy",
        "trainLinearAcceleration.npy",
        "trainJinsGyroscope.npy"
    ]

    print(f"📁 Lade Sensoren aus: {args.datadir}")
    print(f"🎯 Ziel-Zeitlänge (T): {args.target_length}")
    print(f"🔧 Downsampling-Faktor: {args.factor}")

    X = verarbeite_alle_sensoren(args.datadir, SENSOR_FILES, args.factor, args.target_length)

    out_path = os.path.join(args.datadir, f"X_preprocessed_interp{args.target_length}_ds{args.factor}.npy")
    np.save(out_path, X)

    print(f"✅ Gespeichert unter: {out_path}")
    print(f"📐 Shape: {X.shape}  (Samples, Zeitpunkte, 15 Kanäle)")
