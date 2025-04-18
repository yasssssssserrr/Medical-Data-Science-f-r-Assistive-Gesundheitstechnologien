# preprocessing.py

import numpy as np
import os
import argparse
from sklearn.preprocessing import StandardScaler

# === FUNKTIONEN ===

def lade_sensoren(datapfad, dateinamen):
    arrays = []
    for name in dateinamen:
        pfad = os.path.join(datapfad, name)
        if not os.path.exists(pfad):
            raise FileNotFoundError(f"Datei nicht gefunden: {pfad}")
        arrays.append(np.load(pfad))  # (N, T, C)
    return arrays

def normalisiere_sensor(daten):
    N, T, C = daten.shape
    reshaped = daten.reshape(-1, C)
    scaler = StandardScaler().fit(reshaped)
    normalisiert = scaler.transform(reshaped).reshape(N, T, C)
    return normalisiert

def downsample(daten, faktor):
    return daten[:, ::faktor, :]

def verarbeite_alle_sensoren(datapfad, dateien, faktor):
    sensoren = lade_sensoren(datapfad, dateien)
    sensoren_norm = [normalisiere_sensor(s) for s in sensoren]
    merged = np.concatenate(sensoren_norm, axis=2)  # (N, T, 15)
    return downsample(merged, faktor)

# === HAUPTAUFRUF ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing: Normalisieren & Downsampling")
    parser.add_argument("--factor", type=int, default=2, help="Downsampling-Faktor ")
    parser.add_argument("--datadir", type=str, required=True, help="Pfad zum Ordner mit .npy-Dateien")
    args = parser.parse_args()

    SENSOR_FILES = [
        "trainAccelerometer.npy",
        "trainGravity.npy",
        "trainGyroscope.npy",
        "trainLinearAcceleration.npy",
        "trainJinsGyroscope.npy"
    ]

    print(f"🔧 Downsampling-Faktor: {args.factor}")
    print(f"📁 Datenverzeichnis: {args.datadir}")

    X = verarbeite_alle_sensoren(args.datadir, SENSOR_FILES, args.factor)

    out_path = os.path.join(args.datadir, f"X_preprocessed_factor{args.factor}.npy")
    np.save(out_path, X)

    print(f"✅ Gespeichert unter: {out_path}")
    print(f"📐 Shape: {X.shape}  (Samples, Zeitpunkte, Kanäle)")
