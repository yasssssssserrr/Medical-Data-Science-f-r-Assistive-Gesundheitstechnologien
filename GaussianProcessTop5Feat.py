# GaussianProcessTop5Feat.py

import numpy as np
import os
import argparse
import logging
import time
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

# === Logging konfigurieren ===
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

# === Funktionen ===

def load_data(X_path, y_path):
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"❌ Feature-Datei nicht gefunden: {X_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"❌ Label-Datei nicht gefunden: {y_path}")
    X = np.load(X_path)
    y = np.load(y_path)
    return X, y

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1

def save_predictions(y_pred, output_path):
    np.save(output_path, y_pred)
    logging.info(f"📁 Vorhersagen gespeichert unter: {output_path}")

def main(args):
    logging.info("📥 Lade Trainings- und Testdaten...")
    X_train, y_train = load_data(args.X_train, args.y_train)
    X_test, y_test = load_data(args.X_test, args.y_test)

    logging.info(f"🔢 Trainingsdaten: {X_train.shape}, Labels: {y_train.shape}")
    logging.info(f"🔢 Testdaten:     {X_test.shape}, Labels: {y_test.shape}")

    logging.info("🧠 Initialisiere Gaussian Process Classifier...")
    clf = GaussianProcessClassifier()

    logging.info("🏋️‍♂️ Starte Training...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    logging.info(f"✅ Training abgeschlossen in {training_time:.2f} Sekunden.")

    logging.info("🎯 Starte Vorhersage...")
    y_pred = clf.predict(X_test)

    acc, f1 = evaluate_model(y_test, y_pred)
    logging.info(f"✅ Accuracy:        {acc:.4f}")
    logging.info(f"✅ F1-Score (macro): {f1:.4f}")

    # Vorhersagen speichern
    if args.output_preds:
        save_predictions(y_pred, args.output_preds)

    # Optional: Modell speichern
    if args.output_model:
        joblib.dump(clf, args.output_model)
        logging.info(f"💾 Modell gespeichert unter: {args.output_model}")

    # Optional: Konfusionsmatrix
    if args.confusion:
        cm = confusion_matrix(y_test, y_pred)
        np.set_printoptions(threshold=20, edgeitems=2)
        logging.info(f"📊 Konfusionsmatrix:\n{cm}")

# === Hauptteil ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian Process Classifier auf Top-5 Features")

    parser.add_argument("--X_train", required=True, help="Pfad zur Feature-Datei (Training)")
    parser.add_argument("--y_train", required=True, help="Pfad zu den Labels (Training)")
    parser.add_argument("--X_test", required=True, help="Pfad zur Feature-Datei (Test)")
    parser.add_argument("--y_test", required=True, help="Pfad zu den Labels (Test)")

    parser.add_argument("--output_preds", default="y_pred_top5.npy", help="Dateiname für gespeicherte Vorhersagen")
    parser.add_argument("--output_model", default=None, help="Optional: Speicherpfad für Modell (joblib)")
    parser.add_argument("--confusion", action="store_true", help="Optional: Konfusionsmatrix anzeigen")

    args = parser.parse_args()
    main(args)
