# GaussianProcessAllFeat.py

import numpy as np
import os
import argparse
import logging
import time
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

def load_data(X_path, y_path):
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"❌ Datei nicht gefunden: {X_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"❌ Datei nicht gefunden: {y_path}")
    return np.load(X_path), np.load(y_path)

def evaluate(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')

def main(args):
    logging.info("📥 Lade Daten...")
    X_train, y_train = load_data(args.X_train, args.y_train)
    X_test, y_test = load_data(args.X_test, args.y_test)

    clf = GaussianProcessClassifier()
    logging.info("🧠 Trainiere Gaussian Process Classifier auf allen Features...")
    start = time.time()
    clf.fit(X_train, y_train)
    runtime = time.time() - start

    logging.info("🎯 Vorhersage läuft...")
    y_pred = clf.predict(X_test)

    acc, f1 = evaluate(y_test, y_pred)
    logging.info(f"✅ Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | Trainingszeit: {runtime:.2f}s")

    if args.save_model:
        joblib.dump(clf, args.save_model)
        logging.info(f"💾 Modell gespeichert unter: {args.save_model}")

    if args.save_preds:
        np.save(args.save_preds, y_pred)
        logging.info(f"📁 Vorhersagen gespeichert unter: {args.save_preds}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X_train", required=True)
    parser.add_argument("--y_train", required=True)
    parser.add_argument("--X_test", required=True)
    parser.add_argument("--y_test", required=True)
    parser.add_argument("--save_model", default=None)
    parser.add_argument("--save_preds", default=None)
    args = parser.parse_args()
    main(args)
