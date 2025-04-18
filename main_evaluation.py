# main_evaluation.py

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score, f1_score

def load(X_path, y_path):
    return np.load(X_path), np.load(y_path)

def evaluate_model(X_train, y_train, X_test, y_test):
    clf = GaussianProcessClassifier()
    start = time.time()
    clf.fit(X_train, y_train)
    duration = time.time() - start
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    return acc, f1, duration

def plot_results(df, save_path=None):
    models = df["Modell"]
    acc = df["Accuracy"]
    f1s = df["F1-Score"]
    time_ = df["Trainingszeit (s)"]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.bar(models, acc, label="Accuracy")
    plt.bar(models, f1s, label="F1-Score", alpha=0.6)
    plt.ylim(0, 1)
    plt.title("Accuracy & F1-Score")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(models, time_, color="gray")
    plt.title("Trainingszeit (Sekunden)")

    plt.suptitle("Modellvergleich – Top 5 vs. Alle Features")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # === Pfade definieren ===
    train_top5 = "PATH/TO/X_top5.npy"
    test_top5 = "PATH/TO/X_top5_test.npy"
    train_all = "PATH/TO/X_features_150.npy"
    test_all = "PATH/TO/X_features_150_test.npy"
    y_train = "PATH/TO/trainLabels.npy"
    y_test = "PATH/TO/testLabels.npy"

    # === Daten laden ===
    X_top5_train, y_tr = load(train_top5, y_train)
    X_top5_test, y_te = load(test_top5, y_test)
    X_all_train, _ = load(train_all, y_train)
    X_all_test, _ = load(test_all, y_test)

    # === Top 5 Features ===
    print("🔎 Starte Top-5 Modell...")
    acc1, f1_1, time1 = evaluate_model(X_top5_train, y_tr, X_top5_test, y_te)

    # === Alle Features ===
    print("🔎 Starte All-Feature Modell...")
    acc2, f1_2, time2 = evaluate_model(X_all_train, y_tr, X_all_test, y_te)

    # === Ergebnisse zusammenstellen ===
    df = pd.DataFrame({
        "Modell": ["Top-5 Features", "Alle 150 Features"],
        "Features": [5, 150],
        "Accuracy": [acc1, acc2],
        "F1-Score": [f1_1, f1_2],
        "Trainingszeit (s)": [time1, time2]
    })

    # === Ausgabe speichern ===
    df.to_csv("evaluation_results.csv", index=False)
    print("\n📄 Evaluation abgeschlossen. Ergebnisse gespeichert in: evaluation_results.csv")
    print(df)

    # === Diagramm anzeigen
    plot_results(df, save_path="vergleich.png")
