
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import numpy as np
import pandas as pd


def plot_series(x, series_dict, title, ylabel, path):
    plt.figure(figsize=(8, 5))
    for label, y in series_dict.items():
        plt.plot(x, y, marker="o", label=label)
    plt.title(title)
    plt.xlabel("n_estimators")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_metrics(metrics_df, out_dir):
    x = metrics_df["n_estimators"].values

    plot_series(
        x,
        {
            "Bagging": metrics_df["bag_rmse"],
            "AdaBoost": metrics_df["ada_rmse"],
            "ANN": metrics_df["ann_rmse"],
            "GradientBoost": metrics_df["gb_rmse"],
        },
        "Figure 5 — RMSE vs n_estimators",
        "RMSE",
        out_dir / "figure5_rmse.png",
    )

    plot_series(
        x,
        {
            "Bagging": metrics_df["bag_mae"],
            "AdaBoost": metrics_df["ada_mae"],
            "ANN": metrics_df["ann_mae"],
            "GradientBoost": metrics_df["gb_mae"],
        },
        "Figure 6 — MAE vs n_estimators",
        "MAE",
        out_dir / "figure6_mae.png",
    )

    plot_series(
        x,
        {
            "Bagging": metrics_df["bag_vaf"],
            "AdaBoost": metrics_df["ada_vaf"],
            "ANN": metrics_df["ann_vaf"],
            "GradientBoost": metrics_df["gb_vaf"],
        },
        "Figure 7 — VAF (%) vs n_estimators",
        "VAF (%)",
        out_dir / "figure7_vaf.png",
    )

    plot_series(
        x,
        {
            "Bagging": metrics_df["bag_r2"],
            "AdaBoost": metrics_df["ada_r2"],
            "ANN": metrics_df["ann_r2"],
            "GradientBoost": metrics_df["gb_r2"],
        },
        "Figure 8 — R² vs n_estimators",
        "R²",
        out_dir / "figure8_r2.png",
    )


def plot_detection_like(X_train, X_test, y_train, y_test, metrics_df, out_dir, threshold=0.5):

    x = metrics_df["n_estimators"].values
    detection_rates = []
    accuracies = []
    false_alarm_rates = []

    for n in x:
        model = GradientBoostingRegressor(n_estimators=int(n), learning_rate=0.1, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_bin = (preds >= threshold).astype(int)
        true_bin = (y_test >= threshold).astype(int)

        acc = accuracy_score(true_bin, preds_bin)
        recall = recall_score(true_bin, preds_bin, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(true_bin, preds_bin, labels=[0, 1]).ravel()
        fa = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        detection_rates.append(recall)
        accuracies.append(acc)
        false_alarm_rates.append(fa)

    plot_series(x, {"GradientBoost_detection_rate": detection_rates}, "Figure 9 — Detection Rate vs n_estimators", "Detection Rate (Recall)", out_dir / "figure9_detection_rate.png")

    plot_series(x, {"GradientBoost_accuracy": accuracies}, "Figure 10 — Accuracy vs n_estimators", "Accuracy", out_dir / "figure10_accuracy.png")

    plot_series(x, {"GradientBoost_false_alarm_rate": false_alarm_rates}, "Figure 11 — False Alarm Rate vs n_estimators", "False Alarm Rate", out_dir / "figure11_false_alarm.png")
