
import logging
from pathlib import Path

from data_prep import load_dataset
from train_eval import run_experiments
from plots import plot_metrics, plot_detection_like

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def main():
    logging.info("Preparing dataset and preprocessing...")
    X_train_scaled, X_test_scaled, y_train, y_test, n_vals = load_dataset(OUT_DIR)

    logging.info("Running experiments (models & metrics)...")
    metrics_df, ann_logs = run_experiments(X_train_scaled, X_test_scaled, y_train, y_test, n_vals, OUT_DIR)

    logging.info("Generating plots (Figures 5-11)...")
    plot_metrics(metrics_df, OUT_DIR)
    plot_detection_like(X_train_scaled, X_test_scaled, y_train, y_test, metrics_df, OUT_DIR)

    logging.info("Pipeline complete. Results saved in %s", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
