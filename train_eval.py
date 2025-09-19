
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from models import build_bagging, build_adaboost, build_gb, build_ann


def vaf_score(y_true, y_pred):

    den = np.var(y_true)
    if den == 0:
        return 0.0
    num = np.var(y_true - y_pred)
    return max(0.0, (1 - num / den)) * 100.0


def eval_metrics(y_true, y_pred):

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    vaf = vaf_score(y_true, y_pred)
    return rmse, mae, vaf, r2


def run_experiments(X_train, X_test, y_train, y_test, n_vals, out_dir):

    results = []
    ann_logs = []

    for n in n_vals:
        # Bagging
        bag = build_bagging(n_estimators=n)
        bag.fit(X_train, y_train)
        bag_pred = bag.predict(X_test)
        bag_metrics = eval_metrics(y_test, bag_pred)

        # AdaBoost
        ada = build_adaboost(n_estimators=n)
        ada.fit(X_train, y_train)
        ada_pred = ada.predict(X_test)
        ada_metrics = eval_metrics(y_test, ada_pred)

        # GradientBoost
        gb = build_gb(n_estimators=n)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        gb_metrics = eval_metrics(y_test, gb_pred)

        # ANN: epochs loosely tied to n to keep runtime reasonable
        ann = build_ann(X_train.shape[1])
        epochs = max(5, min(60, int(n / 2)))
        es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=0)
        hist = ann.fit(X_train, y_train, validation_split=0.12, epochs=epochs, batch_size=64, callbacks=[es], verbose=0)
        ann_pred = ann.predict(X_test).flatten()
        ann_metrics = eval_metrics(y_test, ann_pred)

        results.append(
            {
                "n_estimators": n,
                "bag_rmse": float(bag_metrics[0]),
                "ada_rmse": float(ada_metrics[0]),
                "ann_rmse": float(ann_metrics[0]),
                "gb_rmse": float(gb_metrics[0]),
                "bag_mae": float(bag_metrics[1]),
                "ada_mae": float(ada_metrics[1]),
                "ann_mae": float(ann_metrics[1]),
                "gb_mae": float(gb_metrics[1]),
                "bag_vaf": float(bag_metrics[2]),
                "ada_vaf": float(ada_metrics[2]),
                "ann_vaf": float(ann_metrics[2]),
                "gb_vaf": float(gb_metrics[2]),
                "bag_r2": float(bag_metrics[3]),
                "ada_r2": float(ada_metrics[3]),
                "ann_r2": float(ann_metrics[3]),
                "gb_r2": float(gb_metrics[3]),
            }
        )

        ann_logs.append({"n_estimators": n, "ann_epochs_trained": len(hist.history["loss"]), "ann_best_val_loss": float(min(hist.history["val_loss"]))})

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(out_dir / "metrics_table.csv", index=False)
    pd.DataFrame(ann_logs).to_csv(out_dir / "training_logs.csv", index=False)

    return metrics_df, ann_logs
