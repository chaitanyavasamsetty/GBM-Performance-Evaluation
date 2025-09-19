

from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def build_bagging(n_estimators=50):

    return BaggingRegressor(
        base_estimator=DecisionTreeRegressor(max_depth=6),
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )


def build_adaboost(n_estimators=50):

    return AdaBoostRegressor(
        base_estimator=DecisionTreeRegressor(max_depth=4),
        n_estimators=n_estimators,
        random_state=42,
    )


def build_gb(n_estimators=50):
    return GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=0.1, max_depth=4, random_state=42)


def build_ann(input_dim):
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(input_dim,)),
            Dropout(0.1),
            Dense(32, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model
