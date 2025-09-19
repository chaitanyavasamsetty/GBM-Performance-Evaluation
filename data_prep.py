
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_dataset(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, size=n)
    gender = rng.choice([0, 1], size=n)  # 0 female, 1 male
    heart_rate = rng.normal(72, 8, size=n) + (age - 50) * 0.05 + gender * 1.2
    spo2 = np.clip(rng.normal(97, 1, size=n) - (age - 60) * 0.02, 80, 100)
    body_temp = rng.normal(36.6, 0.3, size=n) + 0.01 * (rng.random(n) - 0.5)
    steps = np.abs(rng.normal(4000, 2500, size=n).astype(int))
    resp_rate = np.clip(rng.normal(16, 2, size=n) + (age - 50) * 0.02, 8, 40)

    # Activity: categorical 0..3 (rest, light, moderate, vigorous)
    activity = rng.choice([0, 1, 2, 3], size=n, p=[0.35, 0.35, 0.2, 0.1])

    # Construct a continuous target 'score' in [0,1] from features (non-linear combination)
    base = (
        0.8
        - 0.002 * (age - 40)
        - 0.003 * (heart_rate - 70) ** 1.1 / 10
        + 0.004 * (spo2 - 95)
        + 0.00002 * np.sqrt(np.maximum(steps, 0))
        - 0.01 * (activity == 0).astype(float)
        + 0.02 * (activity == 2).astype(float)
    )

    noise = rng.normal(0, 0.03, size=n)
    score = np.clip(base + 0.01 * np.sin(heart_rate / 10) + noise, 0, 1)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "heart_rate": heart_rate,
            "spo2": spo2,
            "body_temp": body_temp,
            "steps": steps,
            "resp_rate": resp_rate,
            "activity": activity,
            "score": score,
        }
    )
    return df


def load_dataset(out_dir, csv_path=None):
    out_dir = out_dir
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        df = generate_dataset()
        df.to_csv(out_dir / "dataset.csv", index=False)

    # One-hot encode activity and prepare X,y
    X = pd.get_dummies(df.drop(columns=["score"]), columns=["activity"], prefix="act", drop_first=True)
    y = df["score"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_values = list(range(10, 101, 10))
    return X_train_scaled, X_test_scaled, y_train, y_test, n_values
