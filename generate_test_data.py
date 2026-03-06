"""テスト用サンプルデータを生成する。"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 200

data = {
    "id": range(1, n + 1),
    "name": [f"user_{i}" for i in range(1, n + 1)],
    "age": np.concatenate([
        np.random.normal(25, 3, 70),
        np.random.normal(40, 5, 70),
        np.random.normal(60, 4, 60),
    ]).astype(int),
    "income": np.concatenate([
        np.random.normal(300, 50, 70),
        np.random.normal(600, 80, 70),
        np.random.normal(900, 100, 60),
    ]),
    "signup_date": pd.date_range("2020-01-01", periods=n, freq="3D").strftime("%Y-%m-%d"),
    "region": np.random.choice(["North", "South", "East", "West"], n),
    "plan": np.random.choice(["Free", "Basic", "Premium"], n, p=[0.5, 0.3, 0.2]),
    "score": np.concatenate([
        np.random.normal(50, 10, 70),
        np.random.normal(75, 8, 70),
        np.random.normal(90, 5, 60),
    ]),
    "constant_col": 1,
}

# 欠損を一部に挿入
df = pd.DataFrame(data)
df.loc[np.random.choice(n, 10, replace=False), "income"] = np.nan

df.to_csv("test_data.csv", index=False)
print("test_data.csv generated.")
