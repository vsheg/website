# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor

# %%
np.random.seed(5)
N_POINTS = 128

x = np.linspace(-7, 7, N_POINTS)
y = x
uniform_noise = np.random.uniform(-4, 4, N_POINTS)

# %%


# %%
df = pd.DataFrame(
    {
        "x": x,
        "y": y,
        "y_uniform_1": y + 3 * uniform_noise,
        "y_uniform_2": y + uniform_noise**3,
        "y_uniform_3": y
        + uniform_noise * (uniform_noise >= 0)
        + 4 * uniform_noise * (uniform_noise < 0),
    }
)

# %%
for col in df.columns[2:]:
    X = x.reshape(-1, 1)

    model_ls = LinearRegression()
    model_ls.fit(X, df[col])
    df[f"{col}_pred_ls"] = model_ls.predict(X)

    model_qr = QuantileRegressor(quantile=0.5, alpha=0, solver="highs-ds")
    model_qr.fit(X, df[col])
    df[f"{col}_pred_qr"] = model_qr.predict(X)


# %%
parent = Path(__file__).parent
df.to_csv(parent / "out.csv", index=False)
