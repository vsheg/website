# %%
from pathlib import Path

import pandas as pd
import torch as T
from sklearn.ensemble import GradientBoostingRegressor

# %% Generate data
x = T.linspace(-5, 5, 1024)
X = x.reshape(-1, 1)

y = x + 5 * x.cos() + T.normal(0, 1, size=x.shape)


# %% Predict mean
df = pd.DataFrame({"x": X.reshape(-1), "y": y.reshape(-1)})

model = GradientBoostingRegressor(
    loss="squared_error",
)
model.fit(X, y)
df["mean"] = model.predict(X)
# %% Predict quantiles
quantiles = [0.05, 0.5, 0.95]

for q in quantiles:
    model = GradientBoostingRegressor(loss="quantile", alpha=q)
    model.fit(X, y)
    df[q] = model.predict(X)

# %% Save results
path = Path(__file__).parent / "out.csv"
df.to_csv(path, index=False)
