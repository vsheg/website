# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor

# %% Generate noisy data
np.random.seed(1)
x = np.linspace(0, 1, 512)
X = x.reshape(-1, 1)
y = x + np.random.normal(0, 0.2 + x, 512) + np.random.pareto(3, 512)


# %% Train OLS model
model = LinearRegression()
model.fit(X, y)

df = pd.DataFrame({"x": x, "y": y, "mean": model.predict(X)})


# %% Train quantile model
quantiles = np.array([0.95, 0.75, 0.5, 0.25, 0.05])

for q in quantiles:
    model = QuantileRegressor(quantile=q, alpha=0.0)
    model.fit(X, y)
    df[q] = model.predict(X)

# %% Save results
parent = Path(__file__).parent
df.to_csv(parent / "out.csv", index=False)
