# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, QuantileRegressor
from tqdm import tqdm

# %% Load the data
parent = Path(__file__).parent
df = pd.read_csv(parent / "aids.csv")

# %% Remove censored observations for simplicity
censored = df.pop("censored")
X = df[~censored].copy()
y = X.pop("time")

# %% Apply linear regression to select features
# NOTE: Linear regression is much faster than quantile regression and we can use simple metrics
# to measure the model performance
model_mean = LinearRegression()
sfs = SequentialFeatureSelector(
    model_mean,
    direction="backward",
    scoring="r2",
    cv=5,
    n_jobs=-1,
    n_features_to_select="auto",
    tol=0.0,
)
sfs.fit(X, y)

mask_features = X.columns[sfs.get_support()]
X = X[mask_features].copy()

# %% Run linear quantile regression and extract coefficients
coeffs_dict = {}
quantiles = np.linspace(0.1, 0.9, 20)

for q in tqdm(quantiles):
    model = QuantileRegressor(quantile=q, alpha=0.0)
    model.fit(X, y)
    coeffs_dict[q] = [model.intercept_] + model.coef_.tolist()

df_coeffs = pd.DataFrame(coeffs_dict, index=["intercept"] + list(X.columns)).T
column_order = df_coeffs.mean().sort_values(ascending=False).index
df_coeffs = df_coeffs[column_order]
df_coeffs.index.name = "quantile"

# %% Save
df_coeffs.to_csv(parent / "out.csv")
