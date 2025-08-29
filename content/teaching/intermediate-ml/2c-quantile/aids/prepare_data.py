# %% Import libraries and load the AIDS clinical trial dataset
from pathlib import Path

import pandas as pd
from sksurv.datasets import load_aids

## Load the data
X, y_frame = load_aids()
X = X.rename(
    columns=dict(
        tx="indinavir",
        txgrp="group",
        strat2="stratum",
        sex="male",
        raceth="race",
        ivdrug="narcotics",
        hemophil="hemophilia",
        karnof="karnofsky_score",
        cd4="cd4_cell_count",
        priorzdv="zidovudine_experience",
        age="age",
    ),
)

# %% Remove highly correlated features
X.drop(columns=["indinavir", "stratum"], inplace=True)

# %% Convert Karnofsky score to integer type
X.karnofsky_score = X.karnofsky_score.astype(int)

# %% Map race from numeric codes to meaningful categorical labels
race_map = {1: "caucasian", 2: "black", 3: "hispanic", 4: "asian", 5: "american"}
race_dtype = pd.CategoricalDtype(race_map.values())
X.race = X.race.astype(int).replace(race_map).astype(race_dtype)

# %% Convert sex column from numeric (1,2) to boolean (True/False)
# NOTE: In `sksurv`, 1 and 2 are used, but in other versions of the dataset, 0 and 1 are used
X.male = X.male.astype(int).replace({1: True, 2: False})

# %% Transform narcotics usage into ordered categorical variable
narcotics_map = {1: "never", 2: "currently", 3: "previously"}
narcotics_dtype = pd.CategoricalDtype(categories=narcotics_map.values(), ordered=True)

X.narcotics = (
    pd.to_numeric(X.narcotics, errors="coerce").replace(narcotics_map).astype(narcotics_dtype)
)

# %% Convert hemophilia indicator to boolean type
X.hemophilia = X.hemophilia.astype(int).astype(bool)

# %% Filter to relevant groups and apply descriptive labels
X.group = X.group.astype(int)
mask = X.group.isin([1, 2])
X = X[mask].copy()
group_map = {
    1: "baseline",
    2: "indinavir",
}
group_dtype = pd.CategoricalDtype(categories=group_map.values())
X.group = X.group.replace(group_map).astype(group_dtype)


# %% Perform one-hot encoding on categorical variables
X = pd.get_dummies(
    X,
    drop_first=True,
)

# %% Extract and prepare survival target variables (censoring status and time)
y_data = pd.DataFrame(y_frame)[mask].copy()
censored, y = y_data.values.T

y = y.astype(float)
censored = censored.astype(bool)

# %% Save the dataset to a CSV file
df = X.copy()
df["censored"] = censored
df["time"] = y

path = Path(__file__).parent
df.to_csv(path / "aids.csv", index=False)
