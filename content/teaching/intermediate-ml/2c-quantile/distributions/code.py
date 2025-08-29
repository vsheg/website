from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


x = np.linspace(-5, 5, 128)
y = 2 * sigmoid(x) - 1


df = pd.DataFrame(
    {
        "x": x,
        "y": y,
        "epsilon_normal": np.random.normal(0, 1, 128),
        "epsilon_pareto": np.random.pareto(3, 128),
        "epsilon_laplace": np.random.laplace(0, 1, 128),
    }
)

parent = Path(__file__).parent
df.to_csv(parent / "out.csv", index=False)
