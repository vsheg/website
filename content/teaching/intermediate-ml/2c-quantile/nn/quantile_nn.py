# %%
from pathlib import Path

import lightning as L
import pandas as pd
import torch as T

# %% Generate data
x = T.linspace(-5, 5, 1024)
X = x.reshape(-1, 1)

y = x + 5 * x.cos() + T.normal(0, 1, size=x.shape)
y = y.reshape(-1, 1)


# %%
class QuantileLoss(L.LightningModule):
    def __init__(self, q: float):
        super().__init__()
        self.q = q

    def forward(self, y_pred, y_true):
        return T.where(
            (epsilon := y_true - y_pred) >= 0,
            self.q * epsilon,
            (self.q - 1) * epsilon,
        ).mean()


# %% Define simple model
class Model(L.LightningModule):
    def __init__(self, q: float | None = None):
        super().__init__()

        self.model = T.nn.Sequential(
            T.nn.LazyLinear(64),
            T.nn.GELU(),
            T.nn.LazyLinear(64),
            T.nn.GELU(),
            T.nn.LazyLinear(1),
        )

        self.loss = QuantileLoss(q) if q else T.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        return loss

    def train_dataloader(self):
        dataset = T.utils.data.TensorDataset(X, y)
        return T.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)

    def configure_optimizers(self):
        return T.optim.NAdam(self.parameters(), lr=1e-2)


# %% Predict mean
df = pd.DataFrame({"x": X.reshape(-1), "y": y.reshape(-1)})

model = Model()
trainer = L.Trainer(max_epochs=100)
trainer.fit(model)
df["mean"] = model(X).detach()

# %% Predict quantiles
quantiles = [0.05, 0.5, 0.95]

for q in quantiles:
    model = Model(q=q)
    trainer = L.Trainer(max_epochs=100)
    trainer.fit(model)
    df[q] = model(X).detach()

# %% Save results
path = Path(__file__).parent / "out.csv"
df.to_csv(path, index=False)
