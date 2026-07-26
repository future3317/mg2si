import numpy as np
import torch


class BotorchSurrogate:
    def __init__(self) -> None:
        self.model = None
        self.medians = None
        self.lower = None
        self.span = None

    def fit(self, x, y) -> "BotorchSurrogate":
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood

        values = np.asarray(x, dtype=float)
        finite_columns = [column[np.isfinite(column)] for column in values.T]
        self.medians = np.asarray([np.median(column) if len(column) else 0.0 for column in finite_columns])
        self.lower = np.asarray([np.min(column) if len(column) else 0.0 for column in finite_columns])
        upper = np.asarray([np.max(column) if len(column) else 0.0 for column in finite_columns])
        self.span = np.where(upper > self.lower, upper - self.lower, 1.0)
        filled = np.where(np.isfinite(values), values, self.medians)
        train_x = torch.as_tensor((filled - self.lower) / self.span, dtype=torch.double)
        train_y = torch.as_tensor(np.asarray(y, dtype=float).reshape(-1, 1), dtype=torch.double)
        self.model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(self.model.likelihood, self.model))
        return self

    def predict(self, x) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Surrogate has not been fitted.")
        values = np.asarray(x, dtype=float)
        values = np.where(np.isfinite(values), values, self.medians)
        values = (values - self.lower) / self.span
        with torch.no_grad():
            posterior = self.model.posterior(torch.as_tensor(values, dtype=torch.double))
        return posterior.mean.squeeze(-1).numpy(), posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1).numpy()
