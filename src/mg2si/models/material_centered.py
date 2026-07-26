from dataclasses import dataclass

import numpy as np
import pandas as pd

from mg2si.models.botorch_surrogate import BotorchSurrogate


@dataclass
class StageModel:
    target: str
    features: list[str]
    surrogate: BotorchSurrogate


class MaterialCenteredModel:
    """p(Z|X) and p(Y|Z,C), with state uncertainty propagation."""

    def __init__(self, minimum_rows: int = 12) -> None:
        self.minimum_rows = minimum_rows
        self.state_models: dict[str, StageModel] = {}
        self.biology_models: dict[str, StageModel] = {}

    def fit(self, data: pd.DataFrame, x_fields: list[str], z_fields: list[str], context_fields: list[str]):
        for target in z_fields:
            usable = data[x_fields + [target]].dropna(subset=[target])
            if len(usable) >= self.minimum_rows:
                self.state_models[target] = StageModel(target, x_fields, BotorchSurrogate().fit(usable[x_fields], usable[target]))
        available_z = sorted(self.state_models)
        for target in ("y_tumor_viability_pct", "y_normal_viability_pct"):
            features = available_z + context_fields
            usable = data[features + [target]].dropna(subset=available_z + [target])
            if available_z and len(usable) >= self.minimum_rows:
                self.biology_models[target] = StageModel(target, features, BotorchSurrogate().fit(usable[features], usable[target]))
        return self

    @property
    def ready(self) -> bool:
        return bool(self.state_models) and len(self.biology_models) == 2

    def predict(self, candidates: pd.DataFrame, context_fields: list[str], samples: int = 256, seed: int = 0) -> dict:
        if not self.ready:
            raise RuntimeError("Material-centered model has insufficient stage coverage.")
        rng = np.random.default_rng(seed)
        states = {}
        for name, stage in self.state_models.items():
            mean, std = stage.surrogate.predict(candidates[stage.features])
            states[name] = rng.normal(mean, std, size=(samples, len(candidates)))
        output = {}
        for target, stage in self.biology_models.items():
            draws = []
            for index in range(samples):
                frame = candidates[context_fields].copy()
                for state_name in self.state_models:
                    frame[state_name] = states[state_name][index]
                mean, std = stage.surrogate.predict(frame[stage.features])
                draws.append(rng.normal(mean, std))
            values = np.asarray(draws)
            output[target] = {"mean": values.mean(axis=0), "std": values.std(axis=0), "draws": values}
        output["state"] = states
        return output

