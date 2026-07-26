import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer


class DirectBaseline:
    def __init__(self, seed: int = 20260726) -> None:
        self.imputer = SimpleImputer(strategy="median")
        self.model = ExtraTreesRegressor(n_estimators=256, min_samples_leaf=2, random_state=seed, n_jobs=-1)

    def fit(self, x, y) -> "DirectBaseline":
        self.model.fit(self.imputer.fit_transform(x), y)
        return self

    def predict(self, x) -> tuple[np.ndarray, np.ndarray]:
        transformed = self.imputer.transform(x)
        members = np.vstack([tree.predict(transformed) for tree in self.model.estimators_])
        return members.mean(axis=0), members.std(axis=0, ddof=1)

