import numpy as np
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    rank_true = np.argsort(np.argsort(truth))
    rank_pred = np.argsort(np.argsort(pred))
    return {
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(mean_squared_error(truth, pred) ** 0.5),
        "spearman": float(np.corrcoef(rank_true, rank_pred)[0, 1]) if len(truth) > 1 else float("nan"),
    }


def interval_coverage(y_true, lower, upper) -> float:
    truth = np.asarray(y_true, dtype=float)
    return float(np.mean((truth >= np.asarray(lower)) & (truth <= np.asarray(upper))))


def safety_brier(y_true, probabilities, threshold: float) -> float:
    return float(brier_score_loss((np.asarray(y_true, dtype=float) >= threshold).astype(int), probabilities))

