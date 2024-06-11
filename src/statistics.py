import warnings
from typing import Callable, List, Literal, Tuple, Union
import numpy as np
import sklearn.metrics as sk_metrics

THRESHOLD_METHODS = [
    "balanced",
    "youden",
    "f-score",
    "sensitivity",
    "specificity",
    "ppv",
]
THRESHOLD_METHODS_ALIAS = Literal[tuple(THRESHOLD_METHODS)]


def compute_optimal_th(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
    method: THRESHOLD_METHODS_ALIAS = "youden",
    beta: float = 1,
    operating_point: float = 0.8,
) -> float:
    """Calculates the optimal threshold to discriminate between 2 classes.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predictions
    method : str (default='youden')
        Name of the cutoff strategy.
        Options: ('youden', 'balanced', 'f-score', 'sensitivity', 'specificity', 'ppv')
    beta : float (default=1)
        Controls the balance between precision and recall for 'f-score'.
        The higher the beta the more recall is weighted.
    operating_point : float (default=0.8)
        Used when `method` == 'sensitivity', 'specificity', or 'ppv'.
        Fixed operating point to select the optimal threshold.
        An operating_point of 0.8 with `method='sensitivity'` will return the
        threshold with the highest specificity at a fixed sensitivity of 0.8

    Returns
    -------
    float
        Optimal threshold
    """
    if method == "balanced":
        fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_pred)
        optimal_idx = np.argmin(np.abs(tpr - (1 - fpr)))
    elif method == "youden":
        fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_pred)
        optimal_idx = np.argmax(tpr - fpr)
    elif method == "f-score":
        precision, recall, thresholds = sk_metrics.precision_recall_curve(
            y_true, y_pred
        )
        f_beta = (
            (1 + beta ** 2)
            * (precision * recall)
            / ((beta ** 2 * precision) + recall + 1e-7)
        )
        optimal_idx = np.argmax(f_beta)
    elif method in ["sensitivity", "specificity"]:
        if operating_point is None:
            raise ValueError(
                f"operating_point must be specified when method='{method}'"
            )
        fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_pred)
        if method == "sensitivity":
            optimal_idx = np.argmin(np.abs(tpr - operating_point))
        else:
            optimal_idx = np.argmin(np.abs(1 - fpr - operating_point))
    elif method == "ppv":
        if operating_point is None:
            raise ValueError(
                f"operating_point must be specified when method='{method}'"
            )
        precision, _, thresholds = sk_metrics.precision_recall_curve(y_true, y_pred)
        optimal_idx = np.argmin(np.abs(precision[:-1] - operating_point))
    else:
        raise ValueError(f"Unrecognized threshold method '{method}'")

    return float(thresholds[optimal_idx])


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score_fn: Callable,
    alpha: float = 0.95,
    num_boot: int = 1000,
    seed: int = 42069,
) -> Tuple[float, float]:
    """Computes confidence intervals using non-parametric bootstrap

    Parameters
    ----------
    y_true : np.ndarray
        Ground true labels
    y_pred : np.ndarray
        Predictions
    score_fn : callable
        Scoring function to calculate the bootstrapped confidence interval over.
        It will be called with `score_fn(y_true, y_pred)`
    num_boot : int (default=1000)
        Number of bootstrap iterations.
    alpha : float (default=0.95)
        Confidence interval (0.95 = 95%)
    seed : int (default=42069)
        Random state

    Returns
    -------
    float
        Lower bound of the confidence interval
    float
        Upper bound of the confidence interval
    """
    random_state = np.random.RandomState(seed)
    num_samples = len(y_pred)
    sided_prop = (1 - alpha) / 2

    bootstrapped_scores = []
    for _ in range(num_boot):
        indices = random_state.randint(0, num_samples, num_samples)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = score_fn(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    # sort the bootstrapped scores
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    num_scores = len(sorted_scores)
    if len(sorted_scores) < num_boot:
        warnings.warn(
            f"Estimated using {num_scores} / {num_boot} bootstrapp iterations"
        )

    # sample the lower and upper bounds of the bootstrapped scores
    confidence_lower = sorted_scores[int(sided_prop * num_scores)]
    confidence_upper = sorted_scores[int(1 - sided_prop * num_scores)]
    return (confidence_lower, confidence_upper)
