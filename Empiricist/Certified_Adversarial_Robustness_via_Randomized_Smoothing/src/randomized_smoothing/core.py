from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, binomtest, norm

ABSTAIN = -1


def clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def max_certifiable_radius(sigma: float, trials: int, alpha: float) -> float:
    return float(sigma * norm.ppf(clopper_pearson_lower(trials, trials, alpha)))


def certified_radius(sigma: float, top_probability_lower: float) -> float:
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if not 0.5 < top_probability_lower <= 1:
        raise ValueError("a positive certificate requires 0.5 < p_A <= 1")
    return float(sigma * norm.ppf(top_probability_lower))


@dataclass(frozen=True)
class PredictionResult:
    prediction: int
    counts: np.ndarray
    p_value: float

    @property
    def abstained(self) -> bool:
        return self.prediction == ABSTAIN


@dataclass(frozen=True)
class CertificationResult:
    prediction: int
    radius: float
    selected_class: int
    selection_counts: np.ndarray
    estimation_counts: np.ndarray
    lower_probability: float

    @property
    def abstained(self) -> bool:
        return self.prediction == ABSTAIN


def predict_from_counts(counts: np.ndarray, alpha: float) -> PredictionResult:
    counts = np.asarray(counts)
    top_two = np.argsort(counts)[-2:][::-1]
    first, second = int(counts[top_two[0]]), int(counts[top_two[1]])
    p_value = float(
        binomtest(first, first + second, p=0.5, alternative="two-sided").pvalue
    )
    prediction = int(top_two[0]) if p_value <= alpha else ABSTAIN
    return PredictionResult(prediction, counts, p_value)


def certify_from_counts(
    selection_counts: np.ndarray,
    estimation_counts: np.ndarray,
    sigma: float,
    alpha: float,
) -> CertificationResult:
    selection_counts = np.asarray(selection_counts)
    estimation_counts = np.asarray(estimation_counts)
    selected = int(np.argmax(selection_counts))
    trials = int(estimation_counts.sum())
    lower = clopper_pearson_lower(int(estimation_counts[selected]), trials, alpha)

    if lower <= 0.5:
        return CertificationResult(
            ABSTAIN, 0.0, selected, selection_counts, estimation_counts, lower
        )
    return CertificationResult(
        selected,
        certified_radius(sigma, lower),
        selected,
        selection_counts,
        estimation_counts,
        lower,
    )
