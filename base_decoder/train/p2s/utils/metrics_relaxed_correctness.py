from typing import Any, Callable, Optional, Sequence


def aggregate_metrics(
        targets: Sequence[Sequence[str]],
        predictions: Sequence[str],
        metric_fn: Callable[[str, str], Any],
        normalize_fn: Callable[[str], str] = lambda v: v) -> float:
    """Aggregate target-prediction pair metrics over a dataset."""
    assert len(targets) == len(predictions)
    total = 0
    for prediction, target in zip(predictions, targets):
        p = normalize_fn(prediction)
        total += max(metric_fn(normalize_fn(t), p) for t in target)
    return (100.0 * total) / len(targets)


def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
        target: Target string.
        prediction: Predicted string.
        max_relative_change: Maximum relative change.

    Returns:
        Whether the prediction was correct given the specified tolerance.
  """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)

    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change

    else:
        return prediction.lower() == target.lower()
