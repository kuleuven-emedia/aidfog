"""
Utility functions for the PineBuds Pro audio cueing system.
"""


def update_running_stats(
    count: int,
    mean: float,
    mean2: float,
    sample: float,
    min_sample: float,
    max_sample: float,
) -> tuple[int, float, float, float, float]:
    """Welford's online algorithm for running mean/variance."""
    count += 1
    delta = sample - mean
    mean += delta / count
    delta2 = sample - mean
    mean2 += delta * delta2
    return count, mean, mean2, min(sample, min_sample), max(sample, max_sample)


def finalize_running_stats(
    count: int, mean: float, mean2: float
) -> tuple[float, float, float] | None:
    """Finalize running stats into (mean, variance, sample_variance)."""
    if count < 2:
        return None
    variance = mean2 / count
    sample_variance = mean2 / (count - 1)
    return mean, variance, sample_variance


def compute_percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile of a list using linear interpolation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
