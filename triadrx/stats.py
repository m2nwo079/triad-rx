"""Small statistics helpers shared by the pipeline."""
import math


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom, (centre + margin) / denom


def quality_status(lower: float, rule: dict) -> str:
    """Pass, conditional pass or fail from the Wilson lower bound (preregistration 5.1)."""
    if lower >= rule["pass"]:
        return "pass"
    if lower >= rule["conditional"]:
        return "conditional"
    return "fail"


def r5_min_docs(stability: float, precision_lower: float) -> int:
    """Rule R5: k = ceil(-ln(1 - stability) / pi)."""
    if not 0 < precision_lower <= 1:
        raise ValueError("precision_lower must be in (0, 1]")
    return math.ceil(-math.log(1 - stability) / precision_lower)
