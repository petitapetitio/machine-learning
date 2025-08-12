def divide_by_max(x: list[float]) -> list[float]:
    x_max = max(x)
    return [xi / x_max for xi in x]


def mean_normalization(x: list[float]) -> list[float]:
    avg = sum(x) / len(x)
    return [xi / avg for xi in x]


def z_score_normalization(x: list[float]) -> list[float]:
    avg = sum(x) / len(x)
    stddev = sum([(xi - avg) ** 2 for xi in x]) / len(x)
    return [(xi - avg) / stddev for xi in x]
