"""Расчёты для экспериментов с двумя или тремя группами."""

import math
from statistics import NormalDist


def n_per_group(sigma, mde, alpha=0.05, power=0.8, comparisons=1):
    """Размер каждой группы при двустороннем тесте и поправке Бонферрони."""
    return math.ceil((mde_for_n(sigma, 1, alpha, power, comparisons) / mde) ** 2)


def mde_for_n(sigma, n, alpha=0.05, power=0.8, comparisons=1):
    """Минимальный обнаруживаемый эффект при n наблюдениях на группу."""
    normal = NormalDist()
    critical = normal.inv_cdf(1 - alpha / (2 * comparisons))
    power_quantile = normal.inv_cdf(power)
    return (critical + power_quantile) * sigma * math.sqrt(2 / n)


def diff_ci(t, c, alpha=0.05):
    """Разность средних test − control и интервал по нормальному приближению."""
    diff = t.mean() - c.mean()
    se = math.sqrt(t.var() / len(t) + c.var() / len(c))
    margin = NormalDist().inv_cdf(1 - alpha / 2) * se
    p = math.erfc(abs(diff) / (se * math.sqrt(2))) if se else float(diff == 0)
    return {"diff": diff, "lo": diff - margin, "hi": diff + margin, "p": p}


def srm_pvalue(sizes):
    """Проверка χ² согласия с равными долями для двух или трёх групп."""
    sizes = tuple(sizes)
    if len(sizes) not in (2, 3):
        raise ValueError("SRM поддерживает две или три группы")
    expected = sum(sizes) / len(sizes)
    chi_square = sum((size - expected) ** 2 / expected for size in sizes)
    if len(sizes) == 2:
        return math.erfc(math.sqrt(chi_square / 2))
    return math.exp(-chi_square / 2)


def cuped(y, x):
    """Скорректированная метрика и коэффициент θ для предпериода x."""
    variance = x.var()
    theta = y.cov(x) / variance if variance else 0.0
    return y - theta * (x - x.mean()), theta
