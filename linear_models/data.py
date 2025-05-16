from collections import namedtuple
from typing import Sequence
import numpy as np

DEFAULT_SIZE = 1000

Credit = namedtuple("Credit", ["threshold", "wealth_score", "profile_score", "debt_score"])

dimensionality = len(Credit._fields)
seed = 1

source_parameters = [-1.0, 1.0, 1.0, 1.0]


def dot_product(x: Sequence[float], y: Sequence[float]) -> float:
    return sum([a * b for a, b in zip(x, y)])


def cosine(x: Sequence[float], y: Sequence[float]):
    return dot_product(x, y) / (np.sqrt(dot_product(x, x)) * np.sqrt(dot_product(y, y)))


def f(data_point: Sequence[float]) -> float:
    assert len(data_point) == len(source_parameters)
    return dot_product(x=data_point, y=source_parameters)


def generate_binary_data(
    rng: np.random.Generator,
    dimensionality: int = 4,
    sample_size: int = DEFAULT_SIZE,
) -> list[tuple[Credit, bool]]:
    avg = 0.3
    std = 0.1

    thresholds = [1.0] * sample_size
    data = list(
        map(
            lambda x: (Credit(*x), f(x) >= 0),
            zip(
                thresholds,
                *[
                    rng.normal(loc=avg, scale=std, size=sample_size)
                    for _ in range(1, dimensionality)
                ],
            ),
        )
    )

    return data


def generate_data(
    rng: np.random.Generator,
    dimensionality: int = 4,
    sample_size: int = 1000,
) -> tuple[list[Credit], list[float]]:
    avg = 0.3
    std = 0.1

    thresholds = [1.0] * sample_size
    data = list(
        map(
            lambda x: Credit(*x),
            zip(
                thresholds,
                *[
                    rng.normal(loc=avg, scale=std, size=sample_size)
                    for _ in range(1, dimensionality)
                ],
            ),
        )
    )

    return data, list(map(f, data))
