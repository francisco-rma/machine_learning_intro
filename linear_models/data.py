from collections import namedtuple
import numpy as np

TOLERANCE = 10**-4
DEFAULT_SIZE = 1000
PARAMS = ["threshold", "wealth_score", "profile_score"]

Data = namedtuple("Data", PARAMS)

dimensionality = len(Data._fields)
seed = 1

source_parameters = np.array([-0.7, 1.0, 1.0])


def normalize(target_vector: np.ndarray):
    norm_factor = 1 / np.dot(target_vector, target_vector)
    target_vector *= norm_factor


def cosine(x: np.ndarray, y: np.ndarray):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def f(data_point: np.ndarray) -> float:
    assert len(data_point) == len(source_parameters)
    return np.dot(a=data_point, b=source_parameters)


def generate_binary_data(
    rng: np.random.Generator,
    dimensionality: int,
    sample_size: int = DEFAULT_SIZE,
) -> list[tuple[Data, bool]]:
    avg = 0.4
    std = 0.1

    thresholds = [1.0] * sample_size
    data = list(
        map(
            lambda x: Data(*x),
            zip(
                thresholds,
                *[
                    rng.normal(loc=avg, scale=std, size=sample_size)
                    for _ in range(1, dimensionality)
                ],
            ),
        )
    )

    return data, list(map(lambda x: f(x) >= 0.0, data))


def generate_sample_data(
    rng: np.random.Generator,
    dimensionality: int,
    sample_size: int,
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    avg = 0.4
    std = 0.1

    thresholds = [1.0] * sample_size
    data = list(
        map(
            lambda x: Data(*x),
            zip(
                thresholds,
                *[
                    rng.normal(loc=avg, scale=std, size=sample_size)
                    for _ in range(1, dimensionality)
                ],
            ),
        )
    )

    return np.asarray(a=data, dtype=float), np.array(list(map(f, data)))
