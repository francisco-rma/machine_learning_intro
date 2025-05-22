from collections import namedtuple
import numpy as np
from linear_models.kaggle import path, file_name

TOLERANCE = 10**-4
DEFAULT_SIZE = 1000
COL_INDICES = [2, 3, 4]
PARAMS = ["Threshold", "MinTemp", "MaxTemp"]

Data = namedtuple("Data", PARAMS)

dimensionality = len(Data._fields)
seed = 1

source_parameters = np.array([-0.7, 1.0, 1.0])

kaggle_data: np.ndarray = np.genfromtxt(
    fname=path + "\\" + file_name,
    delimiter=",",
    skip_header=1,
    dtype=float,
    encoding="utf-8",
    usecols=COL_INDICES,
)

mask = np.array(
    list(
        map(
            lambda x, y, z: not (np.isnan(x) or np.isnan(y) or np.isnan(z)),
            kaggle_data[:, 0],
            kaggle_data[:, 1],
            kaggle_data[:, 2],
        )
    )
)

original_len = len(kaggle_data)
kaggle_data = kaggle_data[mask]
assert len(kaggle_data) < original_len

kaggle_result = kaggle_data[:, 2].astype(str)

thresholds = np.array([1.0] * kaggle_data.shape[0])

kaggle_data = np.stack(
    arrays=[thresholds, kaggle_data[:, 0].astype(float), kaggle_data[:, 1].astype(float)], axis=1
)


kaggle_result[kaggle_result == "NA"] = np.nan
kaggle_result = np.nan_to_num(x=kaggle_result)
kaggle_result = kaggle_result.astype(float)
kaggle_result = np.delete(kaggle_result, 0)
kaggle_result = np.insert(arr=kaggle_result, obj=kaggle_result.shape[0] - 1, values=0.0)


def generate_sample_data(
    rng: np.random.Generator,
    sample_size: int = DEFAULT_SIZE,
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
