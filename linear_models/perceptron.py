from typing import Callable
import numpy as np
from utils import benchmark
from linear_models.data import (
    TOLERANCE,
    dimensionality,
    seed,
)


def perceptron(data: np.ndarray, result: np.ndarray, target_parameters: np.ndarray) -> bool:
    assert data.shape[1] == target_parameters.shape[0]

    test = np.array(list(map(lambda x: x >= 0.0, np.matmul(data, target_parameters))))
    assert test.shape == result.shape

    diff: np.ndarray = result != test
    assert diff.shape == test.shape == result.shape

    if diff.any():
        idx = diff.argmax()
        y = -1 if test[idx] else 1
        values = data[idx]
        target_parameters += values * y
        return False
    else:
        return True


def numeric_perceptron(data: np.ndarray, result: np.ndarray, target_parameters: np.ndarray) -> bool:
    assert data.shape[1] == target_parameters.shape[0]
    test: np.ndarray = np.matmul(data, target_parameters)
    assert test.shape == result.shape

    diffs = np.abs(result - test) > TOLERANCE
    idx = diffs.argmax()

    if np.abs(result[idx] - test[idx]) > TOLERANCE:
        assert not np.isnan(result[idx] - test[idx])
        y = result[idx] - test[idx]

        y = min(y, 1) if y >= 0 else max(y, -1)

        values = data[idx]
        target_parameters += values * y
        return False
    else:
        return True


@benchmark
def train_numeric(
    data: np.ndarray,
    result: np.ndarray,
    iterations: int = 1000,
) -> tuple[Callable, np.ndarray]:
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    target_parameters = np.array([rng.random() for _ in range(dimensionality)])
    i = 0
    done = False
    while i < iterations and not done:
        i += 1
        done = numeric_perceptron(data=data, result=result, target_parameters=target_parameters)

    print(f"Processed LINEAR REGRESSION for {i} iterations!")
    print("Final parameters: ", target_parameters)
    print("\n")

    params = target_parameters.copy()

    def g(data_point: np.ndarray) -> float:
        assert len(data_point) == len(params)
        return np.dot(a=data_point, b=params)

    return g, params


@benchmark
def train(
    data: np.ndarray,
    result: np.ndarray,
    iterations: int = 1000,
) -> tuple[Callable, np.ndarray]:
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    target_parameters = np.array([rng.random() for _ in range(dimensionality)])
    i = 0
    done = False
    while i < iterations and not done:
        i += 1
        done = perceptron(data=data, result=result, target_parameters=target_parameters)

    print(f"Processed PERCEPTRON for {i} iterations!")
    print("Final parameters: ", target_parameters)
    print("\n")

    params = target_parameters.copy()

    def g(data_point: np.ndarray) -> float:
        assert len(data_point) == len(params)
        return np.dot(a=data_point, b=params)

    return g, params


def measure_numeric_error(
    data: np.ndarray, control: np.ndarray, map_func: Callable, err_func=lambda x, y: (x - y) ** 2
) -> np.ndarray:
    err_list = np.array(
        [
            err_func(map_func(data_point=data_point), result)
            for data_point, result in zip(data, control)
        ]
    )
    return err_list


def measure_classification_error(
    data: np.ndarray, control: np.ndarray, map_func: Callable, err_func=lambda x, y: x != y
) -> np.ndarray:
    err_list = np.array(
        [
            err_func(map_func(data_point=data_point) >= 0.0, result)
            for data_point, result in zip(data, control)
        ]
    )
    return err_list
