from typing import Callable
import numpy as np
from utils import benchmark
from linear_models.data import dimensionality, seed

GRADIENT_TOLERANCE = 10**-2
ETHA = 0.1


def sigmoid_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == w.shape[0]

    result = (-1 / x.shape[0]) * np.sum(
        list(
            map(
                lambda data_point, classification: data_point
                * (classification / (1 + np.exp(classification * np.dot(a=w, b=data_point)))),
                x,
                y,
            )
        ),
        axis=0,
    )

    return result


def logistic_regression(
    data: np.ndarray, result: np.ndarray, target_parameters: np.ndarray
) -> np.ndarray:
    assert data.shape[0] == result.shape[0]
    assert data.shape[1] == target_parameters.shape[0]

    i = 0
    while i < 1000:
        gradient_vector = sigmoid_gradient(x=data, y=result, w=target_parameters)
        print(gradient_vector)
        gradient_norm = np.linalg.norm(gradient_vector)
        # if gradient_norm >= GRADIENT_TOLERANCE:
        target_parameters = target_parameters - (ETHA / gradient_norm) * gradient_vector
        i += 1

    return target_parameters


@benchmark
def train(data: np.ndarray, result: np.ndarray) -> tuple[Callable, np.ndarray]:
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    target_parameters = np.array([rng.random() for _ in range(dimensionality)])
    binary_result = np.array(list(map(lambda x: 1 if x else -1, result)))
    params = logistic_regression(
        data=data, result=binary_result, target_parameters=target_parameters
    )
    print("Processed LOGISTIC REGRESSION")
    print("Final parameters: ", params)
    print("\n")

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
