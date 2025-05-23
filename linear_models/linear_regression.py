from typing import Callable
import numpy as np
from utils import benchmark


def pseudo_inverse_linear_regression(data: np.ndarray, result: np.ndarray) -> np.ndarray:
    assert data.shape[0] == result.shape[0]

    square_inv = np.linalg.inv(np.linalg.matmul(data.T, data))
    pseudo_inv = np.linalg.matmul(square_inv, data.T)

    target_parameters: np.ndarray = np.matmul(pseudo_inv, result)
    assert data.shape[1] == target_parameters.shape[0]

    return target_parameters


@benchmark
def train(data: np.ndarray, result: np.ndarray) -> tuple[Callable, np.ndarray]:
    params = pseudo_inverse_linear_regression(data=data, result=result)
    print("Processed LINEAR REGRESSION")
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
