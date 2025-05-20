import numpy as np
from utils import benchmark
from linear_models.data import (
    Data,
    cosine,
    dimensionality,
    seed,
)


rng: np.random.Generator = np.random.default_rng(seed=seed)
target_parameters = np.array([rng.random() for _ in range(dimensionality)])


def g(data_point: np.ndarray) -> float:
    assert len(data_point) == len(target_parameters)
    return np.dot(a=data_point, b=target_parameters)


def perceptron(data: np.ndarray, result: np.ndarray, target_params: np.ndarray) -> bool:
    assert data.shape[1] == target_params.shape[0]
    test = np.array(list(map(lambda x: x >= 0.0, np.matmul(data, target_params))))
    assert test.shape == result.shape

    diff: np.ndarray = result == test
    assert diff.shape == test.shape
    assert diff.shape == result.shape

    if diff.any():
        idx = diff.argmax()
        y = -1 if test[idx] else 1
        values = data[idx]
        target_params += values * y


@benchmark
def train(
    data: np.ndarray, result: np.ndarray, iterations: int = 1000, measure_convergence: bool = False
) -> tuple[list[float], list[float], list[float]]:
    # For the purposes of measuring the cosine evolution between target and source parameters
    from linear_models.data import source_parameters

    boolean_result = result >= np.zeros(result.shape)

    convergence = []
    err_num = []
    err_class = []

    if measure_convergence:
        convergence = [cosine(target_parameters, source_parameters)]
        err_num = [np.mean(measure_numeric_error(data=data, control=result))]
        err_class = [
            len(
                list(
                    filter(
                        lambda x: x is True,
                        measure_classification_error(data=data, control=boolean_result),
                    )
                )
            )
            / len(data)
        ]

    i = 0

    while i < iterations:
        i += 1
        perceptron(data=data, result=result, target_params=target_parameters)
        if measure_convergence:
            convergence.append(cosine(target_parameters, source_parameters))
            err_num.append(np.mean(measure_numeric_error(data=data, control=result)))
            class_err_freq = len(
                list(
                    filter(
                        lambda x: x,
                        measure_classification_error(data=data, control=result),
                    )
                )
            ) / len(data)
            err_class.append(class_err_freq)

    print(f"Processed PERCEPTRON for {i} iterations!")
    print(target_parameters)
    print(source_parameters)
    print("\n")

    return convergence, err_num, err_class


def measure_numeric_error(
    data: np.ndarray, control: np.ndarray, err_func=lambda x, y: (x - y) ** 2
) -> np.ndarray:
    err_list = np.array(
        [err_func(g(data_point=data_point), result) for data_point, result in zip(data, control)]
    )
    return err_list


def measure_classification_error(
    data: np.ndarray, control: np.ndarray, err_func=lambda x, y: x != y
) -> np.ndarray:
    err_list = np.array(
        [
            err_func(g(data_point=data_point) >= 0.0, result)
            for data_point, result in zip(data, control)
        ]
    )
    return err_list


def classify(data: list[Data]) -> list[bool]:
    result = [g(data_point=data_point) >= 0.0 for data_point in list(zip(*data))[0]]
    return result


def calculate(data: list[Data]) -> list[float]:
    result = [g(data_point=data_point) for data_point in list(zip(*data))[0]]
    return result
