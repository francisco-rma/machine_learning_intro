import numpy as np
from typing import Sequence
from data import (
    Credit,
    cosine,
    dot_product,
    f,
    source_parameters,
    generate_data,
    DEFAULT_SIZE,
    dimensionality,
    seed,
)


rng: np.random.Generator = np.random.default_rng(seed=seed)

target_parameters = [rng.random() for _ in range(dimensionality)]
source_vector_size = np.sqrt(dot_product(source_parameters, source_parameters))

creds, result = generate_data(rng=rng, dimensionality=dimensionality, sample_size=DEFAULT_SIZE)
sample_data = list(zip(creds, result))


def g(data_point: Sequence[float]) -> float:
    assert len(data_point) == len(target_parameters)
    return dot_product(x=data_point, y=target_parameters)


def linear_regression(data: list[tuple[Credit, float]], target_params: list[float]) -> bool:
    for idx, (data_point, result) in enumerate(data):
        test_value = g(data_point=data_point)

        control_value = f(data_point=data_point)
        control_result = result >= 0.0
        assert control_result == (control_value >= 0.0)

        test_result = test_value >= 0.0

        if test_result != control_result:
            y = control_value - test_value

            for idx, value in enumerate(data_point._asdict().values()):
                target_params[idx] += y * value
            return False

        if idx == len(data) - 1:
            return True


def train(
    data: list[tuple[Credit, float]], iterations: int = 1000
) -> tuple[list[float], list[float], list[float]]:
    convergence = [cosine(target_parameters, source_parameters)]
    err_num = [np.mean(measure_numeric_error(data=data))]
    err_class = [
        len(list(filter(lambda x: x is True, measure_classification_error(data=data)))) / len(data)
    ]

    i = 0

    while i < iterations:
        i += 1
        linear_regression(data=data, target_params=target_parameters)
        convergence.append(cosine(target_parameters, source_parameters))
        err_num.append(np.mean(measure_numeric_error(data=data)))
        err_class.append(
            len(list(filter(lambda x: x is True, measure_classification_error(data=data))))
            / len(data)
        )

    print(f"Processed LINEAR REGRESSION for {i} iterations!")
    print(target_parameters)
    print(source_parameters)
    print("\n")

    return convergence, err_num, err_class


def measure_numeric_error(
    data: list[tuple[Credit, float]], err_func=lambda x, y: (x - y) ** 2
) -> list[float]:
    err_list = [err_func(g(data_point=data_point), result) for data_point, result in data]
    return err_list


def measure_classification_error(
    data: list[tuple[Credit, bool]], err_func=lambda x, y: x != y
) -> list[bool]:
    err_list = [err_func(g(data_point=data_point) >= 0.0, result) for data_point, result in data]
    return err_list


def classify(data: list[Credit]) -> list[bool]:
    result = [g(data_point=data_point) >= 0.0 for data_point in list(zip(*data))[0]]
    return result


def calculate(data: list[Credit]) -> list[float]:
    result = [g(data_point=data_point) for data_point in list(zip(*data))[0]]
    return result
