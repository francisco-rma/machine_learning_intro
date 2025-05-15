import numpy as np
from typing import Sequence
from data import Credit, cosine, dot_product, f, source_parameters, generate_data

seed = 1
dimensionality = len(Credit._fields)
sample_size = 10000
rng: np.random.Generator = np.random.default_rng(seed=seed)

target_parameters = [rng.random() for _ in range(dimensionality)]
source_vector_size = np.sqrt(dot_product(source_parameters, source_parameters))

sample_data = generate_data(rng=rng, dimensionality=dimensionality, sample_size=sample_size)


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


def train(data: list[tuple[Credit, float]], iterations: int = 1000) -> list[float]:
    convergence = [cosine(target_parameters, source_parameters)]
    err_num = [measure_numeric_error(data=data)]
    err_class = [measure_classification_error(data=data)]
    done = False

    i = 0

    while i < iterations and not done:
        i += 1
        done = linear_regression(data=data, target_params=target_parameters)
        convergence.append(cosine(target_parameters, source_parameters))
        err_num.append(measure_numeric_error(data=data))
        err_class.append(measure_classification_error(data=data))

    print(f"Processed LINEAR REGRESSION for {i} iterations!")
    print(target_parameters)
    print(source_parameters)
    print("\n")

    return convergence, err_num, err_class


def measure_numeric_error(data: list[tuple[Credit, float]], err_func=lambda x, y: (x - y) ** 2):
    err_list = []
    for data_point, result in data:
        err_list.append(err_func(g(data_point=data_point), result))

    return np.mean(err_list)


def measure_classification_error(data: list[tuple[Credit, float]], err_func=lambda x, y: x != y):
    err_list = []
    for data_point, result in data:
        err_list.append(err_func(g(data_point=data_point) >= 0.0, result >= 0.0))

    return np.mean(err_list)
