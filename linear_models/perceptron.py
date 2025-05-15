import numpy as np
from typing import Sequence
from data import Credit, cosine, dot_product, f, source_parameters, generate_binary_data

seed = 1
dimensionality = len(Credit._fields)
sample_size = 10000
rng: np.random.Generator = np.random.default_rng(seed=seed)

target_parameters = [rng.random() for _ in range(dimensionality)]
source_vector_size = np.sqrt(dot_product(source_parameters, source_parameters))

sample_data = generate_binary_data(rng=rng, dimensionality=dimensionality, sample_size=sample_size)


def g(data_point: Sequence[float]) -> bool:
    assert len(data_point) == len(target_parameters)
    return dot_product(x=data_point, y=target_parameters)


def perceptron(data: list[tuple[Credit, bool]], target_params: list[float]):
    for idx, (data_point, result) in enumerate(data):
        test = g(data_point=data_point)
        control = f(data_point=data_point)
        assert result == (control >= 0.0)

        test_result = test >= 0.0

        if test_result != result:
            y = 1 if test < control else -1

            for idx, value in enumerate(data_point._asdict().values()):
                target_params[idx] += y * value
            return False

        if idx == len(data) - 1:
            return True


def train(data: list[tuple[Credit, bool]], iterations: int = 1000) -> list[float]:
    convergence = [cosine(target_parameters, source_parameters)]
    done = False

    i = 0

    while i < iterations and not done:
        i += 1
        done = perceptron(data=data, target_params=target_parameters)
        convergence.append(cosine(target_parameters, source_parameters))

    print(f"Processed PERCEPTRON for {i} iterations!")
    print(target_parameters)
    print(source_parameters)
    print("\n")

    return convergence


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
