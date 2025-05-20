import numpy as np
from utils import benchmark
from linear_models.data import (
    Credit,
    cosine,
    source_parameters,
    generate_data,
    DEFAULT_SIZE,
    TOLERANCE,
    dimensionality,
    seed,
)


rng: np.random.Generator = np.random.default_rng(seed=seed)

target_parameters = np.array([rng.random() for _ in range(dimensionality)])
source_vector_size = np.sqrt(np.dot(source_parameters, source_parameters))

creds, result = generate_data(rng=rng, dimensionality=dimensionality, sample_size=DEFAULT_SIZE)
sample_data = list(zip(creds, result))


def g(data_point: np.ndarray) -> float:
    assert len(data_point) == len(target_parameters)
    return np.dot(a=data_point, b=target_parameters)


def linear_regression(data: np.ndarray, result: np.ndarray, target_params: np.ndarray) -> bool:
    assert data.shape[1] == target_params.shape[0]
    test: np.ndarray = np.matmul(data, target_params)
    assert test.shape == result.shape

    diff = np.abs(result - test)

    if diff.max() >= TOLERANCE:
        idx = diff.argmax()
        y = result[idx] - test[idx]
        values = data[idx]
        target_params += values * y


@benchmark
def train(
    rows: list[tuple[Credit, float]], iterations: int = 1000
) -> tuple[list[float], list[float], list[float]]:
    data = np.ndarray((len(rows), dimensionality))
    result = np.ndarray((len(rows)))
    boolean_result = np.ndarray((len(rows), 1))

    for idx, (row, control_value) in enumerate(rows):
        data[idx, :] = row
        result[idx] = control_value
        boolean_result[idx] = control_value >= 0.0

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
        linear_regression(data=data, result=result, target_params=target_parameters)
        convergence.append(cosine(target_parameters, source_parameters))
        err_num.append(np.mean(measure_numeric_error(data=data, control=result)))
        err_class.append(
            len(
                list(
                    filter(
                        lambda x: x is True,
                        measure_classification_error(data=data, control=boolean_result),
                    )
                )
            )
            / len(data)
        )

    print(f"Processed LINEAR REGRESSION for {i} iterations!")
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


def classify(data: list[Credit]) -> list[bool]:
    result = [g(data_point=data_point) >= 0.0 for data_point in list(zip(*data))[0]]
    return result


def calculate(data: list[Credit]) -> list[float]:
    result = [g(data_point=data_point) for data_point in list(zip(*data))[0]]
    return result
