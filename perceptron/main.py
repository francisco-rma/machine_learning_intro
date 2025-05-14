import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from data import Credit, cosine, dot_product, f, training_data, source_parameters


seed = 1
dimensionality = len(Credit._fields)
sample_size = 10000
rng: np.random.Generator = np.random.default_rng(seed=seed)

target_parameters = [rng.random() for _ in range(dimensionality)]
source_vector_size = np.sqrt(dot_product(source_parameters, source_parameters))


def g(data_point: Sequence[float]) -> bool:
    assert len(data_point) == len(target_parameters)
    return dot_product(x=data_point, y=target_parameters)


data = training_data(rng=rng, dimensionality=dimensionality, sample_size=sample_size)


def binary_perceptron(data: list[tuple[Credit, bool]], target_params: list[float]):
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


def dynamic_perceptron(data: list[tuple[Credit, bool]], target_params: list[float]):
    for idx, (data_point, result) in enumerate(data):
        test = g(data_point=data_point)
        control = f(data_point=data_point)
        assert result == (control >= 0.0)

        test_result = test >= 0.0

        if test_result != result:
            y = control - test

            for idx, value in enumerate(data_point._asdict().values()):
                target_params[idx] += y * value
            return False

        if idx == len(data) - 1:
            return True


convergence = [cosine(target_parameters, source_parameters)]

done = False

idx = 0


while not done:
    idx += 1
    # done = binary_perceptron(data=data, target_params=target_parameters)
    done = dynamic_perceptron(data=data, target_params=target_parameters)
    convergence.append(cosine(target_parameters, source_parameters))

print(f"Processed BINARY perceptron after {idx} iterations!")
print(target_parameters)
print(source_parameters)

plt.plot(convergence[:1000])
# plt.savefig("convergence_constant_adjustment.png")
plt.savefig("convergence_dynamic_adjustment.png")
plt.show()
