from typing import Sequence
import numpy as np
from data import Credit, dot_product, f, training_data, source_parameters


seed = 1
dimensionality = len(Credit._fields)
sample_size = 100000
rng: np.random.Generator = np.random.default_rng(seed=seed)

target_parameters = [rng.random() for _ in range(dimensionality)]


def g(data_point: Sequence[float]) -> bool:
    assert len(data_point) == len(target_parameters)
    return dot_product(x=data_point, y=target_parameters)


done = False
i = 0

data = training_data(rng=rng, dimensionality=dimensionality, sample_size=sample_size)

while not done:
    i += 1
    for idx, (data_point, result) in enumerate(data):
        test = g(data_point=data_point)
        control = f(data_point=data_point)
        assert result == (control >= 0.0)

        test_result = test >= 0.0

        if test_result != result:
            y = 1 if test < control else -1
            # y = control - test
            for idx, value in enumerate(data_point._asdict().values()):
                target_parameters[idx] += y * value
            break

        if idx == len(data) - 1:
            done = True

print(f"Processed after {i} iterations!")
print(target_parameters)
print(source_parameters)

print(
    dot_product(target_parameters, source_parameters)
    / (
        np.sqrt(dot_product(target_parameters, target_parameters))
        * np.sqrt(dot_product(source_parameters, source_parameters))
    )
)
