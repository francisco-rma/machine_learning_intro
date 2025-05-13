import numpy as np
from data import Credit, data, rng, dimensionality, source_parameters


print(data)

target_parameters = [rng.random() for _ in range(dimensionality)]


def f(data_point: Credit) -> bool:
    return sum(
        [
            source_parameters[0] * data_point.threshold,
            source_parameters[2] * data_point.wealth_score,
            source_parameters[3] * data_point.profile_score,
            source_parameters[1] * data_point.debt_score,
        ]
    )


def g(data_point: Credit) -> bool:
    return sum(
        [
            target_parameters[0] * data_point.threshold,
            target_parameters[2] * data_point.wealth_score,
            target_parameters[3] * data_point.profile_score,
            target_parameters[1] * data_point.debt_score,
        ]
    )


print([g(data_point=row) for row in data])

done = False

while not done:
    for idx, row in enumerate(data):
        test = g(row)
        control = f(row)

        test_result = test >= 0
        control_result = control >= 0

        assert row.result == control_result

        if test_result != control_result:
            print(f"Mismatch at index {idx}")
            print(f"row: {row}")
            print(f"g(row): {test}")
            print(f"Target parameters: {target_parameters}")

            y = f(row)

            print("y ", y)
            target_parameters[0] += y * row.threshold
            target_parameters[1] += y * row.wealth_score
            target_parameters[2] += y * row.profile_score
            target_parameters[3] += y * row.debt_score
            break

        if idx == len(data) - 1:
            done = True

print(target_parameters)
print("Done!")
