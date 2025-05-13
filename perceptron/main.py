from data import Credit, data, rng, dimensionality, f

target_parameters = [rng.random() for _ in range(dimensionality)]


def g(data_point: Credit) -> bool:
    return sum(
        [
            target_parameters[0] * data_point.threshold,
            target_parameters[1] * data_point.wealth_score,
            target_parameters[2] * data_point.profile_score,
            target_parameters[3] * data_point.debt_score,
        ]
    )


done = False
i = 0

while not done:
    i += 1
    for idx, row in enumerate(data):
        test = g(row)
        control = f(row)

        test_result = test >= 0.0
        control_result = control >= 0.0

        assert row.result == control_result

        if test_result != control_result:
            y = 1 if test < control else -1  # * f(row)

            target_parameters[0] += y * row.threshold
            target_parameters[1] += y * row.wealth_score
            target_parameters[2] += y * row.profile_score
            target_parameters[3] += y * row.debt_score
            break

        if idx == len(data) - 1:
            done = True

print(f"Processed after {i} iterations!")
print(target_parameters)
