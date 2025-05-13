from collections import namedtuple
import numpy as np

seed = 1
rng = np.random.default_rng(seed=seed)

Credit = namedtuple(
    "Credit", ["threshold", "wealth_score", "profile_score", "debt_score", "result"]
)

source_parameters = [-0.7, 1.0, 1.0, 1.0]


def f(data_point: Credit) -> bool:
    return sum(
        [
            source_parameters[0] * data_point.threshold,
            source_parameters[1] * data_point.wealth_score,
            source_parameters[2] * data_point.profile_score,
            source_parameters[3] * data_point.debt_score,
        ]
    )


scale = 1
dimensionality = 4

sample_size = 10000


# wealth_score = rng.normal(0.5 * scale, std, sample_size)
# profile_score = rng.normal(0.5 * scale, std, sample_size)

avg = 0.3
std = 0.1


thresholds = [1.0] * sample_size
wealth_score = rng.normal(loc=avg, scale=std, size=sample_size)
profile_score = rng.normal(loc=avg, scale=std, size=sample_size)
debt_score = rng.normal(loc=avg, scale=std, size=sample_size)

data = list(
    map(
        lambda x: Credit(
            *x,
            result=sum(
                [
                    x[0] * source_parameters[0],
                    x[1] * source_parameters[1],
                    x[2] * source_parameters[2],
                    x[3] * source_parameters[3],
                ]
            )
            >= 0
        ),
        zip(thresholds, wealth_score, profile_score, debt_score),
    )
)
