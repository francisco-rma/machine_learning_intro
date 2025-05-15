import numpy as np
import matplotlib.pyplot as plt
from data import generate_data
import perceptron
import linear_regression

iterations = 10000
size = 100000

perceptron_convergence = perceptron.train(data=perceptron.sample_data, iterations=iterations)
linear_regression_convergence = linear_regression.train(
    data=linear_regression.sample_data, iterations=iterations
)
if len(linear_regression_convergence) < len(perceptron_convergence):
    linear_regression_convergence.extend(
        [None] * abs(len(linear_regression_convergence) - len(perceptron_convergence))
    )
elif len(linear_regression_convergence) > len(perceptron_convergence):
    perceptron_convergence.extend(
        [None] * abs(len(linear_regression_convergence) - len(perceptron_convergence))
    )

plt.plot(perceptron_convergence, label="perceptron")
plt.plot(linear_regression_convergence, label="linear regression")
plt.tight_layout()
plt.show()
plt.savefig("linear_models_convergence")

out_of_sample_data = generate_data(rng=np.random.default_rng(seed=3), sample_size=size)

perceptron_classification_error = perceptron.measure_classification_error(data=out_of_sample_data)
perceptron_numeric_error = perceptron.measure_numeric_error(data=out_of_sample_data)

linear_regression_classification_error = linear_regression.measure_classification_error(
    data=out_of_sample_data
)
linear_regression_numeric_error = linear_regression.measure_numeric_error(data=out_of_sample_data)


print(f"perceptron classification error: {perceptron_classification_error}")
print(f"perceptron numeric error: {perceptron_numeric_error}")

print(f"linear regression classification error: {linear_regression_classification_error}")
print(f"linear regression numeric error: {linear_regression_numeric_error}")
