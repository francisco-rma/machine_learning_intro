import numpy as np
import matplotlib.pyplot as plt
from data import generate_data
import perceptron
import linear_regression


def normalize(a: list, b: list):
    if len(a) < len(b):
        a.extend([None] * abs(len(a) - len(b)))
    elif len(a) > len(b):
        b.extend([None] * abs(len(a) - len(b)))


iterations = 1000
size = 10000


perceptron_convergence, perceptron_numeric_error, perceptron_classification_error = (
    perceptron.train(data=perceptron.sample_data, iterations=iterations)
)

(
    linear_regression_convergence,
    linear_regression_numeric_error,
    linear_regression_classification_error,
) = linear_regression.train(data=linear_regression.sample_data, iterations=iterations)


normalize(a=perceptron_convergence, b=linear_regression_convergence)
normalize(a=perceptron_numeric_error, b=linear_regression_numeric_error)
normalize(a=perceptron_classification_error, b=linear_regression_classification_error)

plt.plot(perceptron_convergence, label="perceptron")
plt.plot(linear_regression_convergence, label="linear regression")
plt.xlabel("Iteration")
plt.ylabel("Angle cosine")
plt.legend()
plt.tight_layout()
plt.savefig("linear_models_convergence.png")
plt.show()

plt.scatter(
    x=range(len(perceptron_classification_error)),
    y=perceptron_classification_error,
    label="perceptron",
    s=10,
)
plt.scatter(
    x=range(len(linear_regression_classification_error)),
    y=linear_regression_classification_error,
    label="linear regression",
    s=10,
)
plt.xlabel("Iteration")
plt.ylabel("Classification error")
plt.legend()
plt.tight_layout()
plt.savefig("perceptron_classification_error")
plt.show()

plt.scatter(
    x=range(len(perceptron_numeric_error)), y=perceptron_numeric_error, label="perceptron", s=10
)
plt.scatter(
    x=range(len(linear_regression_numeric_error)),
    y=linear_regression_numeric_error,
    label="linear regression",
    s=10,
)
plt.xlabel("Iteration")
plt.ylabel("Numeric error")
plt.legend()
plt.tight_layout()
plt.savefig("perceptron_numeric_error")
plt.show()

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
