import numpy as np
import matplotlib.pyplot as plt
from linear_models.data import (
    DEFAULT_SIZE,
    f,
    generate_sample_data,
    dimensionality,
    source_parameters,
)
import linear_models.perceptron as perceptron
import linear_models.linear_regression as linear_regression
import linear_models.plots as plots


def normalize(a: list, b: list):
    if len(a) < len(b):
        a.extend([None] * abs(len(a) - len(b)))
    elif len(a) > len(b):
        b.extend([None] * abs(len(a) - len(b)))


iter_n = 1000
size = DEFAULT_SIZE * 10

sample_data, sample_result = generate_sample_data(
    rng=np.random.default_rng(seed=0), dimensionality=dimensionality, sample_size=DEFAULT_SIZE
)
sample_boolean_result = sample_result >= 0.0


(
    linear_regression_cosine_convergence,
    linear_regression_num_err_convergence,
    linear_regression_class_err_convergence,
) = linear_regression.train(
    data=sample_data,
    result=sample_result,
    iterations=iter_n,
    measure_convergence=True,
)

perceptron_cosine_convergence, perceptron_num_err_convergence, perceptron_class_err_convergence = (
    perceptron.train(
        data=sample_data, result=sample_boolean_result, iterations=iter_n, measure_convergence=True
    )
)

x_axis = range(len(perceptron_class_err_convergence))
assert len(x_axis) == len(perceptron_class_err_convergence)
assert len(x_axis) == len(perceptron_cosine_convergence)
assert len(x_axis) == len(perceptron_num_err_convergence)
assert len(x_axis) == len(linear_regression_cosine_convergence)
assert len(x_axis) == len(linear_regression_class_err_convergence)
assert len(x_axis) == len(linear_regression_num_err_convergence)

# cosine, classification error and numeric error convergences
plt.figure()
plt.plot(perceptron_cosine_convergence, label="perceptron")
plt.plot(linear_regression_cosine_convergence, label="linear regression")
plt.xlabel("Iteration")
plt.ylabel("Angle cosine")
plt.legend()
plt.tight_layout()
plt.savefig("linear_models_convergence.png")
plt.show(block=False)

plt.figure()
plt.scatter(
    x=x_axis,
    y=perceptron_class_err_convergence,
    label="perceptron",
    s=10,
)
plt.scatter(
    x=x_axis,
    y=linear_regression_class_err_convergence,
    label="linear regression",
    s=10,
)
plt.xlabel("Iteration")
plt.ylabel("Classification error")
plt.legend()
plt.tight_layout()
plt.savefig("perceptron_classification_error")
plt.show(block=False)

plt.figure()
plt.scatter(x=x_axis, y=perceptron_num_err_convergence, label="perceptron", s=10)
plt.scatter(
    x=x_axis,
    y=linear_regression_num_err_convergence,
    label="linear regression",
    s=10,
)
plt.xlabel("Iteration")
plt.ylabel("Numeric error")
plt.legend()
plt.tight_layout()
plt.savefig("perceptron_numeric_error")
plt.show(block=False)

out_of_sample_data, out_of_sample_result = generate_sample_data(
    rng=np.random.default_rng(seed=3), sample_size=size, dimensionality=dimensionality
)
out_of_sample_data = np.array(out_of_sample_data)
out_of_sample_result = np.array(out_of_sample_result)
out_of_sample_bool_result = np.array(list(map(lambda x: x >= 0.0, out_of_sample_result)))


perceptron_class_error = perceptron.measure_classification_error(
    data=out_of_sample_data, control=out_of_sample_bool_result
)
perceptron_num_err_convergence = perceptron.measure_numeric_error(
    out_of_sample_data, control=out_of_sample_result
)

linear_regression_class_error = linear_regression.measure_classification_error(
    data=out_of_sample_data, control=out_of_sample_bool_result
)

linear_regression_num_err_convergence = linear_regression.measure_numeric_error(
    out_of_sample_data, control=out_of_sample_result
)

print(f"perceptron classification error: {np.mean(perceptron_class_err_convergence)}")
print(f"perceptron numeric error: {np.mean(perceptron_num_err_convergence)}")

print(f"linear regression classification error: {np.mean(linear_regression_class_err_convergence)}")
print(f"linear regression numeric error: {np.mean(linear_regression_num_err_convergence)}")

# resulting data sets

plots.scatter_plot(
    title="Generated Data",
    data=out_of_sample_data,
    h=f,
    file="source_func_scatter.png",
    params=source_parameters,
)
plots.scatter_plot(
    title="Perceptron Data",
    data=out_of_sample_data,
    h=perceptron.g,
    file="perceptron_func_scatter.png",
    params=perceptron.target_parameters,
)
plots.scatter_plot(
    title="Linear Regression Data",
    data=out_of_sample_data,
    h=linear_regression.g,
    file="linear_regression_func_scatter.png",
    params=linear_regression.target_parameters,
)

plt.show()
