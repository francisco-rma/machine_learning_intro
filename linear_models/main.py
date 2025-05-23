import numpy as np
import matplotlib.pyplot as plt
import linear_models.perceptron as perceptron
import linear_models.linear_regression as linear_regression
import linear_models.plots as plots
import linear_models.data as data


CONVERGENCE = False
SAMPLE_SIZE = 10**4
ITERN_N = 2 * 10**4


def normalize(a: list, b: list):
    if len(a) < len(b):
        a.extend([None] * abs(len(a) - len(b)))
    elif len(a) > len(b):
        b.extend([None] * abs(len(a) - len(b)))


training_data, training_result = data.generate_sample_data(rng=np.random.default_rng(seed=0))

# indices = np.random.default_rng(seed=10).integers(0, kaggle_data.shape[0], size=SAMPLE_SIZE)

# training_data = kaggle_data[indices]
# training_result = kaggle_result[indices]

# kaggle_data[:SAMPLE_SIZE], kaggle_result[:SAMPLE_SIZE]

print("training data shape:", data.kaggle_data.shape)
print("training data dtype:", data.kaggle_data.dtype)
print("training result shape:", data.kaggle_result.shape)
print("training result dtype:", data.kaggle_result.dtype)
print("\n")

training_boolean_result = training_result >= 0.0
training_classification_result = np.array(
    list(map(lambda x: 1 if x else -1, training_boolean_result))
)

# lr_g, lr_params = linear_regression.train(data=training_data, result=training_classification_result)
lr_g, lr_params = linear_regression.train(data=training_data, result=training_result)

perceptron_g, perceptron_params = perceptron.train(
    data=training_data, result=training_boolean_result, iterations=ITERN_N
)

out_of_sample_data, out_of_sample_result = data.generate_sample_data(
    rng=np.random.default_rng(seed=3), sample_size=data.DEFAULT_SIZE * 10
)
# out_of_sample_data, out_of_sample_result = kaggle_data, kaggle_result

out_of_sample_bool_result = np.array(list(map(lambda x: x >= 0.0, out_of_sample_result)))


perceptron_class_error = perceptron.measure_classification_error(
    data=out_of_sample_data, control=out_of_sample_bool_result, map_func=perceptron_g
)
perceptron_num_error = perceptron.measure_numeric_error(
    out_of_sample_data, control=out_of_sample_result, map_func=perceptron_g
)

linear_regression_class_error = linear_regression.measure_classification_error(
    data=out_of_sample_data, control=out_of_sample_bool_result, map_func=lr_g
)

linear_regression_num_error = linear_regression.measure_numeric_error(
    out_of_sample_data, control=out_of_sample_result, map_func=lr_g
)

print(f"perceptron classification error: {np.mean(perceptron_class_error)}")
print(f"perceptron numeric error: {np.mean(perceptron_num_error)}")

print(f"linear regression classification error: {np.mean(linear_regression_class_error)}")
print(f"linear regression numeric error: {np.mean(linear_regression_num_error)}")

# resulting data sets

plots.result_plot(
    title="Source Data",
    data=out_of_sample_data,
    result=out_of_sample_result,
    file="result_scatter.png",
)
plots.scatter_plot(
    title="Source Data",
    data=out_of_sample_data,
    h=data.f,
    params=data.source_parameters,
    file="result_scatter.png",
)
plots.scatter_plot(
    title="Perceptron Data",
    data=out_of_sample_data,
    h=perceptron_g,
    file="perceptron_func_scatter.png",
    params=perceptron_params,
)
plots.scatter_plot(
    title="Linear Regression Data",
    data=out_of_sample_data,
    h=lr_g,
    file="linear_regression_func_scatter.png",
    params=lr_params,
)

plt.show()
