import time


def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function '{func.__name__}' executed in {end - start:.6f} seconds.")
        return result

    return wrapper
