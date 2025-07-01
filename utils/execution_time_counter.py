import time
from functools import wraps

def execution_time(predict_func):
    """Decorator to calculate and log the execution time of a function."""
    @wraps(predict_func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = predict_func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"{predict_func.__name__} executed in {execution_time:.6f} seconds")
        return result, execution_time
    return wrapper
