"""
Module for optimization state indicators.
"""
import time

class State:

    def __init__(self):
        self.iterations = 0
        self.last_updated = time.time_ns()
        self.training_time = 0.0
        self.run_time = 0.0

    def update_write(self):
        self.iterations += 1
        self.last_updated = time.time_ns()

    def update_read(self):
        self.last_updated = time.time_ns()

    def time_run(self, fn):
        with Timing as t:
            fn()
        self.run_time = t

    class Timing:
        def __init__(self):
            self.start_time = 0.0

        def __enter__(self):
            self.start_time = time.time()

        def __exit__(self, exc_type, exc_value, exc_tb):
            end_time = time.time() - start_time


    # class Timing:
    # def __enter__(self, attribute):
    #     self.set_attr = attribute
    #     self.start_time = time.time()

    # def __exit__(self, exc_type, exc_value, exc_tb):
    #     end_time = time.time() - self.start_time
    #     if self.set_attr is "training":
    #         self.training_time = end_time
    #     elif self.set_attr is "run":
    #         self.run_time = end_time
    #     else:
    #         raise KeyError(f"Timing expected one of training or run, got {self.set_attr}")

    # def __call__(self, fn):
    #     """
    #     Decorator to use if whole function needs resource checked out.
    #     """
    #     def decorator():
    #         with self:
    #             return fn()
    #     return decorator

state = State()
