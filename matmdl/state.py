"""
Module for optimization state indicators.
"""
import time

class State:

    def __init__(self):
        self.iterations = 0
        self.last_updated = time.time_ns()
        self.tell_time = 0.0
        self.run_time = 0.0

    def update_write(self):
        self.iterations += 1
        self.last_updated = time.time_ns()

    def update_read(self):
        self.last_updated = time.time_ns()

    def TimeRun(self):
        class TimeRun:
            def __enter__(innerself):
                innerself.tic = time.time()
            def __exit__(innerself, exc_type, exc_value, exc_tb):
                self.run_time = time.time() - innerself.tic
        return TimeRun

    def TimeTell(self):
        class TimeTell:
            def __enter__(innerself):
                innerself.tic = time.time()
            def __exit__(innerself, exc_type, exc_value, exc_tb):
                self.tell_time = time.time() - innerself.tic
        return TimeTell


state = State()
