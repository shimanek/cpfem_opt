"""
Module for optimization state meta-indicators, like iteration number and time.
"""
import time
import warnings

class State:
    """
    Contains and updates iteration and timing for each optimization process.

    Attributes:
        iterations (int): number of iterations performed by this process
        last_updated (int): time in unix nanoseconds of the last update to the
            optimizer state from any process
        tell_time (float): duration of time in seconds for the opt.tell process
        run_time (float): duration of time in seconds for a single iteration of the
            run process

    Note:
        Warns when `tell_time` > `run_time` but does not change behavior.
    """

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
                new_time_tell = time.time() - innerself.tic
                if new_time_tell > self.run_time:
                    warnings.warn(f"Taking longer to tell than to run: {new_time_tell:.1f} vs {self.run_time:.1f} seconds.", RuntimeWarning)
                self.tell_time = new_time_tell
        return TimeTell


state = State()
