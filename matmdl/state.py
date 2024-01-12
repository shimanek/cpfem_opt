"""
Module for optimization state indicators.
"""
import time

class State:

    def __init__(self):
        self.iterations = 0
        self.last_updated = time.time_ns()

    def update_write(self):
        self.iterations += 1
        self.last_updated = time.time_ns()

    def update_read(self):
        self.last_updated = time.time_ns()

state = State()
