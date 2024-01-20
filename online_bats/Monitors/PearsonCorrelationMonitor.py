import numpy as np
import cupy as cp

from online_bats.AbstractMonitor import AbstractMonitor


class PearsonCorrelationMonitor(AbstractMonitor):
    def __init__(self, name: str, **kwargs):
        super().__init__(name + " correlation", **kwargs)
        self.__correlation = None

    def add(self, var1: cp.ndarray, var2: cp.ndarray) -> None:
        self.__correlation = (cp.mean((var1 - var1.mean()) * (var2 - var2.mean())) /
                              (var1.std() * var2.std())).get()
        if np.isnan(self.__correlation):
            self.__correlation = 0.0

    def record(self, epoch) -> float:
        super()._record(epoch, self.__correlation)
        return self.__correlation
