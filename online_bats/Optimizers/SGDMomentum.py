from typing import Optional, List
import cupy as cp
import numpy as np

from ..AbstractOptimizer import AbstractOptimizer

update_v_kernel = cp.ElementwiseKernel("float32 m, float32 beta, float32 one_minus_beta, float32 grad",
                                       "float32 new_m",
                                       "new_m = beta * m + one_minus_beta * grad",
                                       "update_m_kernel")


class SGDMomentumOptimizer(AbstractOptimizer):
    def __init__(self, beta=0.9, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.__beta: cp.float32 = cp.float32(beta)
        self.__one_minus_beta: cp.float32 = cp.float32(1.0 - beta)

        self.__m: Optional[List[List[cp.array]]] = None

    def step(self, gradient: List[cp.ndarray]) -> List[cp.ndarray]:
        # Set m to 0 at first iteration
        if self.__m is None:
            self.__m = [None if grad is None else cp.zeros(grad.shape, dtype=cp.float32) for grad in gradient]
        # Update m
        self.__m = [None if grad is None else update_v_kernel(pre_m, self.__beta, self.__one_minus_beta, grad)
                    for pre_m, grad in zip(self.__m, gradient)]

        return [None if m is None else -self._learning_rate * m for m in self.__m]
