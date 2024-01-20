from typing import Optional, List
import cupy as cp
import numpy as np

from ..AbstractOptimizer import AbstractOptimizer

update_m_kernel = cp.ElementwiseKernel("float32 m, float32 beta_1, float32 one_minus_beta_1, float32 grad",
                                       "float32 new_m",
                                       "new_m = beta_1 * m + one_minus_beta_1 * grad",
                                       "update_m_kernel")
compute_deltas_kernel = cp.ElementwiseKernel("float32 grad, float32 m, float32 learning_rate",
                                             "float32 delta",
                                             "delta = -learning_rate * m",
                                             "compute_deltas_kernel")


class MomentumOptimizer(AbstractOptimizer):
    def __init__(self, beta_1=0.9, **kwargs):
        super().__init__(**kwargs)
        self.__beta_1: cp.float32 = cp.float32(beta_1)
        self.__one_minus_beta_1: cp.float32 = cp.float32(1.0 - beta_1)

        self.__m: Optional[List[List[cp.array]]] = None
        self.__t: cp.int32 = cp.int32(0)

    def step(self, gradient: List[cp.ndarray]) -> List[cp.ndarray]:
        self.__t += 1

        # Set m and v to 0 at first iteration
        if self.__m is None:
            self.__m = [None if grad is None else cp.zeros(grad.shape, dtype=cp.float32) for grad in gradient]
        # Update m and v
        self.__m = [None if grad is None else update_m_kernel(pre_m, self.__beta_1, self.__one_minus_beta_1, grad)
                    for pre_m, grad in zip(self.__m, gradient)]

        return [None if g is None else compute_deltas_kernel(g, m, self._learning_rate)
                for g, m in zip(gradient, self.__m)]
