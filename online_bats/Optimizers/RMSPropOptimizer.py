from typing import Optional, List
import cupy as cp
import numpy as np

from ..AbstractOptimizer import AbstractOptimizer

update_v_kernel = cp.ElementwiseKernel("float32 v, float32 beta, float32 one_minus_beta, float32 grad",
                                       "float32 new_v",
                                       "new_v = beta * v + one_minus_beta * grad * grad",
                                       "update_v_kernel")
compute_deltas_kernel = cp.ElementwiseKernel("float32 grad, float32 v, float32 learning_rate,"
                                             "float32 epsilon",
                                             "float32 delta",
                                             "delta = -(learning_rate * grad / (sqrtf(v) + epsilon))",
                                             "compute_deltas_kernel")


class RMSPropOptimizer(AbstractOptimizer):
    def __init__(self, beta=0.9, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.__beta: cp.float32 = cp.float32(beta)
        self.__one_minus_beta: cp.float32 = cp.float32(1.0 - beta)
        self.__epsilon: cp.float32 = cp.float32(epsilon)

        self.__v: Optional[List[List[cp.array]]] = None

    def step(self, gradient: List[cp.ndarray]) -> List[cp.ndarray]:

        # Set v to 0 at first iteration
        if self.__v is None:
            self.__v = [None if grad is None else cp.zeros(grad.shape, dtype=cp.float32) for grad in gradient]
        # Update v
        self.__v = [None if grad is None else update_v_kernel(pre_v, self.__beta, self.__one_minus_beta, grad)
                    for pre_v, grad in zip(self.__v, gradient)]

        return [None if g is None else compute_deltas_kernel(g, v, self._learning_rate, self.__epsilon)
                for g, v in zip(gradient, self.__v)]
