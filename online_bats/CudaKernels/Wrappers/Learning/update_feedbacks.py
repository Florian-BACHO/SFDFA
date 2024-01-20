from typing import Tuple
from online_bats.CudaKernels.load_kernel import load_kernel
import cupy as cp

__update_feedbacks_kernel = cp.ElementwiseKernel("float32 d_v, float32 alpha",
                                                 "float32 feedback",
                                                 "feedback = feedback + alpha * (d_v - feedback)",
                                                 "update_feedback", no_return=True)


def update_feedbacks(feedbacks: cp.ndarray, d_v: cp.ndarray, alpha: cp.float32) -> None:
    __update_feedbacks_kernel(d_v, alpha, feedbacks)