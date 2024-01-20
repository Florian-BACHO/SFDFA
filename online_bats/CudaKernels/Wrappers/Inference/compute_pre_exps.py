from typing import Tuple
from online_bats.CudaKernels.load_kernel import load_kernel
import cupy as cp

KERNEL_FILE = "Inference/compute_pre_exps.cu"
KERNEL_NAME = "compute_pre_exps_kernel"

__compute_pre_exps_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)

def compute_pre_exps(spike_times: cp.ndarray, tau_s: cp.float32, tau: cp.float32) -> Tuple[cp.ndarray, cp.ndarray]:
    batch_size, n_neurons, max_n_spike = spike_times.shape
    n_neuron_block = n_neurons // 512 + 1
    block_dim = (512, 2, 1)
    grid_dim = (batch_size, n_neuron_block, max_n_spike)

    exp_tau_s = cp.ndarray(spike_times.shape, dtype=cp.float32)
    exp_tau = cp.ndarray(spike_times.shape, dtype=cp.float32)

    __compute_pre_exps_kernel(grid_dim, block_dim, (spike_times, exp_tau_s, exp_tau, tau_s, tau, cp.int32(n_neurons)))
    return exp_tau_s, exp_tau