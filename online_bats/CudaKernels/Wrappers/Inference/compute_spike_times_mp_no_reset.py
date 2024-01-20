from online_bats.CudaKernels.load_kernel import load_kernel
import cupy as cp
import numpy as np

KERNEL_FILE = "Inference/compute_spike_times_mp_no_reset.cu"
KERNEL_NAME = "compute_spike_times_kernel"

__compute_spike_times_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)


def compute_spike_times_mp_no_reset(spike_indices: cp.ndarray, spike_times: cp.ndarray,
                                    exp_tau_s: cp.ndarray, exp_tau: cp.ndarray, pre_dir_derivatives,
                                    weights: cp.ndarray,
                                    c: cp.float32, tau: np.float32,
                                    max_simulation: np.float32, max_n_post_spikes: np.int32, perturbations: cp.ndarray):
    batch_size, max_n_pre_spike = spike_indices.shape
    n_post_neurons, n_pre_neurons = weights.shape
    block_dim = (n_post_neurons, 1, 1)
    grid_dim = (batch_size, 1, 1)

    res_shape = (batch_size, n_post_neurons, max_n_post_spikes)
    n_spikes = cp.zeros((batch_size, n_post_neurons), dtype=cp.int32)
    post_spike_times = cp.full(res_shape, cp.inf, dtype=cp.float32)
    post_dir_derivatives = cp.zeros(res_shape, dtype=cp.float32)

    grad_shape = (batch_size, n_post_neurons, n_pre_neurons)
    trace_a = cp.zeros(grad_shape, dtype=cp.float32)
    trace_b = cp.zeros(grad_shape, dtype=cp.float32)
    eligibility_trace = cp.zeros(grad_shape, dtype=cp.float32)

    args = (spike_indices, spike_times, exp_tau_s, exp_tau, pre_dir_derivatives, weights, perturbations,
            c, tau, max_simulation, max_n_pre_spike, max_n_post_spikes, n_pre_neurons,
            n_spikes, post_spike_times, post_dir_derivatives, trace_a, trace_b, eligibility_trace)
    __compute_spike_times_kernel(grid_dim, block_dim, args)

    return n_spikes, post_spike_times, post_dir_derivatives, eligibility_trace
