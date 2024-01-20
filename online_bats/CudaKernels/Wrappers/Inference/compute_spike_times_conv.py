from online_bats.CudaKernels.load_kernel import load_kernel
import cupy as cp
import numpy as np

KERNEL_FILE = "Inference/compute_spike_times_conv.cu"
KERNEL_NAME = "compute_spike_times_conv_kernel"

__compute_spike_times_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)


def compute_spike_times_conv(spike_indices: cp.ndarray, spike_times: cp.ndarray,
                             exp_tau_s: cp.ndarray, exp_tau: cp.ndarray, pre_dir_derivatives, weights: cp.ndarray,
                             c: cp.float32, delta_theta_tau: np.float32, tau: np.float32,
                             max_simulation: np.float32, max_n_post_spikes: np.int32,
                             pre_shape: cp.ndarray, post_shape: cp.ndarray, filters_shape: cp.ndarray,
                             perturbations: cp.ndarray):
    batch_size, max_n_pre_spike = spike_times.shape
    n_neuron_x, n_neuron_y, n_neuron_c = post_shape.get()
    n_neurons = n_neuron_x * n_neuron_y * n_neuron_c

    block_dim = (batch_size, 1, 1)
    grid_dim = (n_neuron_x, n_neuron_y, n_neuron_c)

    res_shape = (batch_size, n_neurons, max_n_post_spikes)
    n_spikes = cp.zeros((batch_size, n_neurons), dtype=cp.int32)
    post_spike_times = cp.full(res_shape, cp.inf, dtype=cp.float32)
    post_dir_derivatives = cp.zeros(res_shape, dtype=cp.float32)

    _, filter_x, filter_y, prev_c = filters_shape
    grad_shape = (batch_size, n_neuron_x, n_neuron_y, n_neuron_c, int(filter_x), int(filter_y), int(prev_c))
    trace_a = cp.zeros(grad_shape, dtype=cp.float32)
    trace_b = cp.zeros(grad_shape, dtype=cp.float32)
    eligibility_trace = cp.zeros(grad_shape, dtype=cp.float32)

    args = (spike_indices, spike_times, exp_tau_s, exp_tau, pre_dir_derivatives, weights, perturbations,
            pre_shape, post_shape, filters_shape,
            n_neurons, c, delta_theta_tau, tau, max_simulation, max_n_pre_spike, max_n_post_spikes,
            n_spikes, post_spike_times, post_dir_derivatives, trace_a, trace_b, eligibility_trace)
    __compute_spike_times_kernel(grid_dim, block_dim, args)
    eligibility_trace = cp.sum(eligibility_trace, axis=(1, 2))

    return n_spikes, post_spike_times, post_dir_derivatives, eligibility_trace
