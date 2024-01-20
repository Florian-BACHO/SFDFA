from typing import Callable, Tuple
from typing import Optional
import cupy as cp
import numpy as np

from online_bats.AbstractConvLayer import AbstractConvLayer
from online_bats.CudaKernels.Wrappers.Inference import *
from online_bats.CudaKernels.Wrappers.Inference.compute_spike_times_conv import compute_spike_times_conv
from online_bats.CudaKernels.Wrappers.Learning import update_feedbacks


class ConvLIFLayer(AbstractConvLayer):
    def __init__(self, previous_layer: AbstractConvLayer, filters_shape: np.ndarray, tau_s: float, theta: float,
                 delta_theta: float, n_outputs: int,
                 weight_initializer: Callable[[int, int, int, int], cp.ndarray] = None, max_n_spike: int = 32,
                 alpha: float = 0.999, reg_factor: float = 0.0, **kwargs):
        prev_x, prev_y, prev_c = previous_layer._neurons_shape.get()
        filter_x, filter_y, filter_c = filters_shape
        n_x = prev_x - filter_x + 1
        n_y = prev_y - filter_y + 1
        neurons_shape: cp.ndarray = np.array([n_x, n_y, filter_c], dtype=cp.int32)

        super().__init__(neurons_shape=neurons_shape, **kwargs)

        self.__filters_shape = cp.array([filter_c, filter_x, filter_y, prev_c], dtype=cp.int32)
        self.__previous_layer: AbstractConvLayer = previous_layer
        self.__tau_s: cp.float32 = cp.float32(tau_s)
        self.__tau: cp.float32 = cp.float32(2 * tau_s)
        self.__theta_tau: cp.float32 = cp.float32(theta / self.__tau)
        self.__delta_theta_tau: cp.float32 = cp.float32(delta_theta / self.__tau)
        if weight_initializer is None:
            self.__weights: cp.ndarray = cp.zeros((filter_c, filter_x, filter_y, prev_c), dtype=cp.float32)
        else:
            self.__weights: cp.ndarray = weight_initializer(filter_c, filter_x, filter_y, prev_c)
        self.__max_n_spike: cp.int32 = cp.int32(max_n_spike)

        self.__n_spike_per_neuron: Optional[cp.ndarray] = None
        self.__spike_times_per_neuron: Optional[cp.ndarray] = None
        self.__dir_derivative: Optional[cp.ndarray] = None
        self.__eligibility_traces: Optional[cp.ndarray] = None

        self.__c: Optional[cp.float32] = self.__theta_tau
        self.__n_outputs: int = n_outputs
        self.__alpha: cp.float32 = cp.float32(alpha)
        self.__feedback_weights = cp.zeros((filter_c, self.__n_outputs), dtype=cp.float32)
        self.__perturbations: Optional[cp.ndarray] = None
        self.__reg_factor: cp.float32 = cp.float32(reg_factor)

    @property
    def trainable(self) -> bool:
        return True

    @property
    def spike_trains(self) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        return self.__spike_times_per_neuron, self.__n_spike_per_neuron, self.__dir_derivative

    @property
    def weights(self) -> Optional[cp.ndarray]:
        return self.__weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        self.__weights = cp.array(weights, dtype=cp.float32)

    def reset(self) -> None:
        self.__n_spike_per_neuron = None
        self.__spike_times_per_neuron = None
        self.__dir_derivative = None
        self.__eligibility_traces = None
        self.__perturbations = None

    def forward(self, max_simulation: float, training: bool = False) -> None:
        pre_spike_per_neuron, pre_n_spike_per_neuron, pre_dir_derivatives = self.__previous_layer.spike_trains

        pre_exp_tau_s, pre_exp_tau = compute_pre_exps(pre_spike_per_neuron, self.__tau_s, self.__tau)

        batch_size = pre_spike_per_neuron.shape[0]

        # Sort spikes for inference
        new_shape, sorted_indices, spike_times_reshaped = get_sorted_spikes_indices(pre_spike_per_neuron,
                                                                                    pre_n_spike_per_neuron)

        if sorted_indices.size == 0:  # No input spike in the batch
            shape = (batch_size, self.n_neurons, self.__max_n_spike)
            self.__n_spike_per_neuron = cp.zeros((batch_size, self.n_neurons), dtype=cp.int32)
            self.__spike_times_per_neuron = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__perturbations = cp.zeros((batch_size, self.neurons_shape[2]), dtype=cp.float32)
        else:
            sorted_spike_indices = (sorted_indices.astype(cp.int32) // pre_spike_per_neuron.shape[2])
            sorted_spike_times = cp.take_along_axis(spike_times_reshaped, sorted_indices, axis=1)
            non_spike_mask = sorted_indices == -1
            sorted_spike_times[non_spike_mask] = cp.inf
            sorted_pre_exp_tau_s = cp.take_along_axis(cp.reshape(pre_exp_tau_s, new_shape), sorted_indices, axis=1)
            sorted_pre_exp_tau = cp.take_along_axis(cp.reshape(pre_exp_tau, new_shape), sorted_indices, axis=1)
            sorted_dir_derivatives = cp.take_along_axis(cp.reshape(pre_dir_derivatives, new_shape), sorted_indices,
                                                        axis=1)
            self.__perturbations = cp.random.normal(0.0, 1.0, (batch_size, int(self.neurons_shape[2])),
                                                    dtype=cp.float32)

            self.__n_spike_per_neuron, self.__spike_times_per_neuron, self.__dir_derivative, self.__eligibility_traces = \
                compute_spike_times_conv(sorted_spike_indices, sorted_spike_times,
                                         sorted_pre_exp_tau_s, sorted_pre_exp_tau, sorted_dir_derivatives,
                                         self.weights, self.__c, self.__delta_theta_tau,
                                         self.__tau, cp.float32(max_simulation), self.__max_n_spike,
                                         self.__previous_layer.neurons_shape, self.neurons_shape,
                                         self.__filters_shape, self.__perturbations)

    def compute_avg_gradient(self, dir_derivatives, errors, n_output_spikes) -> Optional[cp.ndarray]:
        batch_size = self.__eligibility_traces.shape[0]
        dir_derivatives = cp.reshape(dir_derivatives, (batch_size, 1, self.__n_outputs))
        n_spike_per_neuron_3d = cp.reshape(self.__n_spike_per_neuron, (batch_size, *self.neurons_shape.tolist()))
        n_spike_per_filter = cp.maximum(cp.sum(n_spike_per_neuron_3d, axis=(1, 2)), 1.0)

        perturbations = cp.expand_dims(self.__perturbations / n_spike_per_filter.astype(cp.float32), 2)

        d_v = cp.mean(dir_derivatives * perturbations, axis=0)

        update_feedbacks(self.__feedback_weights, d_v, self.__alpha)
        errors = errors * n_output_spikes.astype(cp.float32)

        errors = cp.reshape(errors, (batch_size, 1, self.__n_outputs))
        neuron_errors = cp.sum(self.__feedback_weights * errors, axis=2)
        #neuron_errors -= abs(neuron_errors) * self.__reg_factor * cp.square(self.__n_spike_per_neuron)
        grad = cp.mean(self.__eligibility_traces * neuron_errors.reshape(*neuron_errors.shape + (1,) * 3), axis=0)

        return grad

    def add_deltas(self, delta_weights: cp.ndarray) -> None:
        self.__weights += delta_weights
