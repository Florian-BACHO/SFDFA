from typing import Callable, Tuple
from typing import Optional
import cupy as cp
import numpy as np

from online_bats.AbstractLayer import AbstractLayer
from online_bats.CudaKernels.Wrappers.Inference import *
from online_bats.CudaKernels.Wrappers.Learning import update_feedbacks


class LIFLayer(AbstractLayer):
    def __init__(self, previous_layer: AbstractLayer, tau_s: float, theta: float, n_outputs: int,
                 kernel_type: str = "STDP",
                 weight_initializer: Callable[[int, int], cp.ndarray] = None,
                 feedback_initializer: Callable[[int, int], cp.ndarray] = None, max_n_spike: int = 32,
                 reg_factor: float = 0.0, mean_fr_target: float = 0.0, normalize_grad: bool = False,
                 is_output: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.__previous_layer: AbstractLayer = previous_layer
        self.__tau_s: cp.float32 = cp.float32(tau_s)
        self.__tau: cp.float32 = cp.float32(2 * tau_s)
        self.__threshold: cp.float32 = cp.float32(theta)
        self.__theta_tau: cp.float32 = cp.float32(theta / self.__tau)

        if kernel_type == "STDP":
            self.__kernel: Callable = compute_spike_times_stdp
        elif kernel_type == "MP":
            self.__kernel: Callable = compute_spike_times_mp
        elif kernel_type == "MP-NR":
            self.__kernel: Callable = compute_spike_times_mp_no_reset
        elif kernel_type == "STD":
            self.__kernel: Callable = compute_spike_times_std
        elif kernel_type == "NO-ALPHA":
            self.__kernel: Callable = compute_spike_times_no_alpha
        elif kernel_type == "STD-NR":
            self.__kernel: Callable = compute_spike_times_std_no_reset
        else:
            raise Exception("Invalid kernel")
        if weight_initializer is None:
            self.__weights: cp.ndarray = cp.zeros((self.n_neurons, previous_layer.n_neurons), dtype=cp.float32)
        else:
            self.__weights: cp.ndarray = weight_initializer(self.n_neurons, previous_layer.n_neurons)
        self.__max_n_spike: cp.int32 = cp.int32(max_n_spike)

        self.__n_spike_per_neuron: Optional[cp.ndarray] = None
        self.__spike_times_per_neuron: Optional[cp.ndarray] = None
        self.__dir_derivative: Optional[cp.ndarray] = None
        self.eligibility_traces: Optional[cp.ndarray] = None

        self.__c: cp.float32 = self.__theta_tau
        self.__n_outputs: int = n_outputs
        self.__feedback_weights: Optional[cp.ndarray]
        if is_output:
            self.__feedback_weights = cp.identity(self.n_neurons, dtype=cp.float32)
        elif feedback_initializer is not None:
            self.__feedback_weights = feedback_initializer(self.n_neurons, self.__n_outputs)
        else:
            self.__feedback_weights = cp.zeros((self.n_neurons, self.__n_outputs), dtype=cp.float32)
        self.__perturbations: Optional[cp.ndarray] = None
        self.__normalize_grad: bool = normalize_grad
        self.__is_output: bool = is_output
        self.__reg_factor: cp.float32 = cp.float32(reg_factor)
        self.__mean_fr_target: cp.float32 = cp.float32(mean_fr_target)
        self.neuron_errors = None

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

    @property
    def feedbacks(self) -> cp.ndarray:
        return self.__feedback_weights

    @feedbacks.setter
    def feedbacks(self, feedbacks: np.ndarray) -> None:
        self.__feedback_weights = cp.array(feedbacks, dtype=cp.float32)

    def reset(self) -> None:
        self.neuron_errors = None
        self.__n_spike_per_neuron = None
        self.__spike_times_per_neuron = None
        self.__dir_derivative = None
        self.eligibility_traces = None
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
            self.__dir_derivative = cp.zeros(shape, dtype=cp.float32)
            self.eligibility_traces = cp.zeros((batch_size, *self.weights.shape), dtype=cp.float32)
            self.__perturbations = cp.zeros((batch_size, self.n_neurons), dtype=cp.float32)
        else:
            sorted_spike_indices = (sorted_indices.astype(cp.int32) // pre_spike_per_neuron.shape[2])
            sorted_spike_times = cp.take_along_axis(spike_times_reshaped, sorted_indices, axis=1)
            non_spike_mask = sorted_indices == -1
            sorted_spike_times[non_spike_mask] = cp.inf
            sorted_pre_exp_tau_s = cp.take_along_axis(cp.reshape(pre_exp_tau_s, new_shape), sorted_indices, axis=1)
            sorted_pre_exp_tau = cp.take_along_axis(cp.reshape(pre_exp_tau, new_shape), sorted_indices, axis=1)

            if training:
                sorted_dir_derivatives = cp.take_along_axis(cp.reshape(pre_dir_derivatives, new_shape), sorted_indices,
                                                            axis=1)

                if self.__is_output:
                    self.__perturbations = cp.zeros((batch_size, self.n_neurons), dtype=cp.float32)
                else:
                    self.__perturbations = cp.random.normal(0.0, 1.0, (batch_size, self.n_neurons),
                                                            dtype=cp.float32)

                self.__n_spike_per_neuron, self.__spike_times_per_neuron, self.__dir_derivative, self.eligibility_traces = \
                    self.__kernel(sorted_spike_indices, sorted_spike_times, sorted_pre_exp_tau_s, sorted_pre_exp_tau,
                                  sorted_dir_derivatives, self.weights, self.__c,
                                  self.__tau, cp.float32(max_simulation), self.__max_n_spike,
                                  self.__perturbations, self.__is_output)

                # self.__dir_derivative /= cp.maximum(1.0, self.__n_spike_per_neuron.astype(cp.float32)[:, :, cp.newaxis])
            else:
                self.__n_spike_per_neuron, self.__spike_times_per_neuron, self.__dir_derivative = \
                    compute_spike_times_inference(sorted_spike_indices, sorted_spike_times, sorted_pre_exp_tau_s,
                                                  sorted_pre_exp_tau, self.weights, self.__c, self.__tau,
                                                  cp.float32(max_simulation), self.__max_n_spike)

    def compute_avg_gradient(self, errors) -> Optional[cp.ndarray]:
        errors = cp.expand_dims(errors, axis=1)

        _, pre_n_spike_per_neuron, _ = self.__previous_layer.spike_trains

        if self.__normalize_grad:
            self.eligibility_traces /= cp.expand_dims(cp.maximum(self.__n_spike_per_neuron.astype(cp.float32), 1.0), 2)

        self.neuron_errors = cp.sum(self.feedbacks * errors, axis=2)

        if not self.__is_output:
            mean_fr = cp.mean(self.__n_spike_per_neuron.astype(cp.float32), axis=0)
            self.neuron_errors = cp.zeros_like(self.neuron_errors) + self.__reg_factor * \
                                 (self.__mean_fr_target - mean_fr)

        grad = self.eligibility_traces * cp.expand_dims(self.neuron_errors, 2)

        return cp.mean(grad, axis=0)

    def compute_avg_feedback_gradient(self, dir_derivatives, errors) -> Optional[cp.ndarray]:
        if self.__is_output:
            return None
        dir_derivatives = cp.expand_dims(dir_derivatives, axis=1)
        mean_perturbations = cp.expand_dims(self.__perturbations, axis=2)
        """mean_perturbations = cp.mean(self.__perturbations * cp.isfinite(self.__spike_times_per_neuron),
                                    axis=2, keepdims=True)"""

        d_v = dir_derivatives * mean_perturbations

        active_neurons = cp.expand_dims(self.__n_spike_per_neuron > 0, 2).astype(cp.float32)
        feedback_grad = (self.__feedback_weights - d_v) * active_neurons
        return cp.sum(feedback_grad, axis=0) / cp.maximum(1.0, cp.sum(active_neurons, axis=0))

    def add_deltas(self, delta_weights: cp.ndarray) -> None:
        self.__weights += delta_weights

    def add_feedback_deltas(self, feedback_delta: cp.ndarray) -> None:
        self.__feedback_weights += feedback_delta
