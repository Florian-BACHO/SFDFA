from typing import Tuple

from ..AbstractLoss import AbstractLoss
import cupy as cp


class SpikeCountLoss(AbstractLoss):
    def __init__(self):
        self.__loss_kernel = cp.ReductionKernel("float32 out_count, float32 out_target",
                                                "float32 loss",
                                                "(out_target - out_count) * (out_target - out_count)",
                                                "a + b",
                                                "loss = a / 2",
                                                "0",
                                                "loss_kernel")

    def predict(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray) -> cp.ndarray:
        return cp.argmax(n_spike_per_neuron, axis=1)

    def compute_loss(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                     targets: cp.ndarray) -> cp.ndarray:
        targets = cp.array(targets, dtype=cp.float32)
        float_n_spike_per_neuron = n_spike_per_neuron.astype(cp.float32)
        return self.__loss_kernel(float_n_spike_per_neuron, targets, axis=1)

    def compute_directional_derivative(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                                       dir_derivatives: cp.ndarray, targets: cp.ndarray) \
            -> Tuple[cp.ndarray, cp.ndarray]:
        targets = cp.array(targets, dtype=cp.float32)
        sum_derivatives = cp.sum(dir_derivatives, axis=2)
        neurons_errors = targets - n_spike_per_neuron.astype(cp.float32)
        return sum_derivatives, neurons_errors

    def compute_loss_and_directional_derivative(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                                                dir_derivatives: cp.ndarray, targets: cp.ndarray) \
            -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        targets = cp.array(targets, dtype=cp.float32)
        float_n_spike_per_neuron = n_spike_per_neuron.astype(cp.float32)
        neurons_errors = targets - float_n_spike_per_neuron
        loss = self.__loss_kernel(float_n_spike_per_neuron, targets, axis=1)
        sum_derivatives = cp.sum(dir_derivatives, axis=2)# / cp.maximum(n_spike_per_neuron.astype(cp.float32), 1.0)
        return loss, sum_derivatives, neurons_errors