from typing import Tuple

from ..AbstractLoss import AbstractLoss
import cupy as cp


class SoftmaxCrossEntropy(AbstractLoss):
    def __init__(self, max_n_spikes):
        self.__exp_kernel = cp.ElementwiseKernel("float32 t",
                                                 "float32 out",
                                                 f"out = __expf(t / {max_n_spikes})",
                                                 "sce_exp_kernel")

        self.__cross_entropy_kernel = cp.ElementwiseKernel("float32 labels_exps, float32 sums",
                                                           "float32 out",
                                                           "out = - __logf(labels_exps / sums)",
                                                           "sce_cross_entropy_kernel")

    def predict(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray) -> cp.ndarray:
        return cp.argmax(n_spike_per_neuron, axis=1)

    def compute_loss(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                     labels: cp.ndarray) -> cp.ndarray:
        exps = self.__exp_kernel(n_spike_per_neuron.astype(cp.float32))
        sums = cp.sum(exps, axis=1)
        labels_exps = exps[cp.arange(labels.size), labels]
        return self.__cross_entropy_kernel(labels_exps, sums)

    def compute_directional_derivative(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                                       dir_derivatives: cp.ndarray, labels: cp.ndarray) \
            -> Tuple[cp.ndarray, cp.ndarray]:
        exps = self.__exp_kernel(n_spike_per_neuron.astype(cp.float32))
        sums = cp.sum(exps, axis=1)
        neg_softmax = -exps / sums[:, cp.newaxis]
        neg_softmax[cp.arange(labels.size), labels] += 1
        neurons_errors = cp.nan_to_num(neg_softmax)

        sum_derivatives = cp.sum(dir_derivatives, axis=2)
        return sum_derivatives, neurons_errors

    def compute_loss_and_directional_derivative(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                                                dir_derivatives: cp.ndarray, labels: cp.ndarray) \
            -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        exps = self.__exp_kernel(n_spike_per_neuron.astype(cp.float32))
        sums = cp.sum(exps, axis=1)

        # Loss
        labels_exps = exps[cp.arange(labels.size), labels]
        loss = self.__cross_entropy_kernel(labels_exps, sums)

        # Error
        neg_softmax = -exps / sums[:, cp.newaxis]
        neg_softmax[cp.arange(labels.size), labels] += 1
        neurons_errors = cp.nan_to_num(neg_softmax)

        sum_derivatives = cp.sum(dir_derivatives, axis=2)
        return loss, sum_derivatives, neurons_errors
