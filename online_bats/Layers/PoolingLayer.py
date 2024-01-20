from typing import Callable, Tuple
from typing import Optional
import cupy as cp
import numpy as np

from online_bats.AbstractConvLayer import AbstractConvLayer
from online_bats.CudaKernels.Wrappers.Inference import *


class PoolingLayer(AbstractConvLayer):
    def __init__(self, previous_layer: AbstractConvLayer, **kwargs):
        prev_x, prev_y, prev_c = previous_layer._neurons_shape.get()
        n_x = prev_x // 2
        n_y = prev_y // 2
        neurons_shape: cp.ndarray = np.array([n_x, n_y, prev_c], dtype=cp.int32)

        super().__init__(neurons_shape=neurons_shape, **kwargs)

        self.__previous_layer: AbstractConvLayer = previous_layer

        self.__n_spike_per_neuron: Optional[cp.ndarray] = None
        self.__spike_times_per_neuron: Optional[cp.ndarray] = None
        self.__spike_indices: Optional[cp.ndarray] = None
        self.__dir_derivative: Optional[cp.ndarray] = None

    @property
    def trainable(self) -> bool:
        return True

    @property
    def spike_trains(self) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        return self.__spike_times_per_neuron, self.__n_spike_per_neuron, self.__dir_derivative

    @property
    def weights(self) -> Optional[cp.ndarray]:
        return None

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        pass

    def reset(self) -> None:
        self.__n_spike_per_neuron = None
        self.__spike_times_per_neuron = None
        self.__dir_derivative = None

    def forward(self, max_simulation: float, training: bool = False) -> None:
        pre_spike_per_neuron, pre_n_spike_per_neuron, pre_dir_derivatives = self.__previous_layer.spike_trains
        self.__n_spike_per_neuron, self.__spike_times_per_neuron, self.__dir_derivative = \
            aggregate_spikes_conv(pre_n_spike_per_neuron, pre_spike_per_neuron, pre_dir_derivatives,
                                  self.__previous_layer.neurons_shape, self.neurons_shape)

    def compute_avg_gradient(self, dir_derivatives, errors, n_output_spikes) -> Optional[cp.ndarray]:
        return None

    def add_deltas(self, delta_weights: cp.ndarray) -> None:
        pass
