from typing import Optional, Tuple
import numpy as np
import cupy as cp

from online_bats.AbstractConvLayer import AbstractConvLayer


class ConvInputLayer(AbstractConvLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__indices: Optional[cp.ndarray] = None
        self.__times_per_neuron: Optional[cp.ndarray] = None
        self.__n_spike_per_neuron: Optional[cp.ndarray] = None
        self.__dir_derivatives: Optional[cp.ndarray] = None

    @property
    def trainable(self) -> bool:
        return False

    @property
    def spike_trains(self) -> Tuple[cp.ndarray, cp.ndarray, Optional[cp.ndarray]]:
        return self.__times_per_neuron, self.__n_spike_per_neuron, self.__dir_derivatives

    def set_spike_trains(self, times_per_neuron: np.ndarray, n_times_per_neuron: np.ndarray) -> None:
        self.__times_per_neuron = cp.array(times_per_neuron, dtype=cp.float32)
        self.__n_spike_per_neuron = cp.array(n_times_per_neuron, dtype=cp.int32)
        self.__dir_derivatives = cp.zeros(times_per_neuron.shape, dtype=cp.float32)

    def compute_avg_gradient(self, dir_derivatives, errors, n_output_spikes) -> Optional[cp.ndarray]:
        return None
        
    @property
    def weights(self) -> Optional[cp.ndarray]:
        return None

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        pass
        
    def reset(self) -> None:
        pass

    def forward(self, max_simulation: float, training: bool = False) -> None:
        pass

    def add_deltas(self, delta_weights: cp.ndarray) -> None:
        pass