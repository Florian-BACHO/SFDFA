from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cupy as cp

from online_bats import AbstractLayer
from online_bats.Layers import InputLayer


class Network:
    def __init__(self):
        self.__layers: List[AbstractLayer] = []
        self.__input_layer: Optional[InputLayer] = None

    @property
    def layers(self) -> List[AbstractLayer]:
        return self.__layers

    @property
    def output_spike_trains(self) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        return self.__layers[-1].spike_trains

    def add_layer(self, layer: AbstractLayer, input: bool = False) -> None:
        self.__layers.append(layer)
        if input:
            self.__input_layer = layer

    def reset(self):
        for layer in self.__layers:
            layer.reset()

    def forward(self, spikes_per_neuron: np.ndarray, n_spikes_per_neuron: np.ndarray,
                max_simulation: float = np.inf, training: bool = False) -> None:
        self.__input_layer.set_spike_trains(spikes_per_neuron, n_spikes_per_neuron)
        for layer in self.__layers:
            layer.forward(max_simulation, training)

    def compute_avg_gradient(self, errors: cp.ndarray) -> List[Optional[cp.ndarray]]:
        return [layer.compute_avg_gradient(errors) for layer in self.__layers]

    def compute_avg_feedback_gradient(self, dir_derivatives: cp.ndarray, errors: cp.ndarray) \
            -> List[Optional[cp.ndarray]]:
        return [layer.compute_avg_feedback_gradient(dir_derivatives, errors) for layer in self.__layers]

    def apply_deltas(self, deltas: List[cp.array]) -> None:
        for layer, deltas in zip(self.__layers, deltas):
            if deltas is None:
                continue
            layer.add_deltas(deltas)

    def apply_feedback_deltas(self, feedback_deltas: List[cp.array]) -> None:
        for layer, deltas in zip(self.__layers, feedback_deltas):
            if deltas is None:
                continue
            layer.add_feedback_deltas(deltas)

    def store(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        for it in self.__layers:
            it.store(dir_path)

    def restore(self, dir_path: Path) -> None:
        for it in self.__layers:
            it.restore(dir_path)
