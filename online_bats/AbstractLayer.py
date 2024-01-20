from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional
import cupy as cp

import numpy as np

WEIGHTS_FILE_SUFFIX = "_weights.npy"
FEEDBACKS_FILE_SUFFIX = "_feedbacks.npy"


class AbstractLayer(ABC):
    def __init__(self, n_neurons: int, name: str = ""):
        self._n_neurons: int = n_neurons
        self._name: str = name

    @property
    def n_neurons(self) -> int:
        return self._n_neurons

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def trainable(self) -> bool:
        pass

    @property
    @abstractmethod
    def spike_trains(self) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        pass

    @property
    @abstractmethod
    def weights(self) -> Optional[cp.ndarray]:
        pass

    @weights.setter
    @abstractmethod
    def weights(self, weights: np.ndarray) -> None:
        pass

    @property
    @abstractmethod
    def feedbacks(self) -> Optional[cp.ndarray]:
        pass

    @feedbacks.setter
    @abstractmethod
    def feedbacks(self, feedbacks: np.ndarray) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def forward(self, max_simulation: float, training: bool = False) -> None:
        pass

    @abstractmethod
    def compute_avg_gradient(self, errors) -> Optional[cp.ndarray]:
        pass

    @abstractmethod
    def compute_avg_feedback_gradient(self, dir_derivatives, errors) -> Optional[cp.ndarray]:
        pass

    @abstractmethod
    def add_deltas(self, delta_weights: cp.ndarray) -> None:
        pass

    @abstractmethod
    def add_feedback_deltas(self, feedback_delta: cp.ndarray) -> None:
        pass

    def store(self, dir_path: Path) -> None:
        if self.weights is None:
            return
        filename_weights = dir_path / (self._name + WEIGHTS_FILE_SUFFIX)
        np.save(filename_weights, self.weights.get())

        filename_feedbacks = dir_path / (self._name + FEEDBACKS_FILE_SUFFIX)
        np.save(filename_feedbacks, self.feedbacks.get())

    def restore(self, dir_path: Path) -> None:
        filename_weights = dir_path / (self._name + WEIGHTS_FILE_SUFFIX)
        filename_feedbacks = dir_path / (self._name + FEEDBACKS_FILE_SUFFIX)

        if filename_weights.exists():
            self.weights = np.load(filename_weights)

        if filename_feedbacks.exists():
            self.feedbacks = np.load(filename_feedbacks)
