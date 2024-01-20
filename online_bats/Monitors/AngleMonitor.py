import numpy as np
from scipy.spatial import distance

from online_bats.AbstractMonitor import AbstractMonitor

class AngleMonitor(AbstractMonitor):
    def __init__(self, layer_name: str, **kwargs):
        super().__init__(layer_name + " angle (deg.)", **kwargs)
        self._angles = []

    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def add(self, v1, v2) -> None:
        angle = np.rad2deg(self.angle_between(v1, v2))
        if np.isnan(angle):
            return
        self._angles.append(angle)

    def record(self, epoch) -> float:
        mean_angle = np.mean(self._angles)
        super()._record(epoch, mean_angle)
        self._angles = []
        return mean_angle