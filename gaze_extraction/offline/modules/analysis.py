import numpy as np
from typing import Optional, Dict

from offline.data import GazeData
from offline.modules import Module


class IVTFixationDetector(Module):
    def __init__(
        self,
        velocity_threshold: float = 30,
        use_mobility=True,
        mobile_velocity_threshold: Optional[float] = None,
    ) -> None:
        self.velocity_threshold = velocity_threshold
        self.use_mobility = use_mobility
        self.mobile_velocity_threshold = mobile_velocity_threshold

    def update(self, data: GazeData) -> GazeData:
        data.fixation = data.velocity < self.velocity_threshold
        if self.use_mobility:
            data.fixation = np.logical_or(
                data.fixation,
                np.logical_and(
                    data.mobility_mask, data.velocity < self.mobile_velocity_threshold
                ),
            )
        if hasattr(data, "blink"):
            data.fixation = np.logical_and(data.fixation, ~data.blink)

        return data


class BlinkConverter(Module):
    def __init__(self) -> None:
        pass

    def update(self, data: GazeData) -> GazeData:
        # convert blink indices to a numpy list of booleans
        data.blink = np.zeros(len(data), dtype=bool)
        if len(data.blink_indices) > 0:
            data.blink[np.array(data.blink_indices) - data.first_index] = True
        return data


class IVTSaccadeDetector(Module):
    def __init__(self, velocity_threshold: float = 30) -> None:
        self.velocity_threshold = velocity_threshold

    def update(self, data: GazeData) -> GazeData:
        data.saccade = ~data.fixation
        if hasattr(data, "blink"):
            data.saccade = np.logical_and(data.saccade, ~data.blink)
        return data
