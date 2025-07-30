import numpy as np
from typing import List

from offline.data import GazeData
from offline.modules import Module


class FixationMetrics(Module):
    def update(self, data: GazeData) -> GazeData:
        fixation_metrics = {
            "PFT": 0,
            "MFD": 0,
            "FR": 0,
            "PFD": 0,
        }

        if len(data.fixations) > 0:
            fixation_metrics["count"] = len(data.fixations)

            durations = np.array([fixation["duration"] for fixation in data.fixations])

            fixation_metrics["PFT"] = np.sum(durations) / data.get_total_duration()
            fixation_metrics["MFD"] = np.mean(durations)
            fixation_metrics["FR"] = len(data.fixations) / data.get_total_duration()
            fixation_metrics["PFD"] = np.max(durations)

        data.fixation_metrics = fixation_metrics

        return data


class SaccadeMetrics(Module):
    def update(self, data: GazeData) -> GazeData:
        saccade_metrics = {
            "duration_mean": 0,
            "MSA": 0,
            "MSV": 0,
            "MPSV": 0,
        }

        if len(data.saccades) > 0:
            saccade_metrics["count"] = len(data.saccades)

            durations = np.array([saccade["duration"] for saccade in data.saccades])
            saccade_metrics["duration_mean"] = np.mean(durations)

            amplitudes = np.array([saccade["amplitude"] for saccade in data.saccades])
            saccade_metrics["MSA"] = np.mean(amplitudes)

            velocities = np.array([saccade["velocity"] for saccade in data.saccades])
            saccade_metrics["MSV"] = np.mean(velocities)

            peak_velocities = np.array(
                [saccade["peak_velocity"] for saccade in data.saccades]
            )
            saccade_metrics["MPSV"] = np.mean(peak_velocities)

        data.saccade_metrics = saccade_metrics

        return data


class BlinkMetrics(Module):
    def __init__(self, min_count: int = 2) -> None:
        self.min_count = min_count

    def update(self, data: GazeData) -> GazeData:
        reduced_total_duration = 0
        all_chunk_lengths = []
        if hasattr(data, "blink_indices"):
            blink_indices = data.blink_indices
            # print("blink indices: ", blink_indices)
            count = 0
            chunks = []
            for i in range(len(blink_indices)):
                if i == 0 or blink_indices[i] != blink_indices[i - 1] + 1:
                    chunks.append([])
                chunks[-1].append(blink_indices[i])
            # merge chunks if they are too close, i.e. 1 frame apart
            for i in range(len(chunks) - 1):
                if chunks[i + 1][0] - chunks[i][-1] <= 1:
                    chunks[i] += chunks[i + 1]
                    chunks[i + 1] = []
            for chunk in chunks:
                if len(chunk) >= self.min_count:
                    if len(chunk) > 40:  # if too long, exclude it from the data
                        reduced_total_duration += (
                            data.start_timestamp[
                                np.where(data.indices == chunk[-1])[0][0]
                            ]
                            - data.start_timestamp[
                                np.where(data.indices == chunk[0])[0][0]
                            ]
                        )
                        continue
                    all_chunk_lengths.append(len(chunk))
                    count += 1
        else:
            count = -1

        blink_metrics = {
            "count": count,
            "BR": count / (data.get_total_duration() - reduced_total_duration),
        }

        data.blink_metrics = blink_metrics
        return data


class DiameterMetrics(Module):
    def __init__(self) -> None:
        pass

    def update(self, data: GazeData) -> GazeData:
        if hasattr(data, "left_pupil_diameter") and hasattr(
            data, "right_pupil_diameter"
        ):
            left_diameter_mean = np.mean(data.left_pupil_diameter)
            left_diameter_var = np.var(data.left_pupil_diameter)
            right_diameter_mean = np.mean(data.right_pupil_diameter)
            right_diameter_var = np.var(data.right_pupil_diameter)
            combined_diameter_mean = (left_diameter_mean + right_diameter_mean) / 2
            combined_diameter_var = (left_diameter_var + right_diameter_var) / 2
        else:
            combined_diameter_mean, combined_diameter_var = -1, -1
        diameter_metrics = {"MD": combined_diameter_mean, "VD": combined_diameter_var}
        data.diameter_metrics = diameter_metrics
        return data


class ROIMetrics(Module):
    def __init__(self, rois: dict) -> None:
        self.rois = rois

    def update(self, data: GazeData) -> GazeData:
        roi_metrics = {}
        data_total_duration = data.get_total_duration()
        for roi in self.rois:  #  + ["other"]:
            roi_name = self.rois[roi]
            roi_fixations = [
                fixation for fixation in data.fixations if fixation["target"] == roi
            ]
            roi_fixation_durations = [
                fixation["duration"] for fixation in roi_fixations
            ]
            roi_fixation_mean_duration = (
                np.mean(roi_fixation_durations)
                if len(roi_fixation_durations) > 0
                else 0
            )
            roi_fixation_total_duration = np.sum(roi_fixation_durations)
            roi_fixation_count = len(roi_fixations)

            roi_metrics[f"{roi_name}MFD"] = (
                roi_fixation_mean_duration / data.fixation_metrics["MFD"]
                if data.fixation_metrics["MFD"] != 0
                else 0
            )
            roi_metrics[f"{roi_name}FR"] = (
                roi_fixation_count / data_total_duration / data.fixation_metrics["FR"]
                if data.fixation_metrics["FR"] != 0
                else 0
            )
            roi_metrics[f"PF{roi_name}"] = (
                roi_fixation_total_duration
                / data_total_duration
                / data.fixation_metrics["PFT"]
                if data.fixation_metrics["PFT"] != 0
                else 0
            )
        data.roi_metrics = roi_metrics

        return data
