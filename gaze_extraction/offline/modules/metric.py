import numpy as np
from typing import List

from offline.data import GazeData
from offline.modules import Module
from collections import defaultdict


class FixationMetrics(Module):
    def update(self, data: GazeData) -> GazeData:
        fixation_metrics = {}

        if len(data.fixations) > 0:
            fixation_metrics["count"] = len(data.fixations)

            durations = np.array([fixation["duration"] for fixation in data.fixations])

            fixation_metrics["PFT"] = np.sum(durations) / data.get_total_duration()
            fixation_metrics["MFD"] = np.mean(durations)
            fixation_metrics["SFD"] = np.std(durations)
            fixation_metrics["FR"] = len(data.fixations) / data.get_total_duration()

        data.fixation_metrics = fixation_metrics

        return data


class AmountMetrics(Module):
    def update(self, data: GazeData) -> GazeData:
        amount_metrics = {
            "left_mean": 0,
            "left_std": 0,
            "right_mean": 0,
            "right_std": 0,
        }
        left_amount_cleaned = [
            amount for amount in data.left_amount if amount != 10 and amount != -10
        ]
        right_amount_cleaned = [
            amount for amount in data.right_amount if amount != 10 and amount != -10
        ]

        amount_metrics["left_mean"] = (
            np.mean(left_amount_cleaned) if len(left_amount_cleaned) > 0 else -1
        )
        amount_metrics["left_std"] = (
            np.std(left_amount_cleaned) if len(left_amount_cleaned) > 0 else -1
        )
        amount_metrics["right_mean"] = (
            np.mean(right_amount_cleaned) if len(right_amount_cleaned) > 0 else -1
        )
        amount_metrics["right_std"] = (
            np.std(right_amount_cleaned) if len(right_amount_cleaned) > 0 else -1
        )

        data.amount_metrics = amount_metrics
        return data


class SaccadeMetrics(Module):
    def update(self, data: GazeData) -> GazeData:
        saccade_metrics = defaultdict(float)

        if len(data.saccades) > 0:
            saccade_metrics["SC"] = len(data.saccades)

            durations = np.array([saccade["duration"] for saccade in data.saccades])
            # saccade_metrics["duration_total"] = np.sum(durations)
            saccade_metrics["MSD"] = np.mean(durations)
            saccade_metrics["SSD"] = np.std(durations)

            amplitudes = np.array([saccade["amplitude"] for saccade in data.saccades])
            # saccade_metrics["amplitude_total"] = np.sum(amplitudes)
            saccade_metrics["MSA"] = np.mean(amplitudes)
            saccade_metrics["SSA"] = np.std(amplitudes)

            velocities = np.array([saccade["velocity"] for saccade in data.saccades])
            saccade_metrics["MSV"] = np.mean(velocities)
            saccade_metrics["SSV"] = np.std(velocities)

            peak_velocities = np.array(
                [saccade["peak_velocity"] for saccade in data.saccades]
            )
            saccade_metrics["MPSV"] = np.mean(peak_velocities)
            saccade_metrics["SPSV"] = np.std(peak_velocities)
            # saccade_metrics["peak_velocity"] = np.max(peak_velocities)

        data.saccade_metrics = saccade_metrics

        return data


class SmoothPursuitMetrics(Module):
    def update(self, data: GazeData) -> GazeData:
        smooth_pursuit_metrics = {
            "count": 0,
            "duration_total": 0,
            "duration_mean": 0,
            "duration_std": 0,
            "amplitude_total": 0,
            "amplitude_mean": 0,
            "amplitude_std": 0,
            "velocity_mean": 0,
            "velocity_std": 0,
        }

        if hasattr(data, "smooth_pursuits") and len(data.smooth_pursuits) > 0:
            smooth_pursuit_metrics["count"] = len(data.smooth_pursuits)

            durations = np.array(
                [pursuit["duration"] for pursuit in data.smooth_pursuits]
            )
            smooth_pursuit_metrics["duration_total"] = np.sum(durations)
            smooth_pursuit_metrics["duration_mean"] = np.mean(durations)
            smooth_pursuit_metrics["duration_std"] = np.std(durations)

            amplitudes = np.array(
                [pursuit["amplitude"] for pursuit in data.smooth_pursuits]
            )
            smooth_pursuit_metrics["amplitude_total"] = np.sum(amplitudes)
            smooth_pursuit_metrics["amplitude_mean"] = np.mean(amplitudes)
            smooth_pursuit_metrics["amplitude_std"] = np.std(amplitudes)

            velocities = np.array(
                [pursuit["velocity"] for pursuit in data.smooth_pursuits]
            )
            smooth_pursuit_metrics["velocity_mean"] = np.mean(velocities)
            smooth_pursuit_metrics["velocity_std"] = np.std(velocities)

        data.smooth_pursuit_metrics = smooth_pursuit_metrics

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
        # if len(all_chunk_lengths) != 0:
        #     print("detected blinks: ", count)
        #     print("average blink length: ", sum(all_chunk_lengths) / len(all_chunk_lengths))
        blink_metrics = {
            "count": count,
            "BR": count / (data.get_total_duration() - reduced_total_duration),
        }
        # print("count of blinks: ", count)
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
        # print("count of blinks: ", count)
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
