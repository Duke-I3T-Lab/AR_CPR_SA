import copy
import numpy as np
import pandas as pd
import os
import json

import offline.modules as m
from offline.data import CPRARGazeData, CPREyeTrackingMetric

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "my_data/raw_data"
DISCARD_SET = {1, 2, 12, 13, 17, 24}

PARTICIPANTS = list(set(list(range(1, 37))) - DISCARD_SET)
DURATION = 21  # 7, 14, 21


IDENTIFIERS = {"practice": 3, "bleeding": 0, "vomiting": 1}

METRICS = {
    "fixation_metrics": ["MFD", "FR", "PFT"],
    "saccade_metrics": ["MSV", "MPSV", "MSA"],
    "blink_metrics": ["BR"],
    "diameter_metrics": ["MD", "VD"],
    "roi_metrics": ["PFR", "PFV", "VMFD", "RMFD", "VFR", "RFR"],
}

NORMALIZE_PUPIL_DIAMETER = True


def write_data_to_all_metrics(
    data, participant_id, primary, secondary, condition, all_metrics
):
    for event, metric_classes in all_metrics.items():
        for metric_class in metric_classes:
            metric_class.feed_data(
                participant_id,
                getattr(data, event)[metric_class.name],
                [primary, secondary],
                condition,
            )


def save_all_metrics(root_dir, all_metrics):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    for event, metric_classes in all_metrics.items():
        for metric_class in metric_classes:
            metric_class.dump_data(os.path.join(root_dir, f"{metric_class.name}.csv"))


if __name__ == "__main__":

    modules = [
        m.BlinkConverter(),
        m.DurationDistanceVelocity(window_size=3),
        m.MobilityDetection(window_size=5),
        m.SavgolFilter(attr="velocity", window_size=3, order=1),
        m.IVTFixationDetector(
            velocity_threshold=30,
            use_mobility=True,
            mobile_velocity_threshold=100,
        ),
        m.AggregateFixations(merge_direction_threshold=0.5, target_threshold=0.5),
        m.IVTSaccadeDetector(velocity_threshold=30),
        m.AggregateSaccades(),
        m.AggregateBlinks(),
        m.GazeEventSequenceGenerator(),
        m.FixationMetrics(),
        m.SaccadeMetrics(),
        m.ROIMetrics(rois={0: "R", 1: "V"}),
        m.BlinkMetrics(min_count=2),
        m.DiameterMetrics(),
    ]
    save_tool = m.GazeDataExporter(f"my_data/processed_gaze_{DURATION}s/raw")
    label_file_path = DATA_DIR + "/labeling_details.xlsx"
    label_df = pd.read_excel(label_file_path)

    LEFT_PUPIL_DIAMETER_NORMALIZER, RIGHT_PUPIL_DIAMETER_NORMALIZER = (
        StandardScaler(),
        StandardScaler(),
    )

    METRIC_CLASSES = {}
    for event, metric_names in METRICS.items():
        METRIC_CLASSES[event] = [
            CPREyeTrackingMetric(
                name,
                ["bleeding", "vomiting"],
                [0, 1],
                ["Before_Confirmation"],
            )
            for name in metric_names
        ]

    results = {}
    all_speed = []
    for participant in tqdm(PARTICIPANTS):
        p_data_dir = DATA_DIR + f"/p{participant}"
        practice_file_dir = p_data_dir + "/practice.csv"

        if not os.path.exists(practice_file_dir):
            raise FileNotFoundError(f"Practice file not found in {practice_file_dir}")

        practice_df = pd.read_csv(
            practice_file_dir, usecols=["LeftPupilDiameter", "RightPupilDiameter"]
        )
        # Filter pupil diameter values between 0.0015 and 0.009
        practice_df = practice_df[
            (practice_df["LeftPupilDiameter"] >= 0.0015)
            & (practice_df["LeftPupilDiameter"] <= 0.009)
            & (practice_df["RightPupilDiameter"] >= 0.0015)
            & (practice_df["RightPupilDiameter"] <= 0.009)
        ]

        # Check if we have enough samples for normalization
        if len(practice_df) < 10:
            print(
                f"Warning: Only {len(practice_df)} valid pupil diameter samples for participant {participant}!"
            )

        LEFT_PUPIL_DIAMETER_NORMALIZER.fit(
            practice_df["LeftPupilDiameter"].values.reshape(-1, 1)
        )
        RIGHT_PUPIL_DIAMETER_NORMALIZER.fit(
            practice_df["RightPupilDiameter"].values.reshape(-1, 1)
        )
        # print left and right pupil diameter normalizer mean and std
        for root, dirs, files in os.walk(p_data_dir):
            for i, file in enumerate(files):
                if "csv" not in file or "practice" in file:
                    continue
                trial_name = file.split("_")[0]
                if trial_name not in IDENTIFIERS:
                    continue
                detection_value = label_df[
                    (label_df["User"] == participant)
                    & (label_df["Incident"] == trial_name[0].upper())
                ]["Label"].values[0]

                identifier = IDENTIFIERS[trial_name]
                label_data = CPRARGazeData(os.path.join(root, file))

                if NORMALIZE_PUPIL_DIAMETER:
                    label_data.normalize_pupil_diameter(
                        LEFT_PUPIL_DIAMETER_NORMALIZER,
                        RIGHT_PUPIL_DIAMETER_NORMALIZER,
                        use_normal_dist=True,
                    )

                for period, config in enumerate(["Before_Recognition"]):
                    period_data = copy.deepcopy(label_data)
                    period_data.slice_data_with_config(config, duration=DURATION)
                    total_duration = period_data.get_total_duration()
                    for module in modules:
                        period_data = module.update(period_data)

                    save_tool.save_data(
                        period_data, participant, trial_name, detection_value
                    )
                    write_data_to_all_metrics(
                        period_data,
                        participant,
                        trial_name,
                        detection_value,
                        config,
                        METRIC_CLASSES,
                    )
    save_all_metrics(f"my_data/results/period_{DURATION}s", METRIC_CLASSES)
