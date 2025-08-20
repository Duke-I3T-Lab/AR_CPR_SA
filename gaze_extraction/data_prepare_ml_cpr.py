import copy
import numpy as np
import pandas as pd
import os
import json
from typing import List
import offline.modules as m
from offline.data import CPRARGazeData, CPREyeTrackingMetric
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

DATA_DIR = "my_data/raw_data"
WINDOW_SIZE = 420
STEP_SIZE = 60
DURATION = 21
SAVE_DIR = f"my_data/processed_ml_{DURATION}s"
PARTICIPANTS = [1]
IDENTIFIERS = {"practice": 3, "bleeding": 0, "vomiting": 1}
METRICS = {
    "fixation_metrics": ["MFD", "SFD", "FR", "PFT"],
    "saccade_metrics": ["MSV", "SSV", "MPSV", "SPSV", "MSA", "SSA", "MSD", "SSD"],
    "blink_metrics": ["BR"],
    "diameter_metrics": ["MD", "VD"],
    "roi_metrics": ["PFR", "PFV", "VMFD", "RMFD", "VFR", "RFR"],
}
NORMALIZE_PUPIL_DIAMETER = True


def form_metric_row(data: CPRARGazeData, label=1.0):
    data_list = [
        getattr(data, metric_type)[metric_name]
        for metric_type, metric_names in METRICS.items()
        for metric_name in metric_names
    ]
    return ",".join([str(x) for x in data_list]) + f",{label}"


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
        m.AggregateFixations(merge_direction_threshold=0.5),
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

    os.makedirs(SAVE_DIR, exist_ok=True)

    LEFT_PUPIL_DIAMETER_NORMALIZER, RIGHT_PUPIL_DIAMETER_NORMALIZER = (
        StandardScaler(),
        StandardScaler(),
    )
    label_file_path = DATA_DIR + "/labeling_details.xlsx"
    label_df = pd.read_excel(label_file_path)

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
        LEFT_PUPIL_DIAMETER_NORMALIZER.fit(
            practice_df["LeftPupilDiameter"].values.reshape(-1, 1)
        )
        RIGHT_PUPIL_DIAMETER_NORMALIZER.fit(
            practice_df["RightPupilDiameter"].values.reshape(-1, 1)
        )
        for root, dirs, files in os.walk(p_data_dir):
            for i, file in enumerate(files):
                if "csv" not in file or "practice" in file:
                    continue
                trial_name = file.split("_")[0]
                if trial_name not in IDENTIFIERS:
                    continue
                print("user: ", participant, "trial: ", trial_name)
                csv_name = f"p{participant}_{trial_name}.csv"
                all_rows = []
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

                label_data.slice_data_with_config(
                    "Before_Recognition", duration=DURATION
                )
                all_window_data = label_data.slice_data_to_windows(
                    window_size=WINDOW_SIZE, step_size=STEP_SIZE
                )
                for window_data in all_window_data:
                    for module in modules:
                        period_data = module.update(window_data)
                    all_rows.append(form_metric_row(window_data, detection_value))

                # Define output path
                output_path = os.path.join(SAVE_DIR, csv_name)

                # Create header
                header_row = (
                    ",".join(
                        [
                            f"{metric_name}"
                            for _, metric_names in METRICS.items()
                            for metric_name in metric_names
                        ]
                    )
                    + ",label"
                )

                # Write data to CSV
                with open(output_path, "w") as f:
                    f.write(header_row + "\n")
                    for row in all_rows:
                        f.write(row + "\n")
