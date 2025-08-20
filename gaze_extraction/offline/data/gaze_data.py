from datetime import datetime, timezone
import os
import numpy as np
import pandas as pd
from typing import List
from itertools import chain
from scipy.stats import mode
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import copy


class GazeData:
    def __init__(self, file_path: str) -> None:
        self.file_path: str = file_path
        self.start_timestamp: float
        self.gaze_direction: np.ndarray

        self.duration = 0
        self.sliced = False

    def load_data(self, file_path: str) -> None:
        raise NotImplementedError("Subclasses of GazeData should implement load_data.")

    def __len__(self):
        return len(self.indices)


class CPRARGazeData(GazeData):
    def __init__(self, file_path: str) -> None:

        super().__init__(file_path=file_path)

        self.confirm_index_dict = {
            "ambulance": 3,
            "bleeding": 11,
            "vomiting": 12,
            "practice": 0,
        }
        self.load_data(file_path)

    def load_data(self, file_path: str, is_demo=False) -> None:
        data = pd.read_csv(
            file_path,
            usecols=[
                "RealTime",
                "LeftPupilDiameter",
                "RightPupilDiameter",
                "MarkerEyePosition",
                "MarkerEyeDirection",
                "LeftEyeOpenness",
                "RightEyeOpenness",
                "Situation",
                "SituationRecognized",
                "StartCompression",
                "GazeTarget",
            ],
            keep_default_na=False,
            na_values=[""],
        )
        # rename markereyeposition and markereyedirection to eye position and eye direction
        data.rename(
            columns={
                "MarkerEyePosition": "EyePosition",
                "MarkerEyeDirection": "EyeDirection",
            },
            inplace=True,
        )
        # Replace 'nan' values in GazeTarget column with 'Real'
        data["GazeTarget"] = data["GazeTarget"].replace(np.nan, "Real")

        trial_type = (
            file_path.split(os.sep)[-1].split("_")[0]
            if "practice" not in file_path
            else "practice"
        )
        self.confirm_index = self.confirm_index_dict[trial_type]

        first_valid_index = (
            data.loc[data["StartCompression"] == 1].index[0]
            if trial_type != "practice"
            else 0
        )
        # drop CameraTime column
        data = data.loc[first_valid_index:]
        data = data.dropna(axis=0, how="any")
        data.reset_index(drop=True, inplace=True)
        self.first_index = 0

        data = self.split_columns_and_save(data, "EyeDirection")
        data = self.split_columns_and_save(data, "EyePosition")

        data = self.preprocess_target_data(data, binary=True)

        data = self.interpolate_missing_gaze_data(data)
        # eye position need to be timed by 0.12 to get to "meter"

        if trial_type != "practice":
            self.recognition_index = (
                data.loc[data["SituationRecognized"] == 1].index[0]
                if 1 in data["SituationRecognized"].values
                else len(data) - 2
            )

        else:
            self.recognition_index = -1

        # rescale from marker scale to meter
        data["EyePosition_x"] = data["EyePosition_x"] * 0.12
        data["EyePosition_y"] = data["EyePosition_y"] * 0.12
        data["EyePosition_z"] = data["EyePosition_z"] * 0.12

        self.df = data

        self.situation = data["Situation"].to_numpy()

        # save blinks in blink_indices.
        # blinks = either LeftEyeOpenness or RightEyeOpenness is 0
        self.blink_indices = data.loc[
            (data["LeftEyeOpenness"] == 0) | (data["RightEyeOpenness"] == 0)
        ].index.to_numpy()

        data = self.process_pupil_diameter(data)

        self.start_timestamp = (data["RealTime"].to_numpy() / 1000).round(4)
        self.label = data["GazeTarget"].to_numpy()

        # Extract gaze direction vectors
        gaze_vectors = data[
            ["EyeDirection_x", "EyeDirection_y", "EyeDirection_z"]
        ].to_numpy()

        # Calculate vector magnitudes (norms)
        magnitudes = np.sqrt(np.sum(gaze_vectors**2, axis=1))

        # Avoid division by zero
        magnitudes = np.where(magnitudes == 0, 1, magnitudes)

        # Normalize each vector by dividing by its magnitude
        self.gaze_direction = gaze_vectors / magnitudes[:, np.newaxis]

        self.eye_position = data[
            ["EyePosition_x", "EyePosition_y", "EyePosition_z"]
        ].to_numpy()
        self.indices = data.index.to_numpy()

    def process_pupil_diameter(self, df):
        # very loose clamping
        df["LeftPupilDiameter"] = df["LeftPupilDiameter"].clip(0.0015, 0.009)
        df["RightPupilDiameter"] = df["RightPupilDiameter"].clip(0.0015, 0.009)
        self.left_pupil_diameter = df["LeftPupilDiameter"].to_numpy()
        self.right_pupil_diameter = df["RightPupilDiameter"].to_numpy()
        return df

    def normalize_pupil_diameter(
        self,
        normalizer_left: StandardScaler,
        normalizer_right: StandardScaler,
        use_normal_dist=False,
    ) -> None:
        if use_normal_dist:
            self.left_pupil_diameter = normalizer_left.transform(
                self.left_pupil_diameter.reshape(-1, 1)
            ).flatten()
            self.df["LeftPupilDiameter"] = normalizer_left.transform(
                self.df["LeftPupilDiameter"].values.reshape(-1, 1)
            ).flatten()
            # print min and max of the normalized left pupil diameter
            self.right_pupil_diameter = normalizer_right.transform(
                self.right_pupil_diameter.reshape(-1, 1)
            ).flatten()
            self.df["RightPupilDiameter"] = normalizer_right.transform(
                self.df["RightPupilDiameter"].values.reshape(-1, 1)
            ).flatten()
            # print min and max of the normalized right pupil diameter
        else:
            self.left_pupil_diameter = (
                self.left_pupil_diameter - normalizer_left.mean_
            ) / normalizer_left.mean_
            self.df["LeftPupilDiameter"] = (
                self.df["LeftPupilDiameter"] - normalizer_left.mean_
            ) / normalizer_left.mean_
            self.right_pupil_diameter = (
                self.right_pupil_diameter - normalizer_right.mean_
            ) / normalizer_right.mean_
            self.df["RightPupilDiameter"] = (
                self.df["RightPupilDiameter"] - normalizer_right.mean_
            ) / normalizer_right.mean_

    def preprocess_target_data(self, df, binary=True):
        def mode_filter(series):
            result = series.copy()
            for i in range(2, len(series) - 2):
                result[i] = mode(series[i - 2 : i + 3]).mode
            return result

        # Convert 'GazeTarget' column: 'None' becomes 0, everything else becomes 1
        if binary:
            df["GazeTarget"] = df["GazeTarget"].apply(lambda x: 0 if x == "None" else 1)

        return df

    def interpolate_missing_gaze_data(self, df):
        average_columns = [
            "RealTime",
            "LeftPupilDiameter",
            "RightPupilDiameter",
            "EyeDirection_x",
            "EyeDirection_y",
            "EyeDirection_z",
            "LeftEyeOpenness",
            "RightEyeOpenness",
            "EyePosition_x",
            "EyePosition_y",
            "EyePosition_z",
        ]
        fixed_0_columns = ["Situation", "StartCompression"]
        majority_columns = ["GazeTarget"]

        # Identify rows with large gaps in "RealTime"
        large_gaps = df.index[df["RealTime"].diff() > 30].tolist()
        large_gaps = [
            gap - 1
            for gap in large_gaps
            if gap < len(df) - 1
            and (df["RealTime"].iloc[gap] - df["RealTime"].iloc[gap - 1]) < 50
        ]

        # List to store new rows
        new_rows = []

        # Insert interpolated rows between large gaps
        for gap in large_gaps:
            # Compute the new row as the midpoint
            new_row = df.iloc[gap].copy()
            new_row["RealTime"] = (
                df["RealTime"].iloc[gap] + df["RealTime"].iloc[gap + 1]
            ) / 2

            # Interpolate other columns
            for col in df.columns:
                if col in average_columns:
                    new_row[col] = (df[col].iloc[gap] + df[col].iloc[gap + 1]) / 2

                elif col in fixed_0_columns:
                    new_row[col] = 0
                elif col in majority_columns:
                    new_row[col] = mode(df[col].iloc[gap - 2 : gap + 3])[0]

            # Append the new row to the list
            new_rows.append(new_row)

        # Add new rows to the dataframe
        if new_rows:
            new_rows_df = pd.DataFrame(new_rows)
            df = (
                pd.concat([df, new_rows_df])
                .sort_values(by="RealTime")
                .reset_index(drop=True)
            )

        return df

    def split_columns_and_save(self, feature_df, col, split_num=3):
        if split_num == 3:
            feature_df[[f"{col}_x", f"{col}_y", f"{col}_z"]] = (
                feature_df[col].str.strip("()").str.split("|", expand=True)
            )
        elif split_num == 4:
            feature_df[[f"{col}_x", f"{col}_y", f"{col}_z", f"{col}_o"]] = (
                feature_df[col].str.strip("()").str.split("|", expand=True)
            )
        feature_df[f"{col}_x"] = pd.to_numeric(feature_df[f"{col}_x"])
        feature_df[f"{col}_y"] = pd.to_numeric(feature_df[f"{col}_y"])
        feature_df[f"{col}_z"] = pd.to_numeric(feature_df[f"{col}_z"])
        if split_num == 4:
            feature_df[f"{col}_o"] = pd.to_numeric(feature_df[f"{col}_o"])
        feature_df = feature_df.drop(col, axis=1)
        return feature_df

    def slice_data_with_config(self, config="before_activation", duration=30):
        if config == "From_Start":
            # find the index that is "duration" seconds after the start
            first_index = 0
            if duration > 0:
                target_time = self.start_timestamp[0] + duration
                last_index = np.argmin(np.abs(self.start_timestamp - target_time))
            else:
                last_index = self.data.loc[self.data["Situation"] > 10].index[0]

        if config == "Before_Recognition":
            last_index = self.recognition_index
            target_time = self.start_timestamp[last_index] - duration
            if target_time < self.start_timestamp[0]:
                first_index = 0
            else:
                first_index = np.argmin(np.abs(self.start_timestamp - target_time))
                first_index = max(0, first_index - 3)

        self.indices = self.indices[first_index:last_index]
        self.start_timestamp = self.start_timestamp[first_index:last_index]
        self.gaze_direction = self.gaze_direction[first_index:last_index]
        self.label = self.label[first_index:last_index]
        self.left_pupil_diameter = self.left_pupil_diameter[first_index:last_index]
        self.right_pupil_diameter = self.right_pupil_diameter[first_index:last_index]
        self.blink_indices = [
            index
            for index in self.blink_indices
            if index >= first_index and index < last_index
        ]
        self.eye_position = self.eye_position[first_index:last_index]
        self.first_index = first_index
        self.df = self.df.iloc[first_index:last_index]
        return True

    def slice_data_to_windows(self, window_size=300, step_size=60):
        window_starts = list(range(0, len(self.indices) - window_size, step_size))
        if not window_starts:  # If data is smaller than window_size
            return [self]

        windows = []
        for start_idx in window_starts:
            end_idx = start_idx + window_size

            # Create a copy of the current object
            window = copy.deepcopy(self)

            # Assign sliced data to the new window
            window.indices = self.indices[start_idx:end_idx]
            window.start_timestamp = self.start_timestamp[start_idx:end_idx]
            window.gaze_direction = self.gaze_direction[start_idx:end_idx]
            window.eye_position = self.eye_position[start_idx:end_idx]
            window.label = self.label[start_idx:end_idx]
            window.left_pupil_diameter = self.left_pupil_diameter[start_idx:end_idx]
            window.right_pupil_diameter = self.right_pupil_diameter[start_idx:end_idx]
            window.df = self.df.iloc[start_idx:end_idx].reset_index(drop=True)

            window.blink_indices = [
                idx
                for idx in self.blink_indices
                if window.first_index + start_idx <= idx < window.first_index + end_idx
            ]

            window.first_index = self.first_index + start_idx

            window.sliced = True
            window.duration = window.start_timestamp[-1] - window.start_timestamp[0]
            windows.append(window)

        return windows

    def get_total_duration(self) -> float:
        return self.start_timestamp[-1] - self.start_timestamp[0]


if __name__ == "__main__":
    pass
