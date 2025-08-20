import shutil
import pandas as pd
import numpy as np
import warnings
import os

TIME_FORMAT = "%H:%M:%S:%f"
# to ignore warnings
warnings.filterwarnings("ignore")


PARTICIPANTS = [1]
DURATION = 21


def group_Detection(detection, threshold=0.5):
    if detection < threshold:
        return 0
    return 1


USED_COLUMNS = [
    "LeftPupilDiameter",
    "RightPupilDiameter",
    "EyePosition_x",
    "EyePosition_y",
    "EyePosition_z",
    "EyeDirection_x",
    "EyeDirection_y",
    "EyeDirection_z",
    "GazeTarget",
    "IsFixation",
    "IsSaccade",
    "IsBlink",
]


def clean_dataframe(
    df: pd.DataFrame, use_center=True, use_diameter=False, use_behavior=False
):
    # first find the right start and end timestamp
    if not use_center:
        df.drop(
            ["EyePosition_x", "EyePosition_y", "EyePosition_z"], axis=1, inplace=True
        )
    if not use_diameter:
        df.drop(["LeftPupilDiameter", "RightPupilDiameter"], axis=1, inplace=True)
    if not use_behavior:
        df.drop(["IsFixation", "IsSaccade", "IsBlink"], axis=1, inplace=True)
    return df


def raw_eye_tracking_to_time_series(
    input_dataframe,
    start_id=0,
    window_size=420,
    step_size=60,
    ts_label=None,
    use_center=True,
    use_diameter=True,
    use_gaze_behavior=True,
):
    # Load the input CSV file
    df = input_dataframe.copy()
    feature_df = clean_dataframe(
        df,
        use_center=use_center,
        use_diameter=use_diameter,
        use_behavior=use_gaze_behavior,
    )

    # fill the nan with 0
    feature_df.fillna(0, inplace=True)
    num_samples = start_id
    new_df, label_df, num_samples, labels = extract_from_dfs(
        source_dfs=[feature_df],
        window_size=window_size,
        step_size=step_size,
        num_samples=num_samples,
        ts_label=ts_label,
    )
    return (new_df, label_df, num_samples, labels)


def cartesian_to_spherical(x, y, z):
    # Convert cartesian coordinates to spherical coordinates
    horizontal = np.arctan2(x, z)
    vertical = np.arctan2(y, np.sqrt(x**2 + z**2))
    return [horizontal, vertical]


def extract_from_dfs(
    source_dfs,
    window_size,
    step_size,
    num_samples,
    ts_label=None,
):
    new_df = pd.DataFrame()
    labels = []
    target_feature_indices = []
    start = num_samples
    if ts_label is not None:
        for source_df in source_dfs:
            eye_horizontal, eye_vertical = cartesian_to_spherical(
                source_df["EyeDirection_x"].values,
                source_df["EyeDirection_y"].values,
                source_df["EyeDirection_z"].values,
            )
            source_df["EyeDirection_h"] = eye_horizontal
            source_df["EyeDirection_v"] = eye_vertical
            # remove the original columns
            source_df.drop(
                ["EyeDirection_x", "EyeDirection_y", "EyeDirection_z"],
                axis=1,
                inplace=True,
            )

            index = 0
            feature_df = source_df
            while index < len(source_df) - window_size + 1:
                # skip not full window and window that span two phases
                if index + window_size > len(source_df):
                    break
                ind = np.arange(index, index + window_size)
                labels.append(ts_label)
                new_df = pd.concat([new_df, feature_df.iloc[ind]], ignore_index=True)
                target_feature_indices += [num_samples] * window_size
                num_samples += 1
                index += step_size

        label_num = 2  # in case some chunk have only one label

        label_df = pd.get_dummies(labels + list(range(label_num))).astype("float32")

        # drop the added ones
        label_df = label_df.iloc[: len(labels)]

        new_df["ts_index"] = target_feature_indices
        new_df.set_index("ts_index", inplace=True)
        label_df["ts_index"] = list(range(start, num_samples))
        label_df.set_index("ts_index", inplace=True)
        return new_df, label_df, num_samples, labels
    raise NotImplementedError("In-place label is not implemented yet")


def convert_raw_data(
    read_root_path,
    save_root_path,
    window_size=420,
    step_size=60,
    applied_incidents=["bleeding", "vomiting"],
    use_center=True,
    use_diameter=True,
    use_gaze_behavior=True,
):
    dataframes_by_owner = {}
    labels_by_user = {}
    num_samples = 0

    for root, dirs, files in os.walk(read_root_path):
        for file in files:
            if "csv" not in file or "practice" in file or "ambulance" in file:
                continue
            participant = int(file.split("_")[0])
            if participant not in PARTICIPANTS:
                continue

            trial_name = file.split("_")[1]
            if trial_name not in applied_incidents:
                continue
            detection_value = eval(file.split("_")[2][:-4])
            ts_label = group_Detection(detection_value)

            dataframe = pd.read_csv(os.path.join(root, file), usecols=USED_COLUMNS)

            x, y, num_samples, label_count = raw_eye_tracking_to_time_series(
                dataframe,
                start_id=num_samples,
                window_size=window_size,
                step_size=step_size,
                ts_label=ts_label,
                use_center=use_center,
                use_diameter=use_diameter,
                use_gaze_behavior=use_gaze_behavior,
            )
            if participant not in labels_by_user:
                labels_by_user[participant] = np.zeros(2)
            labels_by_user[participant][ts_label] += len(label_count)

            if participant not in dataframes_by_owner:
                dataframes_by_owner[participant] = [x, y]
            else:
                dataframes_by_owner[participant][0] = pd.concat(
                    [dataframes_by_owner[participant][0], x]
                )
                dataframes_by_owner[participant][1] = pd.concat(
                    [dataframes_by_owner[participant][1], y]
                )

    if os.path.exists(save_root_path):
        shutil.rmtree(save_root_path)

    for owner in dataframes_by_owner:
        x, y = dataframes_by_owner[owner]

        if not os.path.exists(os.path.join(save_root_path, f"{owner}")):
            os.makedirs(os.path.join(save_root_path, f"{owner}"))
        x.to_csv(os.path.join(save_root_path, f"{owner}/feature_df.csv"))
        y.to_csv(os.path.join(save_root_path, f"{owner}/label_df.csv"))


if __name__ == "__main__":
    convert_raw_data(
        f"my_data/processed_gaze_21s/raw",
        f"my_data/processed_ts_21s/center_event",
        use_center=True,
        use_diameter=False,
        use_gaze_behavior=True,
        window_size=420,
        step_size=60,
    )
    convert_raw_data(
        f"my_data/processed_gaze_21s/raw",
        f"my_data/processed_ts_21s/center",
        use_center=True,
        use_diameter=False,
        use_gaze_behavior=False,
        window_size=420,
        step_size=60,
    )
