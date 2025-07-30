from typing import List
import pandas as pd


class Metric:
    def __init__(
        self, name: str, primary_group: List[str], conditions: List[str]
    ) -> None:
        self.name = name
        self.primary_group = primary_group
        self.conditions = conditions
        self.users_list = []


class CPREyeTrackingMetric(Metric):
    def __init__(
        self,
        name: str,
        primary_group: List[str],
        secondary_group: List[str],
        conditions: List[str],
    ) -> None:
        super().__init__(name, primary_group, conditions)
        self.secondary_group = secondary_group
        self.per_user_num_rows = len(primary_group)
        self.num_cols = 3 + len(conditions)
        self.data_frame = pd.DataFrame(columns=range(self.num_cols))
        self.column_names = ["UserID", "Incident", "Detection"] + conditions

    def feed_data(self, user_id, data, group_list, condition):
        if (
            user_id not in self.users_list
        ):  # new user, append a full per_user_num_rows rows
            self.users_list.append(user_id)
            for i in range(self.per_user_num_rows):
                self.data_frame.loc[len(self.data_frame)] = [user_id] + [-1] * (
                    self.num_cols - 1
                )
        row_index = self.per_user_num_rows * self.users_list.index(
            user_id
        ) + self.primary_group.index(group_list[0])
        self.data_frame.iloc[row_index, 1] = group_list[0]
        self.data_frame.iloc[row_index, 2] = group_list[1]
        self.data_frame.iloc[row_index, 3 + self.conditions.index(condition)] = float(
            data
        )

    def dump_data(self, file_name):
        self.data_frame.to_csv(file_name, index=False, header=self.column_names)
