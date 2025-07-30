import os
import json
import glob
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData, Dataset
import math


def cartesian_to_spherical(x, y, z):
    # Convert cartesian coordinates to spherical coordinates
    horizontal = np.arctan2(x, z)
    vertical = np.arctan2(y, np.sqrt(x**2 + z**2))
    return [horizontal, vertical]


class FinalGazeDataset(Dataset):
    def __init__(
        self,
        root,
        user_ids,
        window_size=420,
        window_step=None,
        transform=None,
        spatial_threshold=0.065,
        temporal_weight_scaling=2.0,
        directed_temporal_edges=True,  # ablation +BDTE: set to False
        reverse_label=True,
        used_features="2d_centroid, duration, target",
    ):
        """
        Simple gaze dataset that only loads fixations and creates a graph from them.

        Args:
            root: Root directory where data is stored
            user_ids: List of user IDs to include
            window_size: Size of each window in samples
            window_step: Step size between windows (defaults to window_size)
            transform: PyTorch Geometric transforms
            spatial_threshold: Maximum distance for spatial edges (similarity threshold)
            temporal_weight_scaling: Scaling factor for temporal edge weights
        """
        super().__init__(root, transform)
        self.window_size = window_size
        self.window_step = window_step or window_size
        self.user_ids = user_ids
        self.spatial_threshold = spatial_threshold
        self.temporal_weight_scaling = temporal_weight_scaling
        self.file_pairs = self._get_file_pairs()
        self.graphs = []
        self.directed_temporal_edges = directed_temporal_edges
        self.reverse_label = reverse_label
        self.used_features = list(
            map(
                lambda x: x.strip(),
                (
                    used_features.split(",")
                    if isinstance(used_features, str)
                    else used_features
                ),
            )
        )
        self._process_files()

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    def _get_file_pairs(self):
        """Get files for all events of selected users"""
        pairs = []
        for uid in self.user_ids:
            user_files = glob.glob(os.path.join(self.raw_dir, f"{uid}_*.csv"))
            for csv_path in user_files:
                json_path = csv_path.replace(".csv", ".json")
                if os.path.exists(json_path):
                    pairs.append((csv_path, json_path))
        return pairs

    def _process_files(self):
        """Process all files into windowed graphs"""
        for csv_path, json_path in self.file_pairs:
            # Parse filename components
            filename = os.path.basename(csv_path)
            _, _, value = filename.split("_")[:3]
            label = value[:-4]
            if self.reverse_label:
                label = 1 - label

            # Load fixation and saccade data from JSON
            with open(json_path) as f:
                events = json.load(f)

            # Get fixations and saccades
            fixations = events.get("fixations", [])
            saccades = events.get("saccades", [])
            blinks = events.get("blinks", [])

            # Get total data length from CSV
            df = pd.read_csv(csv_path)
            total_length = len(df)

            # Create windows
            for window_start in range(0, total_length, self.window_step):
                window_end = window_start + self.window_size
                if window_end > total_length:
                    break

                # Create graph for this window
                graph = self._create_window_graph(
                    fixations, saccades, blinks, window_start, window_end, label
                )

                if graph:  # Only add if graph has nodes
                    self.graphs.append(graph)

    def compute_angular_distance(self, vec1, vec2):
        """Compute angular distance between two 3D vectors using cosine similarity"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        dot_product = np.dot(np.array(vec1), np.array(vec2))
        norm_src = np.linalg.norm(vec1)
        norm_dst = np.linalg.norm(vec2)
        # Clamp dot product to [-1, 1] to avoid numerical errors
        cos_angle = max(min(dot_product / (norm_src * norm_dst), 1.0), -1.0)
        amplitude = np.arccos(cos_angle)  # Angular distance in radians
        return amplitude  # Return in radians, can be converted to degrees if needed

    def _create_window_graph(
        self, fixations, saccades, blinks, window_start, window_end, label
    ):
        """Create a graph for a single window of fixations"""
        # Filter fixations that are in this window
        window_fixations = []
        for fix in fixations:
            # Check if fixation is in this window
            if fix["end_index"] > window_start and fix["start_index"] < window_end:
                # Create a copy to avoid modifying the original
                fix_copy = fix.copy()
                # Adjust indices to be relative to window
                fix_copy["start_index"] = max(0, fix["start_index"] - window_start)
                fix_copy["end_index"] = min(
                    window_end - window_start, fix["end_index"] - window_start
                )
                window_fixations.append(fix_copy)

        if not window_fixations:
            return None  # No fixations in this window

        # Filter saccades in this window
        window_saccades = []
        for sac in saccades:
            if sac["end_index"] > window_start and sac["start_index"] < window_end:
                sac_copy = sac.copy()
                sac_copy["start_index"] = max(0, sac["start_index"] - window_start)
                sac_copy["end_index"] = min(
                    window_end - window_start, sac["end_index"] - window_start
                )
                sac_copy["amplitude"] = sac["amplitude"]
                window_saccades.append(sac_copy)

        # Filter blinks in this window
        window_blinks = []
        for blink in blinks:
            if blink["end_index"] > window_start and blink["start_index"] < window_end:
                blink_copy = blink.copy()
                blink_copy["start_index"] = max(0, blink["start_index"] - window_start)
                blink_copy["end_index"] = min(
                    window_end - window_start, blink["end_index"] - window_start
                )
                window_blinks.append(blink_copy)

        # Sort by start time
        window_fixations.sort(key=lambda x: x["start_index"])

        # Extract node features: [centroid_x, centroid_y, duration, target]
        node_features = []
        f_features = []
        for fix in window_fixations:
            # Use only 2D coordinates (x, y)
            _2d_centroid = fix["2d_centroid"]
            _3d_centroid = fix["centroid"]
            spherical_3d_centroid = cartesian_to_spherical(
                _3d_centroid[0], _3d_centroid[1], _3d_centroid[2]
            )
            eye_center = fix["eye_center"]
            duration = fix["duration"]
            target = fix["target"]
            # pupil_diameter = float(fix['mean_pupil']) / 10
            start = fix["start_index"] / self.window_size
            end = fix["end_index"] / self.window_size
            node_features.append(
                _2d_centroid + [duration, target] + spherical_3d_centroid + eye_center
            )
            f_feature = []
            for used_feature in self.used_features:
                if used_feature == "spherical_3d_centroid":
                    f_feature += spherical_3d_centroid
                else:
                    f_feature += (
                        fix[used_feature]
                        if isinstance(fix[used_feature], list)
                        else [fix[used_feature]]
                    )
            f_features.append(f_feature)

        # Convert to tensor
        node_features = torch.tensor(node_features, dtype=torch.float)
        f_features = torch.tensor(f_features, dtype=torch.float)

        # Create edges and edge attributes
        temporal_edges_index = []
        temporal_edge_attr = []
        temporal_edge_weight = []

        similarity_edges_index = []
        similarity_edge_weight = []
        similarity_edge_attr = []

        # 1. Create temporal edges between consecutive fixations
        for i in range(len(window_fixations) - 1):
            src = i
            dst = i + 1

            # Calculate edge weight based on spatial proximity (inverted distance)
            _2d_src_pos = torch.tensor(
                window_fixations[src]["2d_centroid"], dtype=float
            )
            _2d_dst_pos = torch.tensor(
                window_fixations[dst]["2d_centroid"], dtype=float
            )
            _2d_distance = torch.norm(_2d_src_pos - _2d_dst_pos).item()
            # Higher weight for closer points, max weight is 1
            weight = math.exp(-_2d_distance / self.temporal_weight_scaling)

            # Calculate edge attributes: [offset_x, offset_y, duration, velocity]
            # _2d_offset = torch.tensor(_2d_dst_pos) - torch.tensor(_2d_src_pos)

            # Find if there's a saccade between these fixations
            saccade = None
            for sac in window_saccades:
                # A saccade connects these fixations if it starts after the first fixation ends
                # and ends before the second fixation starts
                if (
                    sac["start_index"] == window_fixations[src]["end_index"]
                    and sac["end_index"] == window_fixations[dst]["start_index"]
                ):
                    saccade = sac
                    break

            # Find if there's a blink between these fixations
            blink = None
            if not saccade:
                for b in window_blinks:
                    if (
                        b["start_index"] == window_fixations[src]["end_index"]
                        and b["end_index"] == window_fixations[dst]["start_index"]
                    ):
                        blink = b
                        break

            # Calculate duration and velocity
            if saccade:
                duration = saccade["duration"]
                amplitude = saccade["amplitude"]
            elif blink:
                duration = blink["duration"]
            else:
                # No saccade or blink found, use time between fixations
                duration = (
                    window_fixations[dst]["start_timestamp"]
                    - window_fixations[src]["end_timestamp"]
                )
                duration = max(0.001, duration)

            target = window_fixations[src]["target"]
            edge_attr = [duration, _2d_distance, target]

            temporal_edges_index.append([src, dst])
            temporal_edge_attr.append(edge_attr + [target])
            temporal_edge_weight.append(weight)

            if not self.directed_temporal_edges:
                # Add reverse edge for undirected temporal edges
                temporal_edges_index.append([dst, src])
                temporal_edge_attr.append(edge_attr + [window_fixations[dst]["target"]])
                temporal_edge_weight.append(weight)

        # 2. Create similarity edges between fixations close in space
        for i in range(len(window_fixations)):
            # Skip consecutive fixations (already covered by temporal edges)
            for j in range(i + 2, len(window_fixations)):
                # Check spatial distance
                src_pos = torch.tensor(window_fixations[i]["2d_centroid"], dtype=float)
                dst_pos = torch.tensor(window_fixations[j]["2d_centroid"], dtype=float)
                distance = torch.norm(src_pos - dst_pos).item()

                if distance < self.spatial_threshold:
                    # Add edges in both directions (undirected)
                    similarity_edges_index.append([i, j])
                    similarity_edges_index.append([j, i])

                    # Weight based on proximity (inverted distance)
                    weight = math.exp(-distance / self.temporal_weight_scaling)
                    duration = duration = (
                        window_fixations[j]["start_index"]
                        - window_fixations[i]["end_index"]
                    ) / 60.0

                    similarity_edge_weight.extend([weight, weight])
                    for dst in [i, j]:
                        edge_attr = [
                            duration,  # duration between these fixations
                            amplitude,  # angular distance
                            window_fixations[dst]["target"],
                        ]

                        similarity_edge_attr.append(edge_attr)

        # Convert edge indices and attributes to tensors
        if temporal_edges_index:
            temporal_edges_index = (
                torch.tensor(temporal_edges_index, dtype=torch.long).t().contiguous()
            )
            temporal_edge_attr = torch.tensor(temporal_edge_attr, dtype=torch.float)
            temporal_edge_weight = torch.tensor(temporal_edge_weight, dtype=torch.float)
        else:
            temporal_edges_index = torch.zeros((2, 0), dtype=torch.long)
            temporal_edge_attr = torch.zeros((0, 3), dtype=torch.float)
            temporal_edge_weight = torch.zeros(0, dtype=torch.float)

        if similarity_edges_index:
            similarity_edges_index = (
                torch.tensor(similarity_edges_index, dtype=torch.long).t().contiguous()
            )
            similarity_edge_weight = torch.tensor(
                similarity_edge_weight, dtype=torch.float
            )
            similarity_edge_attr = torch.tensor(similarity_edge_attr, dtype=torch.float)
        else:
            similarity_edges_index = torch.zeros((2, 0), dtype=torch.long)
            similarity_edge_weight = torch.zeros(0, dtype=torch.float)
            similarity_edge_attr = torch.zeros((0, 3), dtype=torch.float)

        # Create final graph with all edges combined
        if temporal_edges_index.size(1) > 0 or similarity_edges_index.size(1) > 0:
            # Create data object
            data = HeteroData()
            data["f"].x = f_features
            data["f", "T", "f"].edge_index = temporal_edges_index
            data["f", "T", "f"].edge_attr = temporal_edge_attr
            data["f", "T", "f"].edge_weight = temporal_edge_weight

            data["f", "S", "f"].edge_index = similarity_edges_index
            data["f", "S", "f"].edge_weight = similarity_edge_weight
            data["f", "S", "f"].edge_attr = similarity_edge_attr
            data.y = torch.tensor([label], dtype=torch.long)

            return data

        return None

    def normalize_dataset(self, precomputed_norm_dict=None, method="z-score"):
        """Normalize node features across the dataset, function not used since it doesn't seem necessary"""
        if not self.graphs:
            return {}
        return_dict = {}
        used_dict = (
            return_dict if precomputed_norm_dict is None else precomputed_norm_dict
        )
        if method == "z-score":
            if not precomputed_norm_dict:
                all_node_features = torch.cat(
                    [graph["f"].x for graph in self.graphs], dim=0
                )
                means = torch.mean(all_node_features, dim=0)
                stds = torch.std(all_node_features, dim=0)
                stds = torch.where(
                    stds > 0, stds, torch.ones_like(stds)
                )  # Avoid division by zero
                return_dict["f"] = {"means": means, "stds": stds}
            for graph in self.graphs:
                graph["f"].x = (graph["f"].x - used_dict["f"]["means"]) / used_dict[
                    "f"
                ]["stds"]
            for edge_type in graph.edge_types:
                edge_features = (
                    torch.cat(
                        [
                            graph[edge_type].edge_attr
                            for graph in self.graphs
                            if edge_type in graph.edge_types
                        ],
                        dim=0,
                    )
                    if edge_type in graph.edge_types
                    else None
                )

                edge_means = torch.mean(edge_features, dim=0)
                edge_stds = torch.std(edge_features, dim=0)
                edge_stds = torch.where(
                    edge_stds > 0, edge_stds, torch.ones_like(edge_stds)
                )
                return_dict[edge_type] = {"means": edge_means, "stds": edge_stds}
                for graph in self.graphs:
                    edge_feature = graph[edge_type].edge_attr
                    if edge_feature is not None and edge_feature.size(0) > 0:
                        graph[edge_type].edge_attr = (
                            edge_feature - used_dict[edge_type]["means"]
                        ) / used_dict[edge_type]["stds"]

        elif method == "min-max":
            if not precomputed_norm_dict:
                all_node_features = torch.cat(
                    [graph["f"].x for graph in self.graphs], dim=0
                )
                min_vals = torch.min(all_node_features, dim=0)[0]
                max_vals = torch.max(all_node_features, dim=0)[0]
                ranges = max_vals - min_vals
                return_dict["f"] = {"min_vals": min_vals, "ranges": ranges}

            for graph in self.graphs:
                graph["f"].x = (graph["f"].x - used_dict["f"]["min_vals"]) / used_dict[
                    "f"
                ]["ranges"]
            for edge_type in graph.edge_types:
                edge_features = (
                    torch.cat(
                        [
                            graph[edge_type].edge_attr
                            for graph in self.graphs
                            if edge_type in graph.edge_types
                        ],
                        dim=0,
                    )
                    if edge_type in graph.edge_types
                    else None
                )

                edge_min_vals = torch.min(edge_features, dim=0)[0]
                edge_max_vals = torch.max(edge_features, dim=0)[0]
                edge_ranges = edge_max_vals - edge_min_vals
                return_dict[edge_type] = {
                    "min_vals": edge_min_vals,
                    "ranges": edge_ranges,
                }
                for graph in self.graphs:
                    edge_feature = graph[edge_type].edge_attr
                    if edge_feature is not None and edge_feature.size(0) > 0:
                        graph[edge_type].edge_attr = (
                            edge_feature - used_dict[edge_type]["min_vals"]
                        ) / used_dict[edge_type]["ranges"]

        return precomputed_norm_dict if precomputed_norm_dict else return_dict
