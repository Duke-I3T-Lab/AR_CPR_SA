import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    GATv2Conv,
    GINEConv,
    TransformerConv,
)


class EdgePooling(nn.Module):
    """
    Implementation of EdgePool from the paper:
    "Edge Contraction Pooling for Graph Neural Networks"
    """

    def __init__(
        self,
        in_channels,
        edge_score_method="linear",
        dropout=0.0,
        use_edge_attr=False,
        update_edge_attr=False,
        edge_attr_aggr="sum",
    ):
        super(EdgePooling, self).__init__()

        self.in_channels = in_channels
        self.edge_score_method = edge_score_method
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        self.update_edge_attr = update_edge_attr
        self.edge_attr_aggr = edge_attr_aggr

        hidden_dim = in_channels * 2 if not use_edge_attr else in_channels * 3

        if edge_score_method == "lstm":
            # LSTM for edge scoring
            self.edge_scorer = nn.LSTM(hidden_dim, 1, batch_first=True)
        elif edge_score_method == "mlp":
            # MLP for edge scoring
            self.edge_scorer = nn.Sequential(
                nn.Linear(hidden_dim, in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, 1),
            )
        else:  # default: linear
            # Simple linear projection for edge scoring
            self.edge_scorer = nn.Linear(hidden_dim, 1)

        # self.init_params()

    def init_params(self):
        """Initialize parameters for edge scorer"""
        for layer in self.edge_scorer.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LSTM):
                for name, param in layer.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(
        self,
        x,
        edge_index,
        edge_weight=None,
        edge_attr=None,
        batch=None,
        edge_type_mask=None,
    ):
        """
        Args:
            x: Node features of shape [num_nodes, in_channels]
            edge_index: Graph connectivity of shape [2, num_edges]
            edge_weight: Edge weights of shape [num_edges]
            batch: Batch assignment vector of shape [num_nodes]
        """
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # Get source and target nodes for each edge
        row, col = edge_index

        # Compute edge scores
        edge_score = self._compute_edge_score(
            x, edge_attr, edge_index, batch, edge_type_mask
        )

        # apply dropout to edge scores if training
        if self.training and self.dropout > 0:
            edge_score = F.dropout(edge_score, p=self.dropout, training=True)

        # Apply softmax to get normalized scores (per node)
        edge_score = self._normalize_edge_score(edge_score, row, x.size(0))

        # Sort edges by score
        _, perm = torch.sort(edge_score, descending=True)

        # Contract edges based on score
        edge_index, edge_weight, edge_attr, batch, cluster, x = self._contract_edges(
            edge_index,
            edge_weight,
            edge_attr,
            edge_score,
            batch,
            x,
            perm,
        )

        return x, edge_index, edge_weight, edge_attr, batch

    def _compute_edge_score(self, x, edge_attr, edge_index, batch=None):
        row, col = edge_index
        # Get node features for each edge
        edge_feat = (
            torch.cat([x[row], edge_attr, x[col]], dim=-1)
            if self.use_edge_attr
            else torch.cat([x[row], x[col]], dim=-1)
        )

        # Apply dropout
        # edge_feat = F.dropout(edge_feat, p=self.dropout, training=self.training)

        if self.edge_score_method == "lstm":
            # Reshape for LSTM: [num_edges, 1, 2*in_channels]
            edge_feat = edge_feat.unsqueeze(1)
            # Apply LSTM and get output
            _, (h, _) = self.edge_scorer(edge_feat)
            edge_score = h.squeeze()
        else:
            # Apply MLP or Linear
            edge_score = self.edge_scorer(edge_feat).view(-1)

        if edge_score.dim() == 0:
            edge_score = edge_score.unsqueeze(0)

        return edge_score

    def _normalize_edge_score(self, score, row, num_nodes, add_constant=0.5):
        """Normalize edge scores using softmax over incident edges"""
        # Create empty output tensor for normalized scores
        norm_score = score.new_zeros(score.size())

        # For each node, apply softmax to its incident edges
        for i in range(num_nodes):
            # Find indices of edges connected to node i
            mask = row == i

            # If node has edges, apply softmax
            if mask.sum() > 0:
                # Get scores for edges incident to this node
                node_scores = score[mask]
                # Apply softmax only to non-zero scores (T-type edges when only_pool_t_edges is True)
                non_zero_mask = node_scores > 0
                if non_zero_mask.any():
                    # Apply softmax only to non-zero scores
                    node_softmax = torch.zeros_like(node_scores)
                    node_softmax[non_zero_mask] = (
                        F.softmax(node_scores[non_zero_mask], dim=0) + add_constant
                    )
                    norm_score[mask] = node_softmax
                else:
                    # If all scores are zero, keep them zero
                    norm_score[mask] = node_scores

        return norm_score

    def _contract_edges(
        self,
        edge_index,
        edge_weight,
        edge_attr,
        edge_score,
        batch,
        x,
        perm,
    ):
        """Contract edges based on edge scores"""
        num_nodes = x.size(0)
        row, col = edge_index

        # Create a cluster mapping from node to cluster ID
        cluster = torch.arange(num_nodes, device=x.device)

        # Keep track of accumulated edge scores for each cluster

        # Create a copy of x to track weighted features during merging
        merged_x = x.clone()

        # Process edges in order of score
        for edge_idx in perm:
            # Get nodes of this edge
            attr = edge_attr[edge_idx] if edge_attr is not None else None
            u, v = row[edge_idx].item(), col[edge_idx].item()
            edge_s = edge_score[edge_idx].item()

            # Get current cluster assignments
            cluster_u, cluster_v = cluster[u].item(), cluster[v].item()

            # Skip if nodes are already in the same cluster
            if cluster_u == cluster_v:
                continue

            # skip if any of the nodes has already been merged into another cluster, so that in cluster there should be more than one element being cluster_u or cluster_v
            if (cluster == cluster_u).sum().item() > 1 or (
                cluster == cluster_v
            ).sum().item() > 1:
                continue

            if self.update_edge_attr:
                # find the reverse edge if it exists
                # Check if a reverse edge exists (v -> u)
                rev_mask = (row == v) & (col == u)
                if rev_mask.any():
                    # Get the index of the reverse edge
                    rev_edge_idx = torch.where(rev_mask)[0][0]
                    # Get its score and attribute if needed
                    rev_edge_s = edge_score[rev_edge_idx].item()
                    rev_attr = edge_attr[rev_edge_idx]
                else:
                    rev_attr = torch.zeros(self.in_channels, device=x.device)
                    rev_edge_s = 0.0

            # Update features with weighted average based on edge scores
            # This ensures nodes connected by high-score edges have greater influence
            if not self.update_edge_attr:
                merged_x[cluster_u] = (
                    merged_x[cluster_u] + merged_x[cluster_v]
                ) * edge_s
            else:
                merged_x[cluster_u] = (
                    merged_x[cluster_u] + merged_x[cluster_v] + attr
                ) * edge_s + rev_attr * rev_edge_s

            # Contract edge: merge v's cluster into u's cluster
            # Create a mask of nodes in cluster v
            mask = cluster == cluster_v
            # Update those nodes to be in cluster u
            cluster = torch.where(
                mask, torch.tensor(cluster_u, device=x.device), cluster
            )

        # Get unique clusters
        unique_clusters, inverse = torch.unique(cluster, return_inverse=True)

        # Remap cluster IDs to be contiguous
        cluster = inverse

        # Compute new node features by averaging features in each cluster
        new_x = torch.zeros((unique_clusters.size(0), x.size(1)), device=x.device)
        new_x.index_add_(0, cluster, merged_x)

        # Update batch assignment
        # Compute new batch assignment by mapping from old to new clusters
        # Each new cluster should have the same batch assignment as its constituent nodes
        # Since nodes in the same cluster must be in the same batch, we can simply take
        # the batch assignment of any node in each cluster
        new_batch = torch.zeros_like(unique_clusters)
        for i, c in enumerate(unique_clusters):
            nodes_in_cluster = (cluster == i).nonzero().view(-1)
            if nodes_in_cluster.size(0) > 0:
                # Take batch assignment from any node in this cluster
                new_batch[i] = batch[nodes_in_cluster[0]]

        # Create new edges between clusters
        new_edge_weights = {}
        new_edge_attrs = {}
        new_edge_types = {}

        # Map old edge indices to new cluster indices
        for i in range(edge_index.shape[1]):
            src, dst = row[i].item(), col[i].item()
            new_src = cluster[src].item()
            new_dst = cluster[dst].item()
            attr = edge_attr[i] if edge_attr is not None else None
            weight = edge_weight[i].item() if edge_weight is not None else 1.0

            # Skip self-loops
            if new_src == new_dst:
                continue

            if (new_src, new_dst) not in new_edge_weights:
                new_edge_weights[(new_src, new_dst)] = (weight, 1)
            else:
                current_weight, weight_count = new_edge_weights[(new_src, new_dst)]
                if self.edge_attr_aggr == "sum":
                    new_edge_weights[(new_src, new_dst)] = (
                        current_weight + weight,
                        weight_count + 1,
                    )
                elif self.edge_attr_aggr == "mean":  # mean
                    new_edge_weights[(new_src, new_dst)] = (
                        (current_weight * weight_count + weight) / (weight_count + 1),
                        weight_count + 1,
                    )
                else:
                    # not implemented error
                    raise NotImplementedError(
                        f"Edge attribute aggregation method '{self.edge_attr_aggr}' is not implemented."
                    )

            if self.update_edge_attr:
                if (new_src, new_dst) not in new_edge_attrs:
                    new_edge_attrs[(new_src, new_dst)] = (
                        attr if attr is not None else None,
                        1,  # Counter for averaging
                    )
                else:
                    current_attr, count = new_edge_attrs[(new_src, new_dst)]
                    if attr is not None:
                        if self.edge_attr_aggr == "sum":
                            new_edge_attrs[(new_src, new_dst)] = (
                                current_attr + attr,
                                count + 1,
                            )
                        elif self.edge_attr_aggr == "mean":  # mean
                            new_edge_attrs[(new_src, new_dst)] = (
                                (current_attr * count + attr) / (count + 1),
                                count + 1,
                            )
                        else:
                            # not implemented error
                            raise NotImplementedError(
                                f"Edge attribute aggregation method '{self.edge_attr_aggr}' is not implemented."
                            )

        new_edge_index = []
        new_edge_weight = []
        new_edge_attr = []
        new_edge_type_mask = []

        for (src, dst), weight in new_edge_weights.items():
            new_edge_index.append((src, dst))
            new_edge_weight.append(weight)
            if self.update_edge_attr:
                new_edge_attr.append(
                    new_edge_attrs[(src, dst)][0]
                    if new_edge_attrs is not None
                    and new_edge_attrs[(src, dst)] is not None
                    else torch.zeros_like(x[0])
                )
            new_edge_type_mask.append(new_edge_types[(src, dst)])

        new_edge_index = (
            torch.tensor(new_edge_index, dtype=torch.long, device=x.device)
            .t()
            .contiguous()
        )

        new_edge_weight = torch.tensor(
            new_edge_weight, dtype=torch.float, device=x.device
        )
        new_edge_attr = torch.stack(new_edge_attr, dim=0) if new_edge_attr else None

        return (
            new_edge_index,
            new_edge_weight,
            new_edge_attr,
            new_batch,
            cluster,
            new_x,
        )


class EdgePoolBlock(nn.Module):
    """Block consisting of a graph convolution followed by edge pooling"""

    def __init__(
        self,
        out_channels,
        edge_score_method="lstm",
        dropout=0.0,
        use_edge_attr=False,
        update_edge_attr=False,
        lstm_batch=True,  # Whether to use LSTM batch processing for edge scoring
        edge_attr_aggr="sum",
    ):
        super(EdgePoolBlock, self).__init__()
        self.pool = EdgePooling(
            out_channels,
            edge_score_method,
            dropout=dropout,
            lstm_batch=lstm_batch,
            use_edge_attr=use_edge_attr,
            update_edge_attr=update_edge_attr or use_edge_attr,
            edge_attr_aggr=edge_attr_aggr,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_weight=None,
        edge_attr=None,
        batch=None,
    ):
        x, edge_index, edge_weight, edge_attr, batch = self.pool(
            x, edge_index, edge_weight, edge_attr, batch
        )
        return x, edge_index, edge_weight, edge_attr, batch


class FixGraphPoolModel(nn.Module):
    """
    Graph Classification model using EdgePool as described in:
    "Edge Contraction Pooling for Graph Neural Networks"
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        edge_score_method="linear",
        dropout=0.5,
        gnn="GCNConv",
        use_edge_attr_in_conv=False,
        use_attr_in_edge_score=False,
        edge_dim=3,
        num_heads=2,
        lstm_batch=True,
        cat_layer_features=True,
        edge_attr_aggr="sum",
    ):
        super(ENPoolModel, self).__init__()
        gnn_str_to_class = {
            "GCNConv": GCNConv,
            "GINEConv": GINEConv,
            "GATv2Conv": GATv2Conv,
            "TransformerConv": TransformerConv,
        }

        gnn_str_to_kwargs_start = {
            "GINEConv": {
                "nn": nn.Sequential(
                    nn.Linear(in_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
            },
            "GATv2Conv": {
                "heads": num_heads,
                "in_channels": in_channels,
                "out_channels": hidden_channels,
            },
            "TransformerConv": {
                "heads": num_heads,
                "in_channels": in_channels,
                "out_channels": hidden_channels,
            },
            "GCNConv": {"in_channels": in_channels, "out_channels": hidden_channels},
        }

        gnn_str_to_kwargs = {
            "GINEConv": {
                "nn": nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
            },
            "GATv2Conv": {
                "heads": num_heads,
                "in_channels": hidden_channels,
                "out_channels": hidden_channels,
            },
            "TransformerConv": {
                "heads": num_heads,
                "in_channels": hidden_channels,
                "out_channels": hidden_channels,
            },
            "GCNConv": {
                "in_channels": hidden_channels,
                "out_channels": hidden_channels,
            },
        }

        # Add edge dimension to applicable convolutions if needed
        self.use_edge_attr_in_conv = use_edge_attr_in_conv
        self.use_attr_in_edge_score = use_attr_in_edge_score
        if use_edge_attr_in_conv or use_attr_in_edge_score:
            for gnn_name in ["GINEConv", "GATv2Conv", "TransformerConv"]:
                gnn_str_to_kwargs_start[gnn_name]["edge_dim"] = hidden_channels
                gnn_str_to_kwargs[gnn_name]["edge_dim"] = hidden_channels
            self.edge_attr_upprojection = nn.Linear(edge_dim, hidden_channels)
        self.num_layers = num_layers

        self.gnn = gnn_str_to_class.get(gnn, GCNConv)
        # Initial convolution

        # Create multiple layers of pooling blocks
        self.blocks = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            conv = self.gnn(
                **gnn_str_to_kwargs[gnn] if i != 0 else gnn_str_to_kwargs_start[gnn]
            )

            self.convs.append(conv)

            self.blocks.append(
                EdgePoolBlock(
                    hidden_channels,
                    edge_score_method,
                    dropout=dropout,
                    lstm_batch=lstm_batch,
                    use_edge_attr=use_attr_in_edge_score,
                    update_edge_attr=self.use_edge_attr_in_conv,  # Ensure edge attributes are updated if needed
                    edge_attr_aggr=edge_attr_aggr,
                )
            )
            self.bns.append(
                nn.BatchNorm1d(
                    hidden_channels,
                )
            )

        # Final classification layers
        self.cat_layer_features = cat_layer_features
        self.lin1 = (
            nn.Linear(hidden_channels * num_layers, hidden_channels)
            if cat_layer_features
            else nn.Linear(hidden_channels, hidden_channels)
        )
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        batch = data["f"].batch

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = data["f"].x  # Node features
        edge_index_S = data["f", "S", "f"].edge_index
        edge_index_T = data["f", "T", "f"].edge_index
        edge_index = torch.cat([edge_index_S, edge_index_T], dim=-1)
        edge_weight_S, edge_weight_T = (
            data["f", "S", "f"].edge_weight,
            data["f", "T", "f"].edge_weight,
        )
        edge_attr_S, edge_attr_T = (
            data["f", "S", "f"].edge_attr,
            data["f", "T", "f"].edge_attr,
        )
        if self.use_edge_attr_in_conv or self.use_attr_in_edge_score:
            edge_attr_S = self.edge_attr_upprojection(edge_attr_S)
            edge_attr_T = self.edge_attr_upprojection(edge_attr_T)

        edge_weight = torch.cat([edge_weight_S, edge_weight_T])
        edge_attr = torch.cat([edge_attr_S, edge_attr_T], dim=0)

        # Create masks to identify the type of each edge

        x = (
            self.conv1(x, edge_index, edge_weight)
            if not self.use_edge_attr_in_conv
            else self.conv1(x, edge_index, edge_attr=edge_attr)
        )

        x = F.relu(x)
        x = self.bn1(x)

        # Store all graph representations for jump connections
        if self.cat_layer_features:
            xs = []

        # Apply GNN layers with edge pooling
        for i in range(self.num_layers):

            conv = self.convs[i]
            bn = self.bns[i]
            x = (
                conv(x, edge_index, edge_weight)
                if not self.use_edge_attr_in_conv
                else conv(x, edge_index, edge_attr=edge_attr)
            )
            x = F.relu(x)
            x = bn(x)
            if edge_index.shape[0] == 0:  # if no edges, no need to proceed
                xs.append(global_mean_pool(x, batch))
                continue
            (
                x,
                edge_index,
                edge_weight,
                edge_attr,
                batch,
            ) = self.blocks[i](
                x,
                edge_index,
                edge_weight,
                edge_attr,
                batch,
            )
            if self.cat_layer_features:
                xs.append(global_mean_pool(x, batch))

        # Combine features from all levels (jump knowledge)
        if self.cat_layer_features:
            x = torch.cat(xs, dim=1)

        # MLP for classification
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)
