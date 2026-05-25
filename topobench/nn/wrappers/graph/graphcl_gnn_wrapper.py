"""Wrapper for Graph Contrastive Learning (GraphCL) pre-training with GNN models."""

import torch
import torch.nn as nn
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import dropout_edge, subgraph

from topobench.nn.wrappers.base import AbstractWrapper


class GraphCLGNNWrapper(AbstractWrapper):
    r"""Wrapper for Graph Contrastive Learning (GraphCL) pre-training with GNN models.

    This wrapper implements the augmentation and encoding logic for GraphCL
    self-supervised pre-training on graphs. GraphCL maximizes agreement
    between two augmented views of the same graph.

    The official GraphCL repo has several experiment directories with
    inconsistent augmentation implementations.  Three parameters let users
    select the variant they want:

    * ``mask_attr_strategy`` -- how masked node features are replaced.
    * ``edge_perturbation_mode`` -- whether edge perturbation only drops
      edges or both drops and adds.
    * ``subgraph_ratio_meaning`` -- whether ``aug_ratio`` for the subgraph
      augmentation denotes the fraction of nodes to *keep* or to *drop*.

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model.
    aug1 : str, optional
        First augmentation type (default: "drop_edge").
        Options: "none", "drop_node", "drop_edge", "mask_attr", "subgraph"
    aug2 : str, optional
        Second augmentation type (default: "mask_attr").
        Options: "none", "drop_node", "drop_edge", "mask_attr", "subgraph"
    aug_ratio1 : float, optional
        Ratio for first augmentation (default: 0.2).
    aug_ratio2 : float, optional
        Ratio for second augmentation (default: 0.2).
    readout_type : str, optional
        Type of graph-level pooling (default: "mean").
    mask_attr_strategy : str, optional
        How masked node features are replaced (default: "gaussian").
        * ``"gaussian"``  -- Gaussian noise N(0.5, 0.5)  (``unsupervised_TU``)
        * ``"zeros"``     -- zero vector  (``semisupervised_MNIST_CIFAR10``)
        * ``"mean"``      -- per-batch mean feature vector  (``semisupervised_TU``)
    edge_perturbation_mode : str, optional
        Edge perturbation behaviour (default: "drop_only").
        * ``"drop_only"``   -- only drop edges  (``unsupervised_TU``)
        * ``"drop_and_add"`` -- drop some edges and add random ones  (all others)
    subgraph_ratio_meaning : str, optional
        Semantics of ``aug_ratio`` for the subgraph augmentation
        (default: "keep").
        * ``"keep"`` -- ratio = fraction of nodes to keep
          (``unsupervised_TU``, ``semisupervised_TU``)
        * ``"drop"`` -- ratio = fraction of nodes to drop
          (``semisupervised_MNIST_CIFAR10``)
    residual_connections : bool, optional
        If True, add each view's augmented node features to encoder output and
        apply LayerNorm before graph pooling (same contract as other wrappers).
        Passed via ``**kwargs`` to :class:`AbstractWrapper`.
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.
    """

    _VALID_MASK_STRATEGIES = {"gaussian", "zeros", "mean"}
    _VALID_EDGE_MODES = {"drop_only", "drop_and_add"}
    _VALID_SUBGRAPH_RATIO = {"keep", "drop"}

    def __init__(
        self,
        backbone: nn.Module,
        aug1: str = "drop_edge",
        aug2: str = "mask_attr",
        aug_ratio1: float = 0.2,
        aug_ratio2: float = 0.2,
        readout_type: str = "mean",
        mask_attr_strategy: str = "gaussian",
        edge_perturbation_mode: str = "drop_only",
        subgraph_ratio_meaning: str = "keep",
        **kwargs
    ):
        super().__init__(backbone, **kwargs)
        
        self.aug1 = aug1
        self.aug2 = aug2
        self.aug_ratio1 = aug_ratio1
        self.aug_ratio2 = aug_ratio2
        self.readout_type = readout_type

        if mask_attr_strategy not in self._VALID_MASK_STRATEGIES:
            raise ValueError(
                f"Unknown mask_attr_strategy '{mask_attr_strategy}'. "
                f"Choose from {self._VALID_MASK_STRATEGIES}."
            )
        if edge_perturbation_mode not in self._VALID_EDGE_MODES:
            raise ValueError(
                f"Unknown edge_perturbation_mode '{edge_perturbation_mode}'. "
                f"Choose from {self._VALID_EDGE_MODES}."
            )
        if subgraph_ratio_meaning not in self._VALID_SUBGRAPH_RATIO:
            raise ValueError(
                f"Unknown subgraph_ratio_meaning '{subgraph_ratio_meaning}'. "
                f"Choose from {self._VALID_SUBGRAPH_RATIO}."
            )

        self.mask_attr_strategy = mask_attr_strategy
        self.edge_perturbation_mode = edge_perturbation_mode
        self.subgraph_ratio_meaning = subgraph_ratio_meaning
        
        # Get feature dimension from kwargs
        self.feature_dim = kwargs.get("out_channels")
        
        if self.feature_dim is None:
            raise ValueError("Cannot determine feature dimension. Please provide 'out_channels' in kwargs.")

    def _apply_view_residual(self, enc: torch.Tensor, aug_x: torch.Tensor) -> torch.Tensor:
        """Residual + LayerNorm on node embeddings (aligned augmented inputs)."""
        return self.ln_0(enc + aug_x)

    def residual_connection(self, model_out, batch):
        """Residual is applied per augmented view inside :meth:`forward`.

        The base implementation sums ``model_out['x_0']`` with ``batch.x_0``,
        which is wrong here: contrastive views use augmented subgraphs, and
        ``z1``/``z2`` must be built from post-residual node embeddings.
        """
        return model_out

    def augment(self, x, edge_index, batch_indices, aug_type, aug_ratio, device):
        """Apply augmentation to the graph.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (num_nodes, num_features).
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).
        batch_indices : torch.Tensor
            Batch assignment for each node.
        aug_type : str
            Type of augmentation to apply.
        aug_ratio : float
            Ratio of augmentation.
        device : torch.device
            Device to use.
            
        Returns
        -------
        tuple
            (augmented_x, augmented_edge_index)
        """
        if aug_type == "none":
            return x, edge_index, batch_indices
        
        elif aug_type == "drop_node":
            # Randomly drop nodes
            num_nodes = x.size(0)
            keep_mask = torch.rand(num_nodes, device=device) > aug_ratio
            # Ensure at least one node per graph is kept
            keep_mask = self._ensure_connected(keep_mask, batch_indices)
            
            # Create new node features
            aug_x = x[keep_mask]
            
            # Create mapping from old to new node indices
            node_idx_mapping = torch.zeros(num_nodes, dtype=torch.long, device=device) - 1
            node_idx_mapping[keep_mask] = torch.arange(keep_mask.sum(), device=device)
            
            # Filter edges and remap indices
            edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
            aug_edge_index = node_idx_mapping[edge_index[:, edge_mask]]
            
            # Update batch indices
            new_batch_indices = batch_indices[keep_mask]
            
            return aug_x, aug_edge_index, new_batch_indices
        
        elif aug_type == "drop_edge":
            if self.edge_perturbation_mode == "drop_only":
                aug_edge_index, _ = dropout_edge(edge_index, p=aug_ratio, training=True)
            else:
                # Drop some edges and add random ones (semisupervised variants)
                num_edges = edge_index.size(1)
                permute_num = int(num_edges * aug_ratio)
                keep_num = num_edges - permute_num

                keep_idx = torch.randperm(num_edges, device=device)[:keep_num]
                kept_edges = edge_index[:, keep_idx]

                num_nodes = x.size(0)
                new_src = torch.randint(0, num_nodes, (permute_num,), device=device)
                new_dst = torch.randint(0, num_nodes, (permute_num,), device=device)
                added_edges = torch.stack([new_src, new_dst], dim=0)

                aug_edge_index = torch.cat([kept_edges, added_edges], dim=1)
            return x, aug_edge_index, batch_indices
        
        elif aug_type == "mask_attr":
            num_nodes = x.size(0)
            num_mask = max(int(num_nodes * aug_ratio), 1)
            perm = torch.randperm(num_nodes, device=device)
            idx_mask = perm[:num_mask]
            aug_x = x.clone()

            if self.mask_attr_strategy == "gaussian":
                aug_x[idx_mask] = torch.normal(
                    mean=0.5, std=0.5,
                    size=(num_mask, x.size(1)),
                    device=device, dtype=x.dtype,
                )
            elif self.mask_attr_strategy == "zeros":
                aug_x[idx_mask] = 0.0
            else:  # "mean"
                aug_x[idx_mask] = x.mean(dim=0)

            return aug_x, edge_index, batch_indices
        
        elif aug_type == "subgraph":
            # Random-walk-based subgraph sampling (per graph in the batch)
            num_nodes = x.size(0)
            if self.subgraph_ratio_meaning == "keep":
                keep_ratio = aug_ratio
            else:
                keep_ratio = 1.0 - aug_ratio
            keep_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

            unique_batches = torch.unique(batch_indices)
            row, col = edge_index

            for b in unique_batches:
                graph_mask = batch_indices == b
                graph_nodes = graph_mask.nonzero(as_tuple=True)[0]
                n = graph_nodes.size(0)
                num_keep = max(int(n * keep_ratio), 1)

                # Build per-graph adjacency for fast neighbor lookup
                edge_in_graph = graph_mask[row] & graph_mask[col]
                local_row = row[edge_in_graph]
                local_col = col[edge_in_graph]

                # Start a random walk from a random node in this graph
                start = graph_nodes[torch.randint(n, (1,), device=device).item()]
                visited = {start.item()}
                current = start.item()

                for _ in range(num_keep - 1):
                    neighbors = local_col[local_row == current]
                    if neighbors.numel() == 0:
                        # Restart from a random visited node if stuck
                        visited_t = torch.tensor(list(visited), device=device)
                        current = visited_t[torch.randint(len(visited_t), (1,), device=device).item()].item()
                        neighbors = local_col[local_row == current]
                        if neighbors.numel() == 0:
                            break
                    current = neighbors[torch.randint(neighbors.numel(), (1,), device=device).item()].item()
                    visited.add(current)

                visited_idx = torch.tensor(list(visited), dtype=torch.long, device=device)
                keep_mask[visited_idx] = True

            keep_mask = self._ensure_connected(keep_mask, batch_indices)

            aug_edge_index, _, edge_mask = subgraph(
                keep_mask, edge_index, relabel_nodes=True, return_edge_mask=True
            )
            aug_x = x[keep_mask]
            new_batch_indices = batch_indices[keep_mask]

            return aug_x, aug_edge_index, new_batch_indices
        
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")
    
    def _ensure_connected(self, keep_mask, batch_indices):
        """Ensure at least one node per graph is kept.
        
        Parameters
        ----------
        keep_mask : torch.Tensor
            Boolean mask of nodes to keep.
        batch_indices : torch.Tensor
            Batch assignment for each node.
            
        Returns
        -------
        torch.Tensor
            Updated keep mask with at least one node per graph.
        """
        # Get unique batch indices
        unique_batches = torch.unique(batch_indices)
        
        for b in unique_batches:
            batch_mask = batch_indices == b
            if not keep_mask[batch_mask].any():
                # If no node is kept for this graph, keep a random one
                indices = batch_mask.nonzero(as_tuple=True)[0]
                random_idx = indices[torch.randint(len(indices), (1,))]
                keep_mask[random_idx] = True
        
        return keep_mask
    
    def pool_graph(self, x, batch_indices):
        """Pool node embeddings to get graph-level representation.
        
        Parameters
        ----------
        x : torch.Tensor
            Node embeddings of shape (num_nodes, hidden_dim).
        batch_indices : torch.Tensor
            Batch assignment for each node.
            
        Returns
        -------
        torch.Tensor
            Graph-level representation of shape (num_graphs, hidden_dim).
        """
        if self.readout_type == "mean":
            return global_mean_pool(x, batch_indices)
        elif self.readout_type == "max":
            return global_max_pool(x, batch_indices)
        elif self.readout_type == "sum":
            return global_add_pool(x, batch_indices)
        else:
            raise ValueError(f"Unknown readout type: {self.readout_type}")
    
    def forward(self, batch):
        r"""Forward pass for GraphCL encoding with augmentations.

        Creates two augmented views of each graph, encodes them through a
        shared backbone, and pools to graph-level representations.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing:
            - x_0: Node features from the first view (for compatibility)
            - z1: Graph-level embedding for first augmented view
            - z2: Graph-level embedding for second augmented view
            - batch_0: Original batch assignment
            - labels: Original labels (for compatibility)
        """
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0

        device = x_0.device

        aug_x1, aug_edge_index1, aug_batch1 = self.augment(
            x_0, edge_index, batch_indices, self.aug1, self.aug_ratio1, device
        )

        aug_x2, aug_edge_index2, aug_batch2 = self.augment(
            x_0, edge_index, batch_indices, self.aug2, self.aug_ratio2, device
        )

        enc1 = self.backbone(
            aug_x1,
            aug_edge_index1,
            batch=aug_batch1,
            edge_weight=edge_weight if self.aug1 not in ["drop_edge", "drop_node", "subgraph"] else None,
        )

        enc2 = self.backbone(
            aug_x2,
            aug_edge_index2,
            batch=aug_batch2,
            edge_weight=edge_weight if self.aug2 not in ["drop_edge", "drop_node", "subgraph"] else None,
        )

        if self.residual_connections:
            enc1 = self._apply_view_residual(enc1, aug_x1)
            enc2 = self._apply_view_residual(enc2, aug_x2)

        z1 = self.pool_graph(enc1, aug_batch1)
        z2 = self.pool_graph(enc2, aug_batch2)

        model_out = {
            "x_0": enc1,
            "z1": z1,
            "z2": z2,
            "labels": batch.y if hasattr(batch, "y") else None,
            "batch_0": batch_indices,
        }

        return model_out

