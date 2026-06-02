"""VGAE pretraining: edge masking, within-graph negative sampling, variational latent z."""

import torch
import torch.nn as nn

from topobench.nn.wrappers.base import AbstractWrapper


class _EdgeSamplingGNNWrapper(AbstractWrapper):
    r"""Mask random edges, run the GNN on the subgraph, sample negatives per graph.

    Internal base for :class:`VAEGNNWrapper`.
    """

    def __init__(
        self,
        backbone: nn.Module,
        edge_sample_ratio: float = 0.5,
        neg_sample_ratio: float = 1.0,
        sampling_method: str = "sparse",
        **kwargs,
    ):
        super().__init__(backbone, **kwargs)

        self.edge_sample_ratio = edge_sample_ratio
        self.neg_sample_ratio = neg_sample_ratio
        self.sampling_method = sampling_method

        if not 0.0 < edge_sample_ratio < 1.0:
            raise ValueError(f"edge_sample_ratio must be in (0, 1), got {edge_sample_ratio}")
        if neg_sample_ratio <= 0:
            raise ValueError(f"neg_sample_ratio must be positive, got {neg_sample_ratio}")
        if sampling_method not in ["sparse", "dense"]:
            raise ValueError(f"sampling_method must be 'sparse' or 'dense', got {sampling_method}")

    def sample_edges(self, edge_index, edge_attr, batch_indices, num_nodes, device,
                     virtual_node_mask=None):
        """Split edges into remaining (for encoding) and pos (for link prediction).

        Virtual-node edges are excluded from the positive sample pool — they are
        artificial edges added by the transform, not real graph edges, so they
        should not appear as link-prediction targets.  They are always kept in
        the remaining (encoding) edge set.

        The same random permutation is used to split edge_attr so that it stays
        aligned with the corresponding edge_index subset.
        """
        num_edges = edge_index.size(1)

        if num_edges == 0:
            empty = torch.empty((2, 0), dtype=edge_index.dtype, device=device)
            return empty, empty, None, None

        # Identify real edges (not touching the virtual node)
        if virtual_node_mask is not None:
            is_vn_edge = virtual_node_mask[edge_index[0]] | virtual_node_mask[edge_index[1]]
            real_edge_indices = torch.where(~is_vn_edge)[0]
            vn_edge_indices = torch.where(is_vn_edge)[0]
        else:
            real_edge_indices = torch.arange(num_edges, device=device)
            vn_edge_indices = torch.empty(0, dtype=torch.long, device=device)

        num_real_edges = real_edge_indices.size(0)
        num_pos_samples = max(1, int(self.edge_sample_ratio * num_real_edges))

        perm = torch.randperm(num_real_edges, device=device)
        pos_local = perm[:num_pos_samples]
        remain_local = perm[num_pos_samples:]

        pos_global = real_edge_indices[pos_local]
        remain_global = torch.cat([real_edge_indices[remain_local], vn_edge_indices])

        remaining_edge_index = edge_index[:, remain_global]
        pos_edge_index = edge_index[:, pos_global]

        remaining_edge_attr = edge_attr[remain_global] if edge_attr is not None else None
        pos_edge_attr = edge_attr[pos_global] if edge_attr is not None else None

        return remaining_edge_index, pos_edge_index, remaining_edge_attr, pos_edge_attr

    def sample_negative_edges(
        self, remaining_edge_index, num_pos_edges, batch_indices, num_nodes, device,
        virtual_node_mask=None,
    ):
        """Optimized negative edge sampling - guaranteed faster than original.

        Virtual nodes are excluded from the candidate node pool so that sampled
        negative edges never involve the artificial global aggregator.
        """
        num_neg_samples = int(num_pos_edges * self.neg_sample_ratio)

        if num_neg_samples == 0:
            return torch.empty((2, 0), dtype=remaining_edge_index.dtype, device=device)

        batch_ids = batch_indices.unique()

        nodes_per_graph = torch.bincount(batch_indices, minlength=batch_ids.max() + 1)[batch_ids]
        total_nodes = nodes_per_graph.sum().item()
        
        # Pre-allocate for all negative edges at once
        neg_edge_list = []
        
        for graph_idx, batch_id in enumerate(batch_ids):

            graph_node_mask = (batch_indices == batch_id)
            num_graph_nodes = nodes_per_graph[graph_idx].item()
            
            if num_graph_nodes < 2:
                continue
            
            # Calculate how many negatives for this graph
            graph_neg_samples = max(
                1, int(num_neg_samples * num_graph_nodes / total_nodes)
            )
            
            graph_node_mask[remaining_edge_index[0]]
            graph_node_mask[remaining_edge_index[1]]
            
            sample_size = min(graph_neg_samples * 2, num_graph_nodes * num_graph_nodes)
            
            # Generate random node pairs within this graph's node range.
            # Exclude the virtual node from the candidate pool so that negative
            # edges never involve the artificial global aggregator.
            if virtual_node_mask is not None:
                valid_nodes = torch.where(graph_node_mask & ~virtual_node_mask)[0]
            else:
                valid_nodes = torch.where(graph_node_mask)[0]
            
            # Random sample from valid nodes
            src_sample_idx = torch.randint(0, num_graph_nodes, (sample_size,), device=device)
            dst_sample_idx = torch.randint(0, num_graph_nodes, (sample_size,), device=device)
            
            # Map to global node indices
            neg_src = valid_nodes[src_sample_idx]
            neg_dst = valid_nodes[dst_sample_idx]
            
            # Remove self-loops
            non_self_loop = neg_src != neg_dst
            neg_src = neg_src[non_self_loop][:graph_neg_samples]
            neg_dst = neg_dst[non_self_loop][:graph_neg_samples]
            
            if len(neg_src) > 0:
                neg_edges = torch.stack([neg_src, neg_dst], dim=0)
                neg_edge_list.append(neg_edges)
        
        if len(neg_edge_list) > 0:
            return torch.cat(neg_edge_list, dim=1)
        else:
            return torch.empty((2, 0), dtype=remaining_edge_index.dtype, device=device)

    def forward(self, batch):
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        edge_attr = batch.get("x_1", None)
        batch_indices = batch.batch_0

        num_nodes = x_0.size(0)
        device = x_0.device

        remaining_edge_index, pos_edge_index, remaining_edge_attr, _ = self.sample_edges(
            edge_index, edge_attr, batch_indices, num_nodes, device
        )

        extra = {"edge_attr": remaining_edge_attr} if remaining_edge_attr is not None else {}
        node_embeddings = self.backbone(
            x_0,
            remaining_edge_index,
            batch=batch_indices,
            edge_weight=edge_weight,
            **extra,
        )

        neg_edge_index = self.sample_negative_edges(
            remaining_edge_index,
            pos_edge_index.size(1),
            batch_indices,
            num_nodes,
            device,
        )

        return {
            "x_0": node_embeddings,
            "pos_edge_index": pos_edge_index,
            "neg_edge_index": neg_edge_index,
            "edge_index": edge_index,
            "remaining_edge_index": remaining_edge_index,
            "batch_0": batch_indices,
            "labels": batch.y if hasattr(batch, "y") else None,
        }


class VAEGNNWrapper(_EdgeSamplingGNNWrapper):
    r"""VGAE-style encoder: GNN :math:`\rightarrow` :math:`\mu`, :math:`\log\sigma^2` :math:`\rightarrow` sample :math:`z`.

    Passes ``z`` as ``x_0`` to the readout for inner-product edge logits. Set
    ``residual_connections: false`` in config (latent space :math:`\neq` input features).

    Parameters
    ----------
    latent_dim : int
        Dimension of :math:`z` per node.
    variational : bool
        If True, use reparameterization; if False, :math:`z=\mu` (GAE-style).
    """

    def __init__(
        self,
        backbone: nn.Module,
        edge_sample_ratio: float = 0.5,
        neg_sample_ratio: float = 1.0,
        sampling_method: str = "sparse",
        latent_dim: int = 32,
        variational: bool = True,
        **kwargs,
    ):
        super().__init__(
            backbone,
            edge_sample_ratio,
            neg_sample_ratio,
            sampling_method,
            **kwargs,
        )
        self.latent_dim = latent_dim
        self.variational = variational
        enc_dim = getattr(backbone, "out_channels", None)
        if enc_dim is None:
            enc_dim = getattr(backbone, "hidden_dim", None)
        if enc_dim is None:
            enc_dim = kwargs["out_channels"]
        self._encoder_dim = int(enc_dim)
        self.fc_mu = nn.Linear(self._encoder_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._encoder_dim, latent_dim) if variational else None

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch):
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        edge_attr = batch.get("x_1", None)
        virtual_node_mask = batch.get("virtual_node_mask", None)
        batch_indices = batch.batch_0

        num_nodes = x_0.size(0)
        device = x_0.device

        remaining_edge_index, pos_edge_index, remaining_edge_attr, _ = self.sample_edges(
            edge_index, edge_attr, batch_indices, num_nodes, device, virtual_node_mask
        )

        extra = {"edge_attr": remaining_edge_attr} if remaining_edge_attr is not None else {}
        h = self.backbone(
            x_0,
            remaining_edge_index,
            batch=batch_indices,
            edge_weight=edge_weight,
            **extra,
        )

        mu = self.fc_mu(h)
        if self.variational and self.fc_logvar is not None:
            logvar = self.fc_logvar(h)
            z = self.reparameterize(mu, logvar)
        else:
            logvar = None
            z = mu

        neg_edge_index = self.sample_negative_edges(
            remaining_edge_index,
            pos_edge_index.size(1),
            batch_indices,
            num_nodes,
            device,
        )

        return {
            # x_0 = raw backbone output [N, out_channels] — same shape as batch.x_0 so
            # AbstractWrapper residual connections (batch.x_0 + x_0 → LayerNorm) work fine.
            "x_0": h,
            # z = latent sample [N, latent_dim] — used by VGAEReadOut for edge scoring.
            # Kept separate so latent_dim can differ freely from out_channels.
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "pos_edge_index": pos_edge_index,
            "neg_edge_index": neg_edge_index,
            "edge_index": edge_index,
            "remaining_edge_index": remaining_edge_index,
            "batch_0": batch_indices,
            "labels": batch.y if hasattr(batch, "y") else None,
        }
