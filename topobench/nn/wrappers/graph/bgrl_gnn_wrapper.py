"""Wrapper for BGRL pre-training with GNN backbones. Code adapted from https://github.com/nerdslab/bgrl/blob/main/bgrl"""

import copy

import torch
import torch.nn as nn
from torch_geometric.utils import dropout_edge

from topobench.nn.wrappers.base import AbstractWrapper


class BGRLGNNWrapper(AbstractWrapper):
    r"""BGRL wrapper with online and target encoders.

    This wrapper keeps two encoder branches:
    - Target encoder: EMA-updated, frozen (`self.backbone`)
    - Online encoder: trainable branch (`self.online_encoder`)

    Parameters
    ----------
    backbone : torch.nn.Module
        Encoder backbone used as target encoder in this wrapper.
    drop_edge_rate_1 : float, optional
        Edge drop probability for view 1.
    drop_edge_rate_2 : float, optional
        Edge drop probability for view 2.
    drop_feature_rate_1 : float, optional
        Feature drop probability for view 1.
    drop_feature_rate_2 : float, optional
        Feature drop probability for view 2.
    momentum : float, optional
        EMA momentum for updating target encoder.
    force_undirected : bool, optional
        Whether edge dropout should preserve undirected edges.
    **kwargs : dict
        Additional wrapper kwargs.
    """

    def __init__(
        self,
        backbone: nn.Module,
        drop_edge_rate_1: float = 0.1,
        drop_edge_rate_2: float = 0.2,
        drop_feature_rate_1: float = 0.3,
        drop_feature_rate_2: float = 0.2,
        momentum: float = 0.99,
        force_undirected: bool = False,
        **kwargs,
    ):
        super().__init__(backbone, **kwargs)

        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")

        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        self.momentum = momentum
        self.force_undirected = force_undirected

        self.online_encoder = copy.deepcopy(self.backbone)
        self._reset_target_encoder_parameters()
        self._set_requires_grad(self.backbone, requires_grad=False)
        self._set_requires_grad(self.online_encoder, requires_grad=True)

    def _reset_target_encoder_parameters(self) -> None:
        """Reset target encoder to avoid exact online copy at init."""
        if hasattr(self.backbone, "reset_parameters"):
            self.backbone.reset_parameters()

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for param in module.parameters():
            param.requires_grad = requires_grad

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """Momentum update target encoder using online encoder parameters."""
        for online_param, target_param in zip(
            self.online_encoder.parameters(), self.backbone.parameters(), strict=False
        ):
            target_param.data.mul_(self.momentum).add_(
                online_param.data, alpha=1.0 - self.momentum
            )

    @staticmethod
    def _drop_features(x: torch.Tensor, drop_rate: float) -> torch.Tensor:
        if drop_rate <= 0.0:
            return x
        feature_keep_mask = (
            torch.rand(x.size(1), device=x.device, dtype=torch.float32) > drop_rate
        )
        x_aug = x.clone()
        x_aug[:, ~feature_keep_mask] = 0.0
        return x_aug

    def _drop_edges(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        drop_rate: float,
        virtual_node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if drop_rate <= 0.0:
            return edge_index, edge_attr
        edge_index_aug, edge_mask = dropout_edge(
            edge_index,
            p=drop_rate,
            force_undirected=self.force_undirected,
            training=self.training,
        )
        # Always preserve edges that touch the virtual node so the global
        # aggregator stays connected to the graph after augmentation.
        if virtual_node_mask is not None:
            vn_edge = virtual_node_mask[edge_index[0]] | virtual_node_mask[edge_index[1]]
            edge_mask = edge_mask | vn_edge
            edge_index_aug = edge_index[:, edge_mask]
        edge_attr_aug = edge_attr[edge_mask] if edge_attr is not None else None
        return edge_index_aug, edge_attr_aug

    def _augment_view(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        drop_edge_rate: float,
        drop_feature_rate: float,
        virtual_node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x_aug = self._drop_features(x, drop_feature_rate)
        edge_index_aug, edge_attr_aug = self._drop_edges(
            edge_index, edge_attr, drop_edge_rate, virtual_node_mask
        )
        return x_aug, edge_index_aug, edge_attr_aug

    def _encode(
        self,
        encoder: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_indices: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_kwargs = {"batch": batch_indices}
        if edge_weight is not None:
            encoder_kwargs["edge_weight"] = edge_weight
        if edge_attr is not None:
            encoder_kwargs["edge_attr"] = edge_attr
        try:
            return encoder(x, edge_index, **encoder_kwargs)
        except TypeError:
            encoder_kwargs.pop("edge_attr", None)
            return encoder(x, edge_index, **encoder_kwargs)

    def forward(self, batch):
        r"""Create two views and cross-encode with online/target encoders.
        
        BGRL architecture per the official paper:
        - Online encoder processes view 1 → predictor → compared with target(view 2)
        - Symmetrically: online encoder processes view 2 → predictor → compared with target(view 1)
        
        Both encoders see both views (with independent augmentations), and the loss
        cross-pairs them. The target encoder EMA update happens AFTER optimizer.step()
        via the on_before_zero_grad hook in TBModel, not during this forward pass.
        """
        x_0 = batch.x_0
        edge_index = batch.edge_index
        batch_indices = batch.batch_0
        edge_weight = batch.get("edge_weight", None)
        # Use the *encoded* edge features (x_1) not the raw edge_attr
        edge_attr = batch.get("x_1", None)
        virtual_node_mask = batch.get("virtual_node_mask", None)

        x_1, edge_index_1, edge_attr_1 = self._augment_view(
            x_0,
            edge_index,
            edge_attr,
            self.drop_edge_rate_1,
            self.drop_feature_rate_1,
            virtual_node_mask,
        )
        x_2, edge_index_2, edge_attr_2 = self._augment_view(
            x_0,
            edge_index,
            edge_attr,
            self.drop_edge_rate_2,
            self.drop_feature_rate_2,
            virtual_node_mask,
        )

        online_h_1 = self._encode(
            self.online_encoder,
            x_1,
            edge_index_1,
            batch_indices=batch_indices,
            edge_weight=edge_weight if edge_index_1.size(1) == edge_index.size(1) else None,
            edge_attr=edge_attr_1,
        )
        online_h_2 = self._encode(
            self.online_encoder,
            x_2,
            edge_index_2,
            batch_indices=batch_indices,
            edge_weight=edge_weight if edge_index_2.size(1) == edge_index.size(1) else None,
            edge_attr=edge_attr_2,
        )

        with torch.no_grad():
            target_h_1 = self._encode(
                self.backbone,
                x_1,
                edge_index_1,
                batch_indices=batch_indices,
                edge_weight=edge_weight if edge_index_1.size(1) == edge_index.size(1) else None,
                edge_attr=edge_attr_1,
            )
            target_h_2 = self._encode(
                self.backbone,
                x_2,
                edge_index_2,
                batch_indices=batch_indices,
                edge_weight=edge_weight if edge_index_2.size(1) == edge_index.size(1) else None,
                edge_attr=edge_attr_2,
            )

        return {
            "x_0": online_h_1,
            "online_h_1": online_h_1,
            "online_h_2": online_h_2,
            "target_h_1": target_h_1,
            "target_h_2": target_h_2,
            "batch_0": batch_indices,
            "edge_index": edge_index,
            "labels": batch.y if hasattr(batch, "y") else None,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"drop_edge_rate_1={self.drop_edge_rate_1}, "
            f"drop_edge_rate_2={self.drop_edge_rate_2}, "
            f"drop_feature_rate_1={self.drop_feature_rate_1}, "
            f"drop_feature_rate_2={self.drop_feature_rate_2}, "
            f"momentum={self.momentum})"
        )
