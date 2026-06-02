"""GPSE backbone: deep ResGatedGCN stack with skip connections.

Implements the encoder trunk of the GPSE model from:
    "Graph Positional and Structural Encoder"
    https://arxiv.org/abs/2307.07107

This is the backbone-only version (pre_mp + GNNStackStage) without the PSE
prediction head, suitable for supervised fine-tuning or downstream tasks.
Virtual node support is handled as a pre-transform, not inside the model.
"""

from os import wait
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ResGatedGraphConv
from torch_geometric.nn.resolver import activation_resolver


class _MLPBlock(nn.Module):
    """Single linear layer with optional BatchNorm, dropout, and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_bn: bool = True,
        dropout: float = 0.2,
        act: Optional[str] = "relu",
        has_l2norm: bool = True,
    ):
        super().__init__()
        self.has_l2norm = has_l2norm
        bias = not has_bn
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.1) if has_bn else None
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else None
        self.act = activation_resolver(act) if act is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.drop is not None:
            x = self.drop(x)
        if self.act is not None:
            x = self.act(x)
        if self.has_l2norm:
            x = F.normalize(x, p=2, dim=-1)
        return x


class _ResGatedConvBlock(nn.Module):
    """One ResGatedGraphConv layer with BatchNorm, dropout, and L2-norm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_bn: bool = True,
        dropout: float = 0.2,
        act: Optional[str] = "relu",
        has_l2norm: bool = True,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        self.has_l2norm = has_l2norm
        self.conv = ResGatedGraphConv(
            in_channels,
            out_channels,
            bias=not has_bn,
            edge_dim=edge_dim,
        )
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.1) if has_bn else None
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else None
        self.act = activation_resolver(act) if act is not None else None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        if self.bn is not None:
            x = self.bn(x)
        if self.drop is not None:
            x = self.drop(x)
        if self.act is not None:
            x = self.act(x)
        if self.has_l2norm:
            x = F.normalize(x, p=2, dim=-1)
        return x


class GPSEBackbone(nn.Module):
    r"""ResGatedGCN-based deep GNN backbone, the encoder trunk of GPSE.

    Consists of an optional pre-MP MLP followed by a stack of
    :class:`~torch_geometric.nn.ResGatedGraphConv` layers with skipsum
    or skipconcat residual connections and final L2-normalisation.

    The GPSE prediction head (``post_mp``) is deliberately omitted — this
    class is a backbone for supervised/fine-tuned downstream tasks, not for
    PSE pretraining.

    Parameters
    ----------
    in_channels : int
        Dimension of input node features (after the feature encoder).
    hidden_channels : int
        Width of all hidden layers.
    num_layers : int, optional
        Number of ResGatedGraphConv message-passing layers. Default: 10.
    layers_pre_mp : int, optional
        Number of MLP layers applied before message passing. Default: 1.
    stage_type : str, optional
        Residual strategy: ``"skipsum"`` adds the residual (requires
        ``in_channels == hidden_channels``), ``"skipconcat"`` concatenates
        (increases width). Default: ``"skipsum"``.
    has_bn : bool, optional
        Whether to apply BatchNorm after each layer. Default: ``True``.
    has_l2norm : bool, optional
        Whether to apply L2-normalisation after each layer. Default: ``True``.
    final_l2norm : bool, optional
        Whether to L2-normalise the final node embeddings. Default: ``True``.
    dropout : float, optional
        Dropout rate. Default: 0.2.
    act : str, optional
        Activation function name (passed to PyG ``activation_resolver``).
        Default: ``"relu"``.
    edge_dim : int, optional
        Dimensionality of edge features. Pass ``hidden_channels`` when
        edge features are available; leave ``None`` to ignore edge features.
        Default: ``None``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 10,
        layers_pre_mp: int = 1,
        stage_type: str = "skipsum",
        has_bn: bool = True,
        has_l2norm: bool = True,
        final_l2norm: bool = True,
        dropout: float = 0.2,
        act: str = "relu",
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        assert stage_type in {"skipsum", "skipconcat", "none"}, (
            f"stage_type must be 'skipsum', 'skipconcat', or 'none', got '{stage_type}'"
        )
        self.stage_type = stage_type
        self.final_l2norm = final_l2norm
        self.num_layers = num_layers
        self.out_channels = hidden_channels
        self.edge_dim = edge_dim

        # ------------------------------------------------------------------ #
        # Pre-MP MLP
        # ------------------------------------------------------------------ #
        pre_mp_layers = []
        if layers_pre_mp > 0:
            d_in = in_channels
            for i in range(layers_pre_mp):
                d_out = hidden_channels
                pre_mp_layers.append(
                    _MLPBlock(
                        d_in,
                        d_out,
                        has_bn=has_bn,
                        dropout=dropout,
                        act=act,
                        has_l2norm=has_l2norm,
                    )
                )
                d_in = d_out
            self.pre_mp = nn.Sequential(*pre_mp_layers)
        else:
            self.pre_mp = None

        # ------------------------------------------------------------------ #
        # Message-passing stack
        # ------------------------------------------------------------------ #
        # After pre_mp, the node feature width is always hidden_channels.
        # For skipsum: all layers are hidden_channels → hidden_channels.
        # For skipconcat: after layer i the output is cat([x_in, x_out]) so
        #   layer i+1 input width = hidden_channels + (i+1)*hidden_channels.
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if stage_type == "skipconcat" and i > 0:
                d_in = hidden_channels + i * hidden_channels
            else:
                d_in = hidden_channels
            self.convs.append(
                _ResGatedConvBlock(
                    in_channels=d_in,
                    out_channels=hidden_channels,
                    has_bn=has_bn,
                    dropout=dropout,
                    act=act,
                    has_l2norm=has_l2norm,
                    edge_dim=edge_dim,
                )
            )
        # Width of the final node embedding
        if stage_type == "skipconcat":
            # Last layer still outputs hidden_channels (not concatenated)
            self.out_channels = hidden_channels

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``[N, in_channels]``.
        edge_index : torch.Tensor
            Edge indices of shape ``[2, E]``.
        batch : torch.Tensor, optional
            Batch vector of shape ``[N]``. Not used internally but kept for
            API compatibility with the TopoBench wrapper contract.
        edge_attr : torch.Tensor, optional
            Edge feature matrix of shape ``[E, edge_dim]``. Only consumed
            when the backbone was built with ``edge_dim`` set.
        **kwargs
            Ignored; absorbed for forward-compatibility.

        Returns
        -------
        torch.Tensor
            Node embedding matrix of shape ``[N, hidden_channels]``.
        """
        if self.pre_mp is not None:
            x = self.pre_mp(x)

        # ResGatedGraphConv asserts edge_attr is not None iff edge_dim is set.
        # When the backbone was built with edge_dim but the current batch carries
        # no edge attributes (e.g. datasets without edge features), substitute a
        # zero tensor so the assertion is satisfied and the model trains normally.
        if edge_attr is None and self.edge_dim is not None:
            edge_attr = torch.zeros(
                edge_index.size(1), self.edge_dim,
                device=x.device, dtype=x.dtype,
            )

        for i, conv in enumerate(self.convs):
            x_in = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            if self.stage_type == "skipsum":
                x = x + x_in
            elif self.stage_type == "skipconcat" and i < self.num_layers - 1:
                x = torch.cat([x_in, x], dim=-1)

        if self.final_l2norm:
            x = F.normalize(x, p=2, dim=-1)

        return x
