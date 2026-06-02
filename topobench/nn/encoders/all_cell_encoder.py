"""Class to apply BaseEncoder to the features of higher order structures."""

import torch
import torch_geometric
from torch_geometric.nn.norm import GraphNorm

from topobench.nn.encoders.base import AbstractFeatureEncoder


class AllCellFeatureEncoder(AbstractFeatureEncoder):
    r"""Encoder class to apply BaseEncoder.

    The BaseEncoder is applied to the features of higher order
    structures. The class creates a BaseEncoder for each dimension specified in
    selected_dimensions. Then during the forward pass, the BaseEncoders are
    applied to the features of the corresponding dimensions.

    Parameters
    ----------
    in_channels : list[int]
        Input dimensions for the features.
    out_channels : list[int]
        Output dimensions for the features.
    proj_dropout : float, optional
        Dropout for the BaseEncoders (default: 0).
    selected_dimensions : list[int], optional
        List of indexes to apply the BaseEncoders to (default: None).
    graph_norm : bool, optional
        Whether to apply GraphNorm before the linear projection in each
        BaseEncoder (default: False).
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        proj_dropout=0,
        selected_dimensions=None,
        graph_norm=False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimensions = (
            selected_dimensions
            if (
                selected_dimensions is not None
            )  # and len(selected_dimensions) <= len(self.in_channels))
            else range(len(self.in_channels))
        )
        for i in self.dimensions:
            setattr(
                self,
                f"encoder_{i}",
                BaseEncoder(
                    self.in_channels[i],
                    self.out_channels,
                    dropout=proj_dropout,
                    graph_norm=graph_norm,
                ),
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, dimensions={self.dimensions})"

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Forward pass.

        The method applies the BaseEncoders to the features of the selected_dimensions.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object which should contain x_{i} features for each i in the selected_dimensions.

        Returns
        -------
        torch_geometric.data.Data
            Output data object with updated x_{i} features.
        """
        if not hasattr(data, "x_0") and hasattr(data, "x"):
            data.x_0 = data.x
        if getattr(data, "batch", None) is not None and not hasattr(data, "batch_0"):
            data.batch_0 = data.batch

        # For plain graph datasets (no lifting transform), raw edge features
        # live in data.edge_attr rather than data.x_1.  Promote them so that
        # the dimension-1 encoder can process them, and derive batch_1 from
        # the source-node assignment (each edge belongs to the same graph as
        # its source node).
        if (
            1 in self.dimensions
            and not hasattr(data, "x_1")
            and getattr(data, "edge_attr", None) is not None
        ):
            data.x_1 = data.edge_attr
            data.batch_1 = data.batch_0[data.edge_index[0]]

        for i in self.dimensions:
            if hasattr(data, f"x_{i}") and hasattr(self, f"encoder_{i}"):
                batch = getattr(data, f"batch_{i}")
                data[f"x_{i}"] = getattr(self, f"encoder_{i}")(
                    data[f"x_{i}"], batch
                )
        return data


class BaseEncoder(torch.nn.Module):
    r"""Base encoder class used by AllCellFeatureEncoder.

    This class uses a linear layer with optional GraphNorm, ReLU activation,
    and dropout.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimensions of output features.
    dropout : float, optional
        Percentage of channels to discard after the linear layer (default: 0).
    graph_norm : bool, optional
        Whether to apply GraphNorm before the linear projection (default: False).
    """

    def __init__(self, in_channels, out_channels, dropout=0, graph_norm=False):
        super().__init__()
        self.BN = GraphNorm(in_channels) if graph_norm else None
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.linear.in_features}, out_channels={self.linear.out_features}, graph_norm={self.BN is not None})"

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of dimensions [N, in_channels].
        batch : torch.Tensor
            The batch vector which assigns each element to a specific example.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [N, out_channels].
        """
        if self.BN is not None:
            x = self.BN(x, batch=batch) if batch.shape[0] > 0 else self.BN(x)
        x = self.linear(x)
        x = self.dropout(self.relu(x))
        return x
