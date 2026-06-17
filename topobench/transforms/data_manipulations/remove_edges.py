"""Remove all edges from a graph, leaving isolated nodes.

Used as a pre-transform ablation BEFORE positional / structural encodings so
that PSEs reflect the edgeless topology.  Applied after any dataset-specific
feature-generation transforms.
"""

import torch
import torch_geometric


class RemoveEdges(torch_geometric.transforms.BaseTransform):
    """Clear ``edge_index`` (and ``edge_attr``) so every node is isolated.

    Parameters
    ----------
    **kwargs :
        Absorbed to allow construction via ``DataTransform(transform_name=...,
        ...)`` without raising on extra keys.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        """Apply the transform.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.

        Returns
        -------
        torch_geometric.data.Data
            Graph with empty ``edge_index``.  ``edge_attr`` is cleared when
            present; otherwise left ``None``.
        """
        if data.edge_index is None:
            return data

        dtype = data.edge_index.dtype
        data.edge_index = torch.zeros((2, 0), dtype=dtype)

        if data.edge_attr is not None:
            edge_dim = data.edge_attr.shape[-1] if data.edge_attr.dim() > 1 else 1
            data.edge_attr = torch.zeros(
                0, edge_dim, dtype=data.edge_attr.dtype
            )

        return data
