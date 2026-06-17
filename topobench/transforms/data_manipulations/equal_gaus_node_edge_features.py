"""Replace existing node and edge features with equal Gaussian vectors.

Unlike ``EqualGausFeatures`` (used for datasets *without* semantic features),
this transform is for graphs that already carry informative ``x`` and/or
``edge_attr``: every node receives the same random Gaussian vector and every
edge receives the same (independent) random Gaussian vector.
"""

import torch
import torch_geometric


class EqualGausNodeEdgeFeatures(torch_geometric.transforms.BaseTransform):
    r"""Neutralise semantic node and edge attributes with equal Gaussians.

    Parameters
    ----------
    mean : float
        Mean of the Gaussian distribution.
    std : float
        Standard deviation of the Gaussian distribution.
    num_features : int
        Node feature dimensionality.
    num_edge_features : int, optional
        Edge feature dimensionality.  When provided, ``edge_attr`` is replaced
        for every edge; when omitted, existing ``edge_attr`` is left unchanged.
    **kwargs :
        Absorbed for ``DataTransform`` construction compatibility.
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 0.1,
        num_features: int = 1,
        num_edge_features: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.mean = mean
        self.std = std
        self.num_features = num_features
        self.num_edge_features = num_edge_features

        self._node_vector = torch.normal(
            mean=mean, std=std, size=(1, num_features)
        )
        self._edge_vector = (
            torch.normal(mean=mean, std=std, size=(1, num_edge_features))
            if num_edge_features is not None
            else None
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mean={self.mean!r}, std={self.std!r}, "
            f"num_features={self.num_features!r}, "
            f"num_edge_features={self.num_edge_features!r})"
        )

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        data.x = self._node_vector.expand(data.num_nodes, -1).to(
            dtype=data.x.dtype if data.x is not None else torch.float
        )

        if (
            self._edge_vector is not None
            and data.edge_attr is not None
            and data.edge_index is not None
            and data.edge_index.shape[1] > 0
        ):
            m = data.edge_index.shape[1]
            data.edge_attr = self._edge_vector.expand(m, -1).to(
                dtype=data.edge_attr.dtype
            )

        return data
