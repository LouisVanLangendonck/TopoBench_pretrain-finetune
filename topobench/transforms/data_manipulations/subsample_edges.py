"""Randomly retain a fraction of a graph's directed edges.

Used as a pre-transform ablation BEFORE positional / structural encodings so
that PSEs reflect the partially degraded topology.
"""

import torch
import torch_geometric


class SubsampleEdges(torch_geometric.transforms.BaseTransform):
    """Retain a random subset of directed edges from ``edge_index``.

    Parameters
    ----------
    frac : float
        Fraction of directed edges to keep, in ``(0, 1]``.
    seed : int
        Seed for the internal random generator.
    **kwargs :
        Absorbed to allow construction via ``DataTransform(transform_name=...,
        frac=..., seed=..., ...)`` without raising on extra keys.
    """

    def __init__(self, frac: float = 0.75, seed: int = 0, **kwargs):
        super().__init__()
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"frac must be in (0, 1], got {frac}")
        self.frac = frac
        self.seed = seed
        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

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
            Graph with a random subset of edges retained.  Empty graphs are
            returned unchanged.
        """
        if (
            data.edge_index is None
            or data.edge_index.shape[1] == 0
        ):
            return data

        m = data.edge_index.shape[1]
        keep = int(m * self.frac)
        if keep <= 0:
            dtype = data.edge_index.dtype
            data.edge_index = torch.zeros((2, 0), dtype=dtype)
            if data.edge_attr is not None:
                edge_dim = (
                    data.edge_attr.shape[-1]
                    if data.edge_attr.dim() > 1
                    else 1
                )
                data.edge_attr = torch.zeros(
                    0, edge_dim, dtype=data.edge_attr.dtype
                )
            return data

        if keep >= m:
            return data

        perm = torch.randperm(m, generator=self._generator)[:keep]
        data.edge_index = data.edge_index[:, perm]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[perm]

        return data
