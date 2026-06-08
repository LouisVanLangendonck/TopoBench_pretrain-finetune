"""Randomly rewire a graph's edges while preserving the number of directed edges.

Applied as a pre-transform ablation BEFORE positional / structural encodings
(LapPE, RWSE) so that those encodings reflect the *new* random topology rather
than the real one.  The transform is deterministic after the first run because
PyG's pre-transform caching applies it once and stores the result on disk.

For each directed edge (u, v) in ``edge_index`` both endpoints are resampled
uniformly at random from ``[0, num_nodes)``.  Edge attributes (``edge_attr``)
are reset to zeros of the original shape because they no longer correspond to
any meaningful bond or connection.  If the original ``edge_attr`` was ``None``,
it remains ``None`` after shuffling.
"""

import torch
import torch_geometric


class ShuffleEdges(torch_geometric.transforms.BaseTransform):
    """Replace ``edge_index`` with uniformly random directed edges.

    The number of directed edges (``edge_index.shape[1]``) is preserved.
    ``edge_attr`` is reset to zeros matching the original shape, or kept
    ``None`` if it was already absent.

    Parameters
    ----------
    seed : int
        Seed for the internal random generator.
    **kwargs :
        Absorbed to allow construction via ``DataTransform(transform_name=...,
        seed=..., ...)`` without raising on the extra ``transform_name`` key.
    """

    def __init__(self, seed: int = 0, **kwargs):
        super().__init__()
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
            Graph with randomly rewired ``edge_index`` (and zeroed /
            cleared ``edge_attr``).  Empty graphs are returned unchanged.
        """
        n = data.num_nodes
        if n == 0 or data.edge_index is None or data.edge_index.shape[1] == 0:
            return data

        m = data.edge_index.shape[1]
        src = torch.randint(0, n, (m,), generator=self._generator)
        dst = torch.randint(0, n, (m,), generator=self._generator)
        data.edge_index = torch.stack([src, dst], dim=0)

        if data.edge_attr is not None:
            edge_dim = data.edge_attr.shape[-1] if data.edge_attr.dim() > 1 else 1
            data.edge_attr = torch.zeros(
                m, edge_dim, dtype=data.edge_attr.dtype
            )

        return data
