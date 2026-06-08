"""Replace node features with i.i.d. standard Gaussian noise.

Used as a pre-transform ablation: applied BEFORE positional / structural
encodings (LapPE, RWSE) so that the PSEs still reflect the *real* graph
topology while the semantic node attributes are destroyed.  The transform
is deterministic after the first run because PyG's pre-transform caching
applies it once and stores the result on disk.
"""

import torch
import torch_geometric


class RandomizeNodeFeatures(torch_geometric.transforms.BaseTransform):
    """Replace ``data.x`` with i.i.d. Gaussian noise of the same shape.

    Parameters
    ----------
    seed : int
        Seed for the internal random generator.  Different seeds produce
        different random-feature datasets (and therefore different cache
        directories inside the PreProcessor).
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
            Graph with ``data.x`` replaced by Gaussian noise.  If ``data.x``
            is ``None`` or empty the graph is returned unchanged (no features
            to randomise).
        """
        if data.x is None or data.x.numel() == 0:
            return data
        data.x = torch.randn(
            data.x.shape, generator=self._generator
        ).to(data.x.dtype)
        return data
