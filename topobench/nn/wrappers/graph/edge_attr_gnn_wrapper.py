"""GNN wrapper that forwards encoded edge features to the backbone."""

from topobench.nn.wrappers.base import AbstractWrapper


class EdgeAttrGNNWrapper(AbstractWrapper):
    r"""Wrapper for GNN backbones that consume edge features.

    Extends the standard GNN wrapper by also passing the encoded edge
    features (``batch.x_1``, populated by the feature encoder from raw
    ``edge_attr``) to the backbone via the ``edge_attr`` keyword argument.

    This wrapper is required whenever the backbone has an ``edge_dim``
    parameter (e.g. :class:`~topobench.nn.backbones.GPSEBackbone` built with
    ``edge_dim`` set). For models that do not use edge features, use the
    standard :class:`~topobench.nn.wrappers.GNNWrapper` instead.
    """

    def forward(self, batch):
        r"""Forward pass passing edge features to the backbone.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batched graph data. Expected attributes:

            * ``x_0`` – encoded node features ``[N, hidden_dim]``
            * ``edge_index`` – edge connectivity ``[2, E]``
            * ``batch_0`` – node-to-graph assignment ``[N]``
            * ``x_1`` – encoded edge features ``[E, hidden_dim]`` (optional)
            * ``edge_weight`` – scalar edge weights (optional)

        Returns
        -------
        dict
            Dictionary with keys ``"x_0"``, ``"labels"``, ``"batch_0"``.
        """
        pass  # BP-19 (EdgeAttrWrapper): inspect `batch.x_0.shape` (encoded nodes), `batch.x_1.shape` (encoded edges)
        x_0 = self.backbone(
            batch.x_0,
            batch.edge_index,
            batch=batch.batch_0,
            edge_attr=batch.get("x_1", None),
            edge_weight=batch.get("edge_weight", None),
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0

        return model_out
