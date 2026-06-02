"""Virtual-node pre-transform for graph datasets."""

import torch_geometric
from torch_geometric.transforms import VirtualNode as _PyGVirtualNode
import torch


class VirtualNodeTransform(torch_geometric.transforms.BaseTransform):
    r"""Adds a virtual node to each graph in the dataset.

    Wraps :class:`torch_geometric.transforms.VirtualNode`. For each graph:

    * One extra node is appended with ``x = zeros``.
    * Bidirectional edges are added between the virtual node and every real
      node, extending ``edge_index`` accordingly.
    * If ``edge_attr`` is present, zero rows are appended for the new
      virtual-node edges, keeping ``edge_attr`` aligned with ``edge_index``.

    Using a virtual node allows global information to propagate across the
    entire graph in a single message-passing step, which is beneficial for
    deep GNNs such as :class:`~topobench.nn.backbones.GPSEBackbone`.

    Parameters
    ----------
    **kwargs : optional
        Ignored; kept for config-system compatibility.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._transform = _PyGVirtualNode()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Apply virtual-node augmentation.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.

        Returns
        -------
        torch_geometric.data.Data
            Graph with one virtual node appended per graph.
            ``data.virtual_node_mask`` is a ``bool`` tensor of length
            ``num_nodes`` (after augmentation) with the last element set to
            ``True`` to mark the virtual node.  Pretraining wrappers use this
            mask to exclude the virtual node from masking, edge sampling, and
            augmentation operations.
        """
        data = self._transform(data)
        # Mark the last node (the virtual node just appended) so that
        # pretraining wrappers can protect it from augmentation operations.
        vn_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        vn_mask[-1] = True
        data.virtual_node_mask = vn_mask
        return data
