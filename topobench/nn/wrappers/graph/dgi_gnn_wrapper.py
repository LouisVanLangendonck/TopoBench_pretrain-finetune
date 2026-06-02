"""Wrapper for DGI (Deep Graph Infomax) pre-training with GNN models."""

import torch
import torch.nn as nn

from topobench.nn.wrappers.base import AbstractWrapper


class DGIGNNWrapper(AbstractWrapper):
    r"""Wrapper for Deep Graph Infomax (DGI) pre-training with GNN models.

    DGI maximizes mutual information between patch representations (node embeddings)
    and graph-level summary vectors. This is achieved by:
    1. Encoding the original graph to get node embeddings (positive samples)
    2. Creating negative samples (corrupted version or different graphs)
    3. Encoding the negative samples
    4. Using a discriminator to distinguish between positive and negative node-summary pairs
    
    Two corruption strategies are supported:
    - "feature_shuffle": Shuffles node features within each graph
      * Works for both transductive (single graph) and inductive (multiple graphs)
      * For transductive: the only corruption option
      * For inductive: one of two options
      
    - "graph_diffusion": Uses node embeddings from different graphs as negatives
      * Only for inductive setting (requires batch_size >= 2)
      * As described in DGI paper: "our corruption function simply samples
        a different graph from the training set"
      * Each graph is encoded, then paired with a different graph's node embeddings

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model (GNN).
    corruption_type : str, optional
        Type of corruption to apply (default: "feature_shuffle").
        - "feature_shuffle": Randomly permute node features (transductive & inductive)
        - "graph_diffusion": Use different graphs as negatives (inductive only)
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.
        
    Notes
    -----
    For "graph_diffusion", the batch must contain at least 2 graphs (batch_size >= 2).
    """

    def __init__(
        self,
        backbone: nn.Module,
        corruption_type: str = "feature_shuffle",
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(backbone, **kwargs)
        
        self.corruption_type = corruption_type
        self.verbose = verbose
        self.forward_count = 0  # Track number of forward passes
        
        if corruption_type not in ["feature_shuffle", "graph_diffusion"]:
            raise ValueError(
                f"corruption_type must be 'feature_shuffle' or 'graph_diffusion', got {corruption_type}"
            )
    
    def corrupt_graph(self, x, batch_indices, virtual_node_mask=None):
        """Corrupt node features to create negative samples.

        For "feature_shuffle": Shuffle features within each graph (transductive).
        Virtual nodes (marked by ``virtual_node_mask``) are excluded from the
        shuffle — their zero features are preserved so the global aggregator
        node is not given a meaningless corrupted representation.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (num_nodes, num_features).
        batch_indices : torch.Tensor
            Batch assignment for each node.
        virtual_node_mask : torch.Tensor or None
            Boolean tensor of shape (num_nodes,) with ``True`` at virtual node
            positions, or ``None`` when no virtual node is present.

        Returns
        -------
        torch.Tensor
            Corrupted node features with same shape as input.
        """
        if self.corruption_type != "feature_shuffle":
            raise ValueError(
                "corrupt_graph should only be called with feature_shuffle. "
                "graph_diffusion uses different graphs directly."
            )

        batch_ids = batch_indices.unique()
        corrupted_x = x.clone()

        for batch_id in batch_ids:
            graph_mask = (batch_indices == batch_id)
            # Exclude virtual node from the shuffle pool
            if virtual_node_mask is not None:
                graph_mask = graph_mask & ~virtual_node_mask
            graph_indices = torch.where(graph_mask)[0]

            perm = torch.randperm(len(graph_indices), device=x.device)
            shuffled_indices = graph_indices[perm]
            corrupted_x[graph_indices] = x[shuffled_indices]

        return corrupted_x
    
    def forward(self, batch):
        r"""Forward pass for DGI encoding.

        Two corruption strategies:
        - "feature_shuffle": Corrupts features within each graph, encodes corrupted version
        - "graph_diffusion": Uses different graphs as negatives (for inductive learning)
        
        For downstream tasks, use the standard GNNWrapper instead by loading
        only the backbone encoder without this wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing:
            - x_0: Node embeddings from positive (original) graph
            - x_0_corrupted: Node embeddings from negative samples
            - batch_0: Batch indices for positive samples
            - batch_0_corrupted: Batch indices for negative samples (for graph_diffusion)
            - edge_index: Edge indices (for readout)
        """
        # Input features AFTER feature encoding
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        edge_attr = batch.get("x_1", None)
        virtual_node_mask = batch.get("virtual_node_mask", None)
        batch_indices = batch.batch_0
        
        # Build extra kwargs to forward only when values are present, so that
        # backbones which don't declare these parameters (e.g. plain PyG GIN)
        # are not given unexpected keyword arguments.
        extra = {}
        if edge_attr is not None:
            extra["edge_attr"] = edge_attr

        if self.corruption_type == "feature_shuffle":
            # Transductive: corrupt features, keep same graph structure.
            # Edge topology and edge attributes are unchanged for both views.
            # Encode original graph (positive samples)
            h_positive = self.backbone(
                x_0,
                edge_index,
                batch=batch_indices,
                edge_weight=edge_weight,
                **extra,
            )
            
            # Create corrupted graph (negative samples)
            x_corrupted = self.corrupt_graph(x_0, batch_indices, virtual_node_mask)
            
            # Encode corrupted graph (negative samples)
            h_negative = self.backbone(
                x_corrupted,
                edge_index,
                batch=batch_indices,
                edge_weight=edge_weight,
                **extra,
            )
            
            # Return both positive and negative embeddings for discriminator
            model_out = {
                "x_0": h_positive,  # Positive node embeddings
                "x_0_corrupted": h_negative,  # Negative node embeddings
                "batch_0": batch_indices,
                "batch_0_corrupted": batch_indices,  # Same batch structure
                "edge_index": edge_index,
                "labels": batch.y if hasattr(batch, "y") else None,
            }
            
        elif self.corruption_type == "graph_diffusion":
            # Inductive: use different graphs as negatives
            batch_ids = batch_indices.unique()
            num_graphs = len(batch_ids)
            
            if num_graphs < 2:
                raise ValueError(
                    "graph_diffusion requires at least 2 graphs in the batch, "
                    f"but got {num_graphs}. Increase batch size or use feature_shuffle."
                )
            
            # Logging for verification
            self.forward_count += 1
            if self.verbose or self.forward_count <= 3:
                print(f"\n{'='*80}")
                print(f"DGI graph_diffusion - Forward pass #{self.forward_count}")
                print(f"Mode: {'TRAINING' if self.training else 'VALIDATION'}")
                print(f"Batch contains {num_graphs} graphs with IDs: {batch_ids.tolist()}")
                for gid in batch_ids:
                    n_nodes = (batch_indices == gid).sum().item()
                    print(f"  Graph {gid}: {n_nodes} nodes")
            
            # Encode ALL graphs in the batch ONCE
            h_all = self.backbone(
                x_0,
                edge_index,
                batch=batch_indices,
                edge_weight=edge_weight,
                **extra,
            )
            
            # MEMORY OPTIMIZATION: Work directly with h_all, no copying
            # Create a random permutation mapping: for each graph, pick a different graph as negative
            num_graphs = len(batch_ids)
            
            # Generate TRULY RANDOM permutation ensuring no graph maps to itself
            # For each graph, randomly select a different graph as its negative
            perm = torch.zeros(num_graphs, dtype=torch.long, device=batch_indices.device)
            
            for i in range(num_graphs):
                # Create list of all other graphs (excluding current graph)
                # This ensures no self-pairing
                other_indices = list(range(num_graphs))
                other_indices.remove(i)
                
                # Randomly pick one of the other graphs
                chosen_idx = other_indices[torch.randint(0, len(other_indices), (1,)).item()]
                perm[i] = chosen_idx
            
            # Map original batch indices to negative graph batch indices
            # batch_indices contains values from batch_ids
            # Create a mapping: batch_ids[i] -> batch_ids[perm[i]]
            neg_graph_mapping = batch_ids[perm]
            
            # Create negative batch indices by mapping each node to its negative graph
            # This is a vectorized operation - no loops!
            batch_negative = torch.zeros_like(batch_indices)
            for i, orig_id in enumerate(batch_ids):
                mask = (batch_indices == orig_id)
                batch_negative[mask] = neg_graph_mapping[i]
            
            # Logging
            if self.verbose or self.forward_count <= 3:
                print("\nGraph pairings (positive -> negative):")
                for i, orig_id in enumerate(batch_ids):
                    neg_id = neg_graph_mapping[i]
                    print(f"  Graph {orig_id.item()} paired with Graph {neg_id.item()} (as negative)")
                print(f"{'='*80}\n")
            
            # CRITICAL OPTIMIZATION: Reuse h_all directly!
            # Positive: use original embeddings with original batch
            # Negative: use same embeddings BUT with shuffled batch assignment
            # This way the readout will compute different graph summaries
            
            model_out = {
                "x_0": h_all,  # Positive node embeddings (original)
                "x_0_corrupted": h_all,  # SAME embeddings, but will be paired differently via batch indices!
                "batch_0": batch_indices,  # Original batch assignment
                "batch_0_corrupted": batch_negative,  # Shuffled batch assignment
                "edge_index": edge_index,
                "labels": batch.y if hasattr(batch, "y") else None,
            }
        
        else:
            raise ValueError(f"Unknown corruption_type: {self.corruption_type}")
        
        return model_out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"corruption_type={self.corruption_type})"
        )


