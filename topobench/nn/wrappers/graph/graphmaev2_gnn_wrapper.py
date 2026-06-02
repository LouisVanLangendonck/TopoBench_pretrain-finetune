"""Wrapper for GraphMAEv2 pre-training with GNN models."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from topobench.nn.wrappers.base import AbstractWrapper


def sce_loss(x, y, alpha=1):
    """Scaled Cosine Error loss for latent representation matching."""
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


class GraphMAEv2GNNWrapper(AbstractWrapper):
    r"""Wrapper for GraphMAEv2 pre-training with GNN models.

    GraphMAEv2 improves upon GraphMAE with:
    1. EMA (Exponential Moving Average) encoder as teacher
    2. Latent representation loss
    3. Re-masking strategy during decoding (handled in readout)

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model.
    mask_rate : float, optional
        The rate of nodes to mask (default: 0.5).
    replace_rate : float, optional
        The rate of masked nodes to replace with random features (default: 0.0).
    drop_edge_rate : float, optional
        The rate of edges to drop during training (default: 0.0).
    momentum : float, optional
        Momentum for EMA encoder update (default: 0.996).
    delayed_ema_epoch : int, optional
        Number of epochs to delay EMA updates (default: 0).
    lam : float, optional
        Weight for latent loss (default: 1.0).
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.
    """

    def __init__(
        self,
        backbone: nn.Module,
        mask_rate: float = 0.5,
        replace_rate: float = 0.0,
        drop_edge_rate: float = 0.0,
        momentum: float = 0.996,
        delayed_ema_epoch: int = 0,
        lam: float = 1.0,
        **kwargs
    ):
        super().__init__(backbone, **kwargs)
        
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.drop_edge_rate = drop_edge_rate
        self.momentum = momentum
        self.delayed_ema_epoch = delayed_ema_epoch
        self.lam = lam
        
        # Calculate mask token rate
        self.mask_token_rate = 1 - self.replace_rate
        
        # Get feature dimension from kwargs
        self.feature_dim = kwargs.get("out_channels")
        
        if self.feature_dim is None:
            raise ValueError("Cannot determine feature dimension. Please provide 'out_channels' in kwargs.")
        
        # Create learnable mask tokens
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.feature_dim))
        nn.init.xavier_normal_(self.enc_mask_token)
        
        # Projector for latent loss (student)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.PReLU(),
            nn.Linear(256, self.feature_dim),
        )
        
        # Predictor for latent loss (only on student side)
        self.predictor = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        # EMA encoder (teacher) - will be initialized after first forward
        self.encoder_ema = None
        self.projector_ema = None
        self._ema_initialized = False
        
        # Track current epoch for delayed EMA
        self.current_epoch = 0
    
    def _init_ema(self):
        """Initialize EMA encoder and projector from the main encoder."""
        if self._ema_initialized:
            return
            
        # Deep copy the backbone for EMA
        self.encoder_ema = copy.deepcopy(self.backbone)
        self.projector_ema = copy.deepcopy(self.projector)
        
        # Freeze EMA parameters
        for p in self.encoder_ema.parameters():
            p.requires_grad = False
            p.detach_()
        for p in self.projector_ema.parameters():
            p.requires_grad = False
            p.detach_()
        
        self._ema_initialized = True
    
    @torch.no_grad()
    def ema_update(self):
        """Update EMA encoder and projector with momentum."""
        if not self._ema_initialized:
            return
            
        m = self.momentum
        
        # Update encoder EMA
        for param_q, param_k in zip(self.backbone.parameters(), self.encoder_ema.parameters(), strict=False):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        
        # Update projector EMA
        for param_q, param_k in zip(self.projector.parameters(), self.projector_ema.parameters(), strict=False):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
    
    def encoding_mask_noise(self, x, num_nodes, device, virtual_node_mask=None):
        """Apply masking and noise to node features.

        Virtual nodes (if present) are excluded from the mask candidate pool —
        masking a node that already has zero features would cause the model to
        waste capacity learning to reconstruct a zero vector.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (num_nodes, num_features).
        num_nodes : int
            Number of nodes in the graph.
        device : torch.device
            Device to use.
        virtual_node_mask : torch.Tensor or None
            Boolean mask of shape (num_nodes,) with ``True`` at virtual node
            positions, or ``None`` when no virtual node is present.

        Returns
        -------
        tuple
            (masked_x, mask_nodes, keep_nodes)
        """
        # Build the candidate pool — exclude virtual nodes
        if virtual_node_mask is not None:
            candidates = torch.where(~virtual_node_mask)[0]
        else:
            candidates = torch.arange(num_nodes, device=device)

        perm = candidates[torch.randperm(len(candidates), device=device)]
        num_mask_nodes = int(self.mask_rate * len(candidates))
        
        # Split into masked and kept nodes
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        
        # Clone features
        out_x = x.clone()
        
        if self.replace_rate > 0:
            # Some masked nodes get replaced with random features
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=device)
            
            token_nodes = mask_nodes[perm_mask[:int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
            
            noise_to_be_chosen = torch.randperm(num_nodes, device=device)[:num_noise_nodes]
            
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x[mask_nodes] = 0.0
            token_nodes = mask_nodes
        
        # Add learnable mask token
        out_x[token_nodes] = out_x[token_nodes] + self.enc_mask_token
        
        return out_x, mask_nodes, keep_nodes
    
    def drop_edges(self, edge_index, edge_attr, num_edges, device, virtual_node_mask=None):
        """Drop edges randomly, keeping edge_attr aligned with edge_index.

        Edges that touch the virtual node are always preserved — dropping them
        would disconnect the global aggregator node from the graph.
        """
        if self.drop_edge_rate <= 0:
            return edge_index, edge_attr

        mask = torch.rand(num_edges, device=device) > self.drop_edge_rate
        # Force-keep edges connected to the virtual node
        if virtual_node_mask is not None:
            vn_edge = virtual_node_mask[edge_index[0]] | virtual_node_mask[edge_index[1]]
            mask = mask | vn_edge
        new_edge_attr = edge_attr[mask] if edge_attr is not None else None
        return edge_index[:, mask], new_edge_attr
    
    def forward(self, batch):
        r"""Forward pass for GraphMAEv2 encoding with masking and latent loss.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing encoded representations and latent loss components.
        """
        # Initialize EMA on first forward
        self._init_ema()
        
        # Input features AFTER feature encoding
        if not hasattr(batch, "x_0") and hasattr(batch, "x"):
            batch.x_0 = batch.x
        if getattr(batch, "batch", None) is not None and not hasattr(batch, "batch_0"):
            batch.batch_0 = batch.batch
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        edge_attr = batch.get("x_1", None)
        virtual_node_mask = batch.get("virtual_node_mask", None)
        batch_indices = batch.batch_0
        
        num_nodes = x_0.size(0)
        device = x_0.device
        
        # Store ORIGINAL RAW features for reconstruction
        if hasattr(batch, "x_raw"): # Normally we have this cause I added it as pre-transform
            x_raw_original = batch.x_raw.clone()
        else:
            # Fallback
            x_raw_original = x_0.clone()
        
        # Apply masking to the ENCODED features (x_0) for student input.
        # Virtual nodes are excluded from the mask candidate pool.
        masked_x, mask_nodes, keep_nodes = self.encoding_mask_noise(
            x_0, num_nodes, device, virtual_node_mask
        )
        
        # Drop edges for student (training only); keep edge_attr aligned.
        # Virtual-node edges are always preserved.
        if self.training and self.drop_edge_rate > 0:
            use_edge_index, use_edge_attr = self.drop_edges(
                edge_index, edge_attr, edge_index.size(1), device, virtual_node_mask
            )
        else:
            use_edge_index, use_edge_attr = edge_index, edge_attr
        
        # Build extra kwargs so backbones that don't declare edge_attr (e.g. GIN)
        # are not given unexpected keyword arguments.
        extra_full = {"edge_attr": edge_attr} if edge_attr is not None else {}
        extra_drop = {"edge_attr": use_edge_attr} if use_edge_attr is not None else {}

        # Encode with student encoder (masked input)
        enc_rep = self.backbone(
            masked_x,
            use_edge_index,
            batch=batch_indices,
            edge_weight=edge_weight,
            **extra_drop,
        )
        
        # Compute latent loss components
        latent_loss = torch.tensor(0.0, device=device)
        
        if self.training and self._ema_initialized:
            # EMA encoder sees clean ENCODED input (no masking)
            # This makes sense: we want to match the latent representations
            # of masked vs clean encoded inputs (not raw inputs)
            with torch.no_grad():
                latent_target = self.encoder_ema(
                    x_0,  # Clean encoded input (no masking)
                    edge_index,  # Full edges
                    batch=batch_indices,
                    edge_weight=edge_weight,
                    **extra_full,  # Full (unsubsetted) edge attributes
                )
                # Project and detach
                latent_target = self.projector_ema(latent_target[keep_nodes])
            
            # Student projection and prediction (on keep nodes)
            latent_pred = self.projector(enc_rep[keep_nodes])
            latent_pred = self.predictor(latent_pred)
            
            # Compute latent loss (SCE)
            latent_loss = sce_loss(latent_pred, latent_target, alpha=1)
        
        # Prepare outputs for readout
        model_out = {
            "x_0": enc_rep,  # Encoded representations from GNN
            "x_raw_original": x_raw_original,  # ORIGINAL RAW features (for reconstruction)
            "mask_nodes": mask_nodes,  # Which nodes were masked
            "keep_nodes": keep_nodes,  # Which nodes were kept
            "latent_loss": latent_loss,  # Latent representation loss
            "labels": batch.y if hasattr(batch, "y") else None,
            "batch_0": batch_indices,
            "edge_index": edge_index,  # For GNN decoder (re-masking requires message passing)
            "edge_weight": edge_weight,  # For GNN decoder (if available)
        }
        
        return model_out
    
    def on_epoch_end(self, epoch):
        """Called at the end of each epoch to update EMA."""
        self.current_epoch = epoch
        if self.training and epoch >= self.delayed_ema_epoch:
            self.ema_update()
