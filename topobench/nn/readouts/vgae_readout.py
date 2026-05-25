"""VGAE readout: score edges from node embeddings (dot / MLP decoders)."""

import torch
import torch.nn as nn
import torch_geometric

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class VGAEReadOut(AbstractZeroCellReadOut):
    r"""Edge scoring for VGAE pretraining on a minibatch of pos/neg edges."""

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int = 1,
        decoder_type: str = "dot",
        decoder_hidden_dim: int = 256,
        task_level: str = "node",
        **kwargs,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            task_level=task_level,
            logits_linear_layer=False,
            **kwargs,
        )

        self.decoder_type = decoder_type
        self.decoder_hidden_dim = decoder_hidden_dim

        self.decoder = self._build_decoder(
            decoder_type, hidden_dim, out_channels, decoder_hidden_dim
        )

    def _build_decoder(
        self,
        decoder_type: str,
        hidden_dim: int,
        out_channels: int,
        decoder_hidden_dim: int,
    ) -> nn.Module:
        if decoder_type == "dot":
            return DotProductDecoder()

        if decoder_type == "concat_mlp":
            return nn.Sequential(
                nn.Linear(hidden_dim * 2, decoder_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(decoder_hidden_dim, out_channels),
            )

        if decoder_type == "hadamard_mlp":
            return nn.Sequential(
                nn.Linear(hidden_dim, decoder_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(decoder_hidden_dim, out_channels),
            )

        raise ValueError(
            f"Unknown decoder type: {decoder_type}. "
            "Available options: 'dot', 'concat_mlp', 'hadamard_mlp'"
        )

    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        # Use z (latent sample) for edge scoring; fall back to x_0 for compatibility.
        node_embeddings = model_out.get("z", model_out["x_0"])
        pos_edge_index = model_out["pos_edge_index"]
        neg_edge_index = model_out["neg_edge_index"]

        num_pos_edges = pos_edge_index.size(1)
        num_neg_edges = neg_edge_index.size(1)

        all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

        pos_labels = torch.ones(num_pos_edges, dtype=torch.float, device=node_embeddings.device)
        neg_labels = torch.zeros(num_neg_edges, dtype=torch.float, device=node_embeddings.device)
        edge_labels = torch.cat([pos_labels, neg_labels], dim=0)

        src_embeddings = node_embeddings[all_edge_index[0]]
        dst_embeddings = node_embeddings[all_edge_index[1]]

        if self.decoder_type == "dot":
            edge_scores = self.decoder(src_embeddings, dst_embeddings)
        elif self.decoder_type == "concat_mlp":
            edge_repr = torch.cat([src_embeddings, dst_embeddings], dim=-1)
            edge_scores = self.decoder(edge_repr).squeeze(-1)
        elif self.decoder_type == "hadamard_mlp":
            edge_repr = src_embeddings * dst_embeddings
            edge_scores = self.decoder(edge_repr).squeeze(-1)
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")

        model_out["logits"] = edge_scores
        model_out["labels"] = edge_labels
        model_out["num_pos_edges"] = num_pos_edges
        model_out["num_neg_edges"] = num_neg_edges
        model_out["all_edge_index"] = all_edge_index

        return model_out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"decoder_type={self.decoder_type}, "
            f"hidden_dim={self.hidden_dim}, "
            f"out_channels={self.out_channels})"
        )


class DotProductDecoder(nn.Module):
    """Inner-product decoder (Kipf VGAE): scalar logit per edge."""

    def forward(self, src_embeddings, dst_embeddings):
        return (src_embeddings * dst_embeddings).sum(dim=-1)
