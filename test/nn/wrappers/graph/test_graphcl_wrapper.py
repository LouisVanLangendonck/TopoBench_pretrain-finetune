"""Tests for GraphCL GNN wrapper, augmentations, and end-to-end integration."""

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn.models import GCN

from topobench.loss.dataset.graphcl_loss import GraphCLLoss
from topobench.nn.readouts.graphcl_readout import GraphCLReadOut
from topobench.nn.wrappers.graph.graphcl_gnn_wrapper import GraphCLGNNWrapper
from topobench.evaluator.graphcl_evaluator import GraphCLEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(num_nodes, num_features, num_edges=None, y=0):
    """Create a single random graph Data object."""
    if num_edges is None:
        num_edges = num_nodes * 2
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return Data(x=x, edge_index=edge_index, y=torch.tensor([y]))


def _make_batch(num_graphs=8, num_nodes=20, num_features=16, num_edges=40):
    """Create a batched graph with x_0, edge_index, and batch_0 fields."""
    graphs = [
        _make_graph(num_nodes, num_features, num_edges, y=i % 3)
        for i in range(num_graphs)
    ]
    batch = Batch.from_data_list(graphs)
    batch.x_0 = batch.x
    batch.batch_0 = batch.batch
    return batch


def _make_backbone(in_channels=16, hidden_channels=16, num_layers=2):
    """Create a simple GCN backbone for testing.

    in_channels must equal hidden_channels to satisfy the residual connection
    in AbstractWrapper (which adds model_out["x_0"] + batch["x_0"]).
    """
    return GCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
    )


# ---------------------------------------------------------------------------
# Augmentation Tests
# ---------------------------------------------------------------------------

class TestGraphCLAugmentations:
    """Tests for the augmentation methods in GraphCLGNNWrapper."""

    def setup_method(self):
        backbone = _make_backbone()
        self.wrapper = GraphCLGNNWrapper(
            backbone=backbone,
            aug1="drop_edge",
            aug2="mask_attr",
            aug_ratio1=0.2,
            aug_ratio2=0.2,
            out_channels=16,
            num_cell_dimensions=1,
        )

    def _get_graph_tensors(self, num_nodes=20, num_features=16, num_edges=40):
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        batch_indices = torch.zeros(num_nodes, dtype=torch.long)
        return x, edge_index, batch_indices

    def test_augment_none(self):
        x, edge_index, batch = self._get_graph_tensors()
        aug_x, aug_ei, aug_batch = self.wrapper.augment(
            x, edge_index, batch, "none", 0.2, x.device
        )
        assert torch.equal(aug_x, x)
        assert torch.equal(aug_ei, edge_index)
        assert torch.equal(aug_batch, batch)

    def test_augment_drop_node(self):
        x, edge_index, batch = self._get_graph_tensors()
        aug_x, aug_ei, aug_batch = self.wrapper.augment(
            x, edge_index, batch, "drop_node", 0.3, x.device
        )
        assert aug_x.size(0) <= x.size(0)
        assert aug_x.size(0) >= 1
        assert aug_x.size(1) == x.size(1)
        assert aug_batch.size(0) == aug_x.size(0)
        if aug_ei.numel() > 0:
            assert aug_ei.max() < aug_x.size(0)

    def test_augment_drop_edge(self):
        x, edge_index, batch = self._get_graph_tensors()
        aug_x, aug_ei, aug_batch = self.wrapper.augment(
            x, edge_index, batch, "drop_edge", 0.3, x.device
        )
        assert torch.equal(aug_x, x)
        assert aug_ei.size(1) <= edge_index.size(1)
        assert torch.equal(aug_batch, batch)

    def test_augment_mask_attr_replaces_whole_nodes(self):
        """Attribute masking should replace entire feature vectors of selected nodes."""
        torch.manual_seed(42)
        num_nodes, num_features = 100, 32
        x = torch.zeros(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, 50))
        batch = torch.zeros(num_nodes, dtype=torch.long)

        aug_x, _, _ = self.wrapper.augment(
            x, edge_index, batch, "mask_attr", 0.3, x.device
        )

        changed_mask = (aug_x != x).any(dim=1)
        num_changed = changed_mask.sum().item()
        expected_changed = max(int(num_nodes * 0.3), 1)
        assert num_changed == expected_changed, (
            f"Expected {expected_changed} nodes masked, got {num_changed}"
        )

        for i in range(num_nodes):
            if changed_mask[i]:
                assert (aug_x[i] != x[i]).all(), (
                    f"Node {i} should have ALL features replaced, not partial"
                )

    def test_augment_subgraph(self):
        x, edge_index, batch = self._get_graph_tensors()
        aug_x, aug_ei, aug_batch = self.wrapper.augment(
            x, edge_index, batch, "subgraph", 0.3, x.device
        )
        assert aug_x.size(0) <= x.size(0)
        assert aug_x.size(0) >= 1
        assert aug_batch.size(0) == aug_x.size(0)

    def test_augment_preserves_all_graphs_in_batch(self):
        """Each graph in the batch should have at least one node after augment."""
        num_nodes = 10
        num_graphs = 4
        x = torch.randn(num_nodes * num_graphs, 8)
        edge_index_list = []
        for g in range(num_graphs):
            offset = g * num_nodes
            ei = torch.randint(0, num_nodes, (2, 20)) + offset
            edge_index_list.append(ei)
        edge_index = torch.cat(edge_index_list, dim=1)
        batch = torch.repeat_interleave(
            torch.arange(num_graphs), num_nodes
        )

        for aug_type in ["drop_node", "subgraph"]:
            aug_x, aug_ei, aug_batch = self.wrapper.augment(
                x, edge_index, batch, aug_type, 0.5, x.device
            )
            present_graphs = torch.unique(aug_batch)
            assert len(present_graphs) == num_graphs, (
                f"{aug_type}: lost graphs. Expected {num_graphs}, got {len(present_graphs)}"
            )

    def test_augment_extreme_ratio_still_keeps_nodes(self):
        """Even with extreme ratios, every graph should keep at least 1 node."""
        x = torch.randn(50, 8)
        edge_index = torch.randint(0, 50, (2, 100))
        batch = torch.zeros(50, dtype=torch.long)

        # drop_node: ratio=0.99 means drop 99% of nodes
        aug_x, _, _ = self.wrapper.augment(
            x, edge_index, batch, "drop_node", 0.99, x.device
        )
        assert aug_x.size(0) >= 1

        # subgraph: ratio=0.01 means keep only 1% of nodes
        aug_x, _, _ = self.wrapper.augment(
            x, edge_index, batch, "subgraph", 0.01, x.device
        )
        assert aug_x.size(0) >= 1

    def test_augment_invalid_type_raises(self):
        x, edge_index, batch = self._get_graph_tensors()
        with pytest.raises(ValueError, match="Unknown augmentation"):
            self.wrapper.augment(
                x, edge_index, batch, "invalid", 0.2, x.device
            )


# ---------------------------------------------------------------------------
# Wrapper Forward Pass Tests
# ---------------------------------------------------------------------------

class TestGraphCLGNNWrapper:
    """Tests for the full GraphCLGNNWrapper forward pass."""

    def setup_method(self):
        self.num_features = 16
        self.hidden = self.num_features
        backbone = _make_backbone(
            in_channels=self.num_features, hidden_channels=self.hidden
        )
        self.wrapper = GraphCLGNNWrapper(
            backbone=backbone,
            aug1="drop_edge",
            aug2="mask_attr",
            aug_ratio1=0.2,
            aug_ratio2=0.2,
            readout_type="sum",
            out_channels=self.hidden,
            num_cell_dimensions=1,
        )

    def test_forward_output_keys(self):
        batch = _make_batch(num_graphs=4, num_features=self.num_features)
        out = self.wrapper(batch)
        assert "z1" in out
        assert "z2" in out
        assert "x_0" in out
        assert "batch_0" in out
        assert "labels" in out

    def test_z1_z2_are_graph_level(self):
        """z1 and z2 should have shape (num_graphs, hidden_dim)."""
        num_graphs = 4
        batch = _make_batch(num_graphs=num_graphs, num_features=self.num_features)
        out = self.wrapper(batch)
        assert out["z1"].shape == (num_graphs, self.hidden)
        assert out["z2"].shape == (num_graphs, self.hidden)

    def test_forward_runs_only_two_backbone_passes(self):
        """The wrapper should call the backbone exactly 2 times (no 3rd original pass)."""
        call_count = 0
        original_forward = self.wrapper.backbone.forward

        def counting_forward(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_forward(*args, **kwargs)

        self.wrapper.backbone.forward = counting_forward
        batch = _make_batch(num_graphs=4, num_features=self.num_features)
        self.wrapper(batch)
        assert call_count == 2

    def test_different_augmentations_give_different_views(self):
        """z1 and z2 should generally differ (different augmentations applied)."""
        batch = _make_batch(num_graphs=8, num_features=self.num_features)
        out = self.wrapper(batch)
        assert not torch.allclose(out["z1"], out["z2"], atol=1e-3)

    def test_pooling_types(self):
        """Ensure all three pooling types produce valid output."""
        for pool_type in ["mean", "max", "sum"]:
            backbone = _make_backbone(
                in_channels=self.num_features, hidden_channels=self.num_features
            )
            wrapper = GraphCLGNNWrapper(
                backbone=backbone,
                readout_type=pool_type,
                out_channels=self.num_features,
                num_cell_dimensions=1,
            )
            batch = _make_batch(num_graphs=4, num_features=self.num_features)
            out = wrapper(batch)
            assert out["z1"].shape == (4, self.num_features)


# ---------------------------------------------------------------------------
# End-to-End Integration Tests
# ---------------------------------------------------------------------------

class TestGraphCLEndToEnd:
    """Integration tests running the full pipeline: wrapper -> readout -> loss."""

    def setup_method(self):
        self.num_features = 16
        self.hidden = self.num_features
        self.proj_dim = 16

    def _build_pipeline(self, aug1="drop_edge", aug2="mask_attr"):
        backbone = _make_backbone(
            in_channels=self.num_features, hidden_channels=self.hidden
        )
        wrapper = GraphCLGNNWrapper(
            backbone=backbone,
            aug1=aug1,
            aug2=aug2,
            aug_ratio1=0.2,
            aug_ratio2=0.2,
            readout_type="sum",
            out_channels=self.hidden,
            num_cell_dimensions=1,
        )
        readout = GraphCLReadOut(
            hidden_dim=self.hidden,
            out_channels=self.proj_dim,
            projection_type="mlp",
            task_level="graph",
            num_cell_dimensions=1,
        )
        loss_fn = GraphCLLoss(temperature=0.5)
        evaluator = GraphCLEvaluator()
        return wrapper, readout, loss_fn, evaluator

    def test_full_forward_and_loss(self):
        wrapper, readout, loss_fn, evaluator = self._build_pipeline()
        batch = _make_batch(num_graphs=8, num_features=self.num_features)

        model_out = wrapper(batch)
        model_out = readout(model_out, batch)
        loss = loss_fn(model_out, batch)

        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_full_pipeline_with_evaluator(self):
        wrapper, readout, loss_fn, evaluator = self._build_pipeline()
        batch = _make_batch(num_graphs=8, num_features=self.num_features)

        model_out = wrapper(batch)
        model_out = readout(model_out, batch)
        loss = loss_fn(model_out, batch)
        model_out["loss"] = loss

        evaluator.update(model_out)
        metrics = evaluator.compute()
        assert "contrastive_loss" in metrics
        assert "alignment" in metrics
        assert "cosine_sim" in metrics

    def test_backward_pass(self):
        """Full pipeline should support gradient computation."""
        wrapper, readout, loss_fn, _ = self._build_pipeline()
        batch = _make_batch(num_graphs=8, num_features=self.num_features)

        model_out = wrapper(batch)
        model_out = readout(model_out, batch)
        loss = loss_fn(model_out, batch)

        loss.backward()

        for param in wrapper.backbone.parameters():
            assert param.grad is not None

        for param in readout.projection_head.parameters():
            assert param.grad is not None

    def test_training_step_reduces_loss(self):
        """A few optimization steps should reduce the loss."""
        wrapper, readout, loss_fn, _ = self._build_pipeline()
        batch = _make_batch(num_graphs=16, num_features=self.num_features)

        params = list(wrapper.parameters()) + list(readout.parameters())
        optimizer = torch.optim.Adam(params, lr=0.01)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            model_out = wrapper(batch)
            model_out = readout(model_out, batch)
            loss = loss_fn(model_out, batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    @pytest.mark.parametrize("aug1,aug2", [
        ("none", "none"),
        ("drop_node", "drop_edge"),
        ("mask_attr", "subgraph"),
        ("drop_edge", "mask_attr"),
        ("subgraph", "drop_node"),
    ])
    def test_all_augmentation_combinations(self, aug1, aug2):
        """Every valid augmentation pair should run without errors."""
        wrapper, readout, loss_fn, _ = self._build_pipeline(aug1=aug1, aug2=aug2)
        batch = _make_batch(num_graphs=4, num_features=self.num_features)

        model_out = wrapper(batch)
        model_out = readout(model_out, batch)
        loss = loss_fn(model_out, batch)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_eval_mode(self):
        """Pipeline should also work in eval mode (no errors)."""
        wrapper, readout, loss_fn, _ = self._build_pipeline()
        wrapper.eval()
        readout.eval()

        batch = _make_batch(num_graphs=4, num_features=self.num_features)
        with torch.no_grad():
            model_out = wrapper(batch)
            model_out = readout(model_out, batch)
            loss = loss_fn(model_out, batch)

        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# Parametrized Augmentation Variant Tests
# ---------------------------------------------------------------------------

class TestGraphCLAugmentationVariants:
    """Tests for the parametrised augmentation strategies across official repo variants."""

    def _make_wrapper(self, **overrides):
        kwargs = dict(
            backbone=_make_backbone(),
            aug1="drop_edge",
            aug2="mask_attr",
            out_channels=16,
            num_cell_dimensions=1,
        )
        kwargs.update(overrides)
        return GraphCLGNNWrapper(**kwargs)

    def _get_graph_tensors(self, num_nodes=50, num_features=16, num_edges=100):
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        batch = torch.zeros(num_nodes, dtype=torch.long)
        return x, edge_index, batch

    # -- mask_attr_strategy ------------------------------------------------

    def test_mask_attr_gaussian(self):
        wrapper = self._make_wrapper(mask_attr_strategy="gaussian")
        x = torch.zeros(50, 8)
        ei = torch.randint(0, 50, (2, 40))
        batch = torch.zeros(50, dtype=torch.long)
        aug_x, _, _ = wrapper.augment(x, ei, batch, "mask_attr", 0.2, x.device)
        changed = (aug_x != 0).any(dim=1)
        assert changed.any(), "Gaussian masking should produce non-zero replacements"

    def test_mask_attr_zeros(self):
        wrapper = self._make_wrapper(mask_attr_strategy="zeros")
        x = torch.ones(50, 8)
        ei = torch.randint(0, 50, (2, 40))
        batch = torch.zeros(50, dtype=torch.long)
        aug_x, _, _ = wrapper.augment(x, ei, batch, "mask_attr", 0.2, x.device)
        masked_nodes = (aug_x == 0).all(dim=1)
        expected = max(int(50 * 0.2), 1)
        assert masked_nodes.sum().item() == expected

    def test_mask_attr_mean(self):
        wrapper = self._make_wrapper(mask_attr_strategy="mean")
        torch.manual_seed(0)
        x = torch.randn(50, 8)
        mean_feat = x.mean(dim=0)
        ei = torch.randint(0, 50, (2, 40))
        batch = torch.zeros(50, dtype=torch.long)
        aug_x, _, _ = wrapper.augment(x, ei, batch, "mask_attr", 0.2, x.device)
        changed = (aug_x != x).any(dim=1)
        for i in range(50):
            if changed[i]:
                assert torch.allclose(aug_x[i], mean_feat, atol=1e-5)

    # -- edge_perturbation_mode -------------------------------------------

    def test_edge_drop_only(self):
        wrapper = self._make_wrapper(edge_perturbation_mode="drop_only")
        x, ei, batch = self._get_graph_tensors()
        aug_x, aug_ei, _ = wrapper.augment(x, ei, batch, "drop_edge", 0.3, x.device)
        assert aug_ei.size(1) <= ei.size(1)

    def test_edge_drop_and_add(self):
        """drop_and_add keeps total edge count the same (drops some, adds same number)."""
        wrapper = self._make_wrapper(edge_perturbation_mode="drop_and_add")
        x, ei, batch = self._get_graph_tensors()
        aug_x, aug_ei, _ = wrapper.augment(x, ei, batch, "drop_edge", 0.3, x.device)
        assert aug_ei.size(1) == ei.size(1)

    # -- subgraph_ratio_meaning -------------------------------------------

    def test_subgraph_keep_semantics(self):
        """With ratio_meaning='keep' and ratio=0.2, roughly 20% of nodes are kept."""
        wrapper = self._make_wrapper(subgraph_ratio_meaning="keep")
        x, ei, batch = self._get_graph_tensors(num_nodes=100)
        aug_x, _, _ = wrapper.augment(x, ei, batch, "subgraph", 0.2, x.device)
        assert aug_x.size(0) <= 30  # generous upper bound for 20% of 100

    def test_subgraph_drop_semantics(self):
        """With ratio_meaning='drop' and ratio=0.2, roughly 80% of nodes are kept."""
        wrapper = self._make_wrapper(subgraph_ratio_meaning="drop")
        x, ei, batch = self._get_graph_tensors(num_nodes=100, num_edges=800)
        aug_x, _, _ = wrapper.augment(x, ei, batch, "subgraph", 0.2, x.device)
        assert aug_x.size(0) >= 50  # generous lower bound for 80% of 100

    # -- invalid parameter validation -------------------------------------

    def test_invalid_mask_strategy_raises(self):
        with pytest.raises(ValueError, match="mask_attr_strategy"):
            self._make_wrapper(mask_attr_strategy="invalid")

    def test_invalid_edge_mode_raises(self):
        with pytest.raises(ValueError, match="edge_perturbation_mode"):
            self._make_wrapper(edge_perturbation_mode="invalid")

    def test_invalid_subgraph_ratio_raises(self):
        with pytest.raises(ValueError, match="subgraph_ratio_meaning"):
            self._make_wrapper(subgraph_ratio_meaning="invalid")

    def test_residual_connections_respected(self):
        """residual_connections must not be forced off; z1/z2 use post-residual nodes."""
        wrapper_off = self._make_wrapper(residual_connections=False)
        wrapper_on = self._make_wrapper(residual_connections=True)
        assert wrapper_off.residual_connections is False
        assert wrapper_on.residual_connections is True

        batch = _make_batch(num_graphs=4, num_nodes=12, num_features=16, num_edges=30)
        torch.manual_seed(0)
        out_off = wrapper_off(batch)
        torch.manual_seed(0)
        out_on = wrapper_on(batch)
        assert not torch.allclose(out_off["z1"], out_on["z1"])
