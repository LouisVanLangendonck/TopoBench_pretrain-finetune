"""Unit tests for TBModel.process_outputs."""

import pytest
import torch
from unittest.mock import MagicMock

from topobench.model.model import TBModel


def make_model(task_level):
    """Instantiate TBModel with mocked dependencies for a given task_level.

    Parameters
    ----------
    task_level : str
        The task level to assign to the readout mock.

    Returns
    -------
    TBModel
        A TBModel instance with mocked backbone, readout, loss, evaluator and optimizer.
    """
    backbone = MagicMock()
    backbone.parameters.return_value = []

    readout = MagicMock()
    readout.task_level = task_level
    readout.parameters.return_value = []

    loss = MagicMock()

    evaluator = MagicMock()

    optimizer = MagicMock()
    optimizer.configure_optimizer.return_value = {"optimizer": MagicMock()}

    feature_encoder = MagicMock()
    feature_encoder.parameters.return_value = []

    model = TBModel(
        backbone=backbone,
        readout=readout,
        loss=loss,
        feature_encoder=feature_encoder,
        evaluator=evaluator,
        optimizer=optimizer,
    )
    return model


class TestProcessOutputs:
    """Tests for TBModel.process_outputs covering all branches."""

    def _make_batch(self, n=10):
        """Create a simple batch mock with train/val/test masks.

        Parameters
        ----------
        n : int
            Number of nodes.

        Returns
        -------
        MagicMock
            Batch mock with boolean masks.
        """
        batch = MagicMock()
        # First 6 are train, next 2 val, last 2 test
        batch.train_mask = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool)
        batch.val_mask = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 0, 0], dtype=torch.bool)
        batch.test_mask = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.bool)
        return batch

    def _make_model_out(self, n=10):
        """Create a sample model output dict.

        Parameters
        ----------
        n : int
            Number of nodes.

        Returns
        -------
        dict
            Model output with logits, labels, and an extra key.
        """
        return {
            "logits": torch.randn(n, 3),
            "labels": torch.randint(0, 3, (n,)),
            "x_0": torch.randn(n, 8),
        }

    # ------------------------------------------------------------------
    # node-level masking
    # ------------------------------------------------------------------

    def test_node_training_filters_by_train_mask(self):
        """process_outputs with task_level='node' and state 'Training' filters by train_mask."""
        model = make_model("node")
        model.state_str = "Training"
        batch = self._make_batch()
        model_out = self._make_model_out()

        result = model.process_outputs(model_out, batch)

        n_train = batch.train_mask.sum().item()
        assert result["logits"].shape[0] == n_train
        assert result["labels"].shape[0] == n_train
        # non-masked keys are untouched
        assert result["x_0"].shape[0] == 10

    def test_node_validation_filters_by_val_mask(self):
        """process_outputs with task_level='node' and state 'Validation' filters by val_mask."""
        model = make_model("node")
        model.state_str = "Validation"
        batch = self._make_batch()
        model_out = self._make_model_out()

        result = model.process_outputs(model_out, batch)

        n_val = batch.val_mask.sum().item()
        assert result["logits"].shape[0] == n_val
        assert result["labels"].shape[0] == n_val

    def test_node_test_filters_by_test_mask(self):
        """process_outputs with task_level='node' and state 'Test' filters by test_mask."""
        model = make_model("node")
        model.state_str = "Test"
        batch = self._make_batch()
        model_out = self._make_model_out()

        result = model.process_outputs(model_out, batch)

        n_test = batch.test_mask.sum().item()
        assert result["logits"].shape[0] == n_test
        assert result["labels"].shape[0] == n_test

    def test_node_invalid_state_raises_value_error(self):
        """process_outputs with task_level='node' and an invalid state_str raises ValueError."""
        model = make_model("node")
        model.state_str = "Invalid"
        batch = self._make_batch()
        model_out = self._make_model_out()

        with pytest.raises(ValueError, match="Invalid state_str"):
            model.process_outputs(model_out, batch)

    # ------------------------------------------------------------------
    # no-op task levels
    # ------------------------------------------------------------------

    def test_graph_level_returns_unchanged(self):
        """process_outputs with task_level='graph' returns model_out unchanged."""
        model = make_model("graph")
        model.state_str = "Training"
        batch = self._make_batch()
        model_out = self._make_model_out()
        original_logits = model_out["logits"].clone()

        result = model.process_outputs(model_out, batch)

        assert result["logits"].shape == original_logits.shape
        assert torch.equal(result["logits"], original_logits)

    def test_node_inductive_returns_unchanged(self):
        """process_outputs with task_level='node_inductive' returns model_out unchanged (inductive bug-fix path)."""
        model = make_model("node_inductive")
        model.state_str = "Training"
        batch = self._make_batch()
        model_out = self._make_model_out()
        original_logits = model_out["logits"].clone()

        result = model.process_outputs(model_out, batch)

        assert result["logits"].shape == original_logits.shape
        assert torch.equal(result["logits"], original_logits)
