"""Loaders for Long Range Graph Benchmark (LRGB) datasets."""

import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import LRGBDataset

from topobench.data.loaders.base import AbstractLoader


class LRGBDatasetLoader(AbstractLoader):
    """Load LRGB datasets with predefined splits.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset (e.g., "Peptides-func")
            - data_type: Type of the dataset (e.g., "LRGBDataset")
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)
        self.datasets: list[Dataset] = []

    def load_dataset(self) -> Dataset:
        """Load the LRGB dataset with predefined splits.

        Returns
        -------
        Dataset
            The combined dataset with predefined splits.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        self._load_splits()
        split_idx = self._prepare_split_idx()
        combined_dataset = self._combine_splits()
        combined_dataset.split_idx = split_idx
        return combined_dataset

    def _load_splits(self) -> None:
        """Load the dataset splits for the specified LRGB dataset."""
        dataset_name = self.parameters.data_name.lower()
        for split in ["train", "val", "test"]:
            self.datasets.append(
                LRGBDataset(
                    root=str(self.root_data_dir),
                    name=dataset_name,
                    split=split,
                )
            )

    def _prepare_split_idx(self) -> dict[str, np.ndarray]:
        """Prepare the split indices for the dataset.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary mapping split names to index arrays.
        """
        split_idx = {"train": np.arange(len(self.datasets[0]))}
        split_idx["valid"] = np.arange(
            len(self.datasets[0]),
            len(self.datasets[0]) + len(self.datasets[1]),
        )
        split_idx["test"] = np.arange(
            len(self.datasets[0]) + len(self.datasets[1]),
            len(self.datasets[0])
            + len(self.datasets[1])
            + len(self.datasets[2]),
        )
        return split_idx

    def _combine_splits(self) -> Dataset:
        """Combine the dataset splits into a single dataset.

        Returns
        -------
        Dataset
            The combined dataset containing all splits.
        """
        combined_dataset = self.datasets[0] + self.datasets[1] + self.datasets[2]
        for data in combined_dataset:
            data.x = data.x.to(torch.float)
            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr.to(torch.float)
            if data.y is not None and data.y.dim() > 1:
                data.y = data.y.squeeze(0)
        return combined_dataset

    def get_data_dir(self) -> Path:
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
        """
        return os.path.join(
            self.root_data_dir, self.parameters.data_name.lower()
        )
