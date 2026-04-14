import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
import glob

class ARCDataset(Dataset):
    def __init__(self, data_path, max_size=30, max_pairs=5):
        """
        Dataset for ARC tasks, lazy loads from a local JSON file or a directory of JSON files.
        If data_path is a directory, it finds all .json files.

        Args:
            data_path (str): Path to JSON file or directory.
            max_size (int): Maximum grid dimension for padding (default 30 for ARC).
            max_pairs (int): Max number of (input, output) pairs to yield per task.
        """
        super().__init__()
        self.max_size = max_size
        self.max_pairs = max_pairs

        # We need a map from index to (file_path, task_key_or_index) for lazy loading
        self.task_registry = []

        if os.path.isdir(data_path):
            json_files = glob.glob(os.path.join(data_path, '**', '*.json'), recursive=True)
            for file_path in json_files:
                # We can do a quick scan to count tasks without keeping data in memory
                try:
                    with open(file_path, 'r') as f:
                        # For massive files, loading the structure once might be unavoidable
                        # But typically NVARC splits them nicely
                        data = json.load(f)
                        if isinstance(data, dict) and "train" in data:
                            self.task_registry.append((file_path, None))
                        elif isinstance(data, dict):
                            for key in data.keys():
                                self.task_registry.append((file_path, key))
                        elif isinstance(data, list):
                            for i in range(len(data)):
                                self.task_registry.append((file_path, i))
                except Exception as e:
                    print(f"Error reading structure of {file_path}: {e}")
        else:
            with open(data_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict) and "train" in data:
                self.task_registry.append((data_path, None))
            elif isinstance(data, dict):
                for key in data.keys():
                    self.task_registry.append((data_path, key))
            elif isinstance(data, list):
                for i in range(len(data)):
                    self.task_registry.append((data_path, i))

    def __len__(self):
        return len(self.task_registry)

    def __getitem__(self, idx):
        file_path, key = self.task_registry[idx]

        # Lazy load data
        with open(file_path, 'r') as f:
            data = json.load(f)

        if key is None:
            task = data
        else:
            task = data[key]

        train_examples = task.get("train", [])

        # Pad number of pairs to max_pairs
        inputs = []
        input_masks = []
        outputs = []
        output_masks = []
        pair_mask = [] # True if valid pair, False if padding pair

        for i in range(self.max_pairs):
            if i < len(train_examples):
                example = train_examples[i]
                inp = example.get("input", [])
                out = example.get("output", [])

                inp_padded, inp_m = self._pad_grid(inp)
                out_padded, out_m = self._pad_grid(out)

                inputs.append(inp_padded)
                input_masks.append(inp_m)
                outputs.append(out_padded)
                output_masks.append(out_m)
                pair_mask.append(True)
            else:
                # Padding pair
                inp_padded, inp_m = self._pad_grid([])
                out_padded, out_m = self._pad_grid([])
                inputs.append(inp_padded)
                input_masks.append(inp_m)
                outputs.append(out_padded)
                output_masks.append(out_m)
                pair_mask.append(False)

        return {
            "input": torch.stack(inputs),
            "input_mask": torch.stack(input_masks),
            "output": torch.stack(outputs),
            "output_mask": torch.stack(output_masks),
            "pair_mask": torch.tensor(pair_mask, dtype=torch.bool)
        }

    def _pad_grid(self, grid):
        """
        Pads a 2D grid to max_size x max_size using 10 as pad token.
        Returns:
            padded_grid: Tensor of shape (max_size, max_size)
            mask: Tensor of shape (max_size, max_size), 1 for valid, 0 for padding
        """
        grid = np.array(grid)
        if grid.size == 0:
             return torch.full((self.max_size, self.max_size), 10, dtype=torch.int64), torch.zeros((self.max_size, self.max_size), dtype=torch.bool)

        if grid.ndim == 1:
            grid = grid.reshape(1, -1)

        h, w = grid.shape

        # Clip if somehow larger
        h = min(h, self.max_size)
        w = min(w, self.max_size)
        grid = grid[:h, :w]

        padded = np.full((self.max_size, self.max_size), 10, dtype=np.int64)
        mask = np.zeros((self.max_size, self.max_size), dtype=np.bool_)

        padded[:h, :w] = grid
        mask[:h, :w] = True

        return torch.tensor(padded), torch.tensor(mask)
