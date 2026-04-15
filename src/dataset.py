import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
import glob

class ARCDataset(Dataset):
    def __init__(self, data_path, max_size=30, max_pairs=5):
        """
        Dataset for ARC tasks. Each JSON file is one task containing a list
        of {input, output} pairs (possibly under a "root" key).

        Args:
            data_path (str): Path to a JSON file or directory of JSON files.
            max_size (int): Maximum grid dimension for padding.
            max_pairs (int): Max number of (input, output) pairs per task.
        """
        super().__init__()
        self.max_size = max_size
        self.max_pairs = max_pairs
        self.task_registry = []  # list of file paths, one per task

        if os.path.isdir(data_path):
            json_files = glob.glob(os.path.join(data_path, '**', '*.json'), recursive=True)
            print("Number of Files:")
            print(len(json_files))
            for file_path in json_files:
                try:
                    pairs = self._load_pairs(file_path)
                    if len(pairs) > 0:
                        self.task_registry.append(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        else:
            pairs = self._load_pairs(data_path)
            if len(pairs) > 0:
                self.task_registry.append(data_path)

        print(f"Loaded {len(self.task_registry)} tasks from {data_path}")

    def _load_pairs(self, file_path):
        """
        Load and return the list of {input, output} pairs from a JSON file.
        Handles these structures:
          - {"root": [{input, output}, ...]}   <- your format
          - {"train": [{input, output}, ...]}  <- standard ARC format
          - [{input, output}, ...]             <- bare list
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict):
            if "root" in data:
                return data["root"]
            elif "train" in data:
                return data["train"]
            else:
                # Unknown dict structure — try the first list-valued key
                for v in data.values():
                    if isinstance(v, list):
                        return v
                return []
        elif isinstance(data, list):
            return data

        return []

    def __len__(self):
        return len(self.task_registry)

    def __getitem__(self, idx):
        file_path = self.task_registry[idx]
        train_examples = self._load_pairs(file_path)

        inputs, input_masks, outputs, output_masks, pair_mask = [], [], [], [], []

        for i in range(self.max_pairs):
            if i < len(train_examples):
                example = train_examples[i]
                inp_padded, inp_m = self._pad_grid(example.get("input", []))
                out_padded, out_m = self._pad_grid(example.get("output", []))
                inputs.append(inp_padded)
                input_masks.append(inp_m)
                outputs.append(out_padded)
                output_masks.append(out_m)
                pair_mask.append(True)
            else:
                # Padding slot
                inp_padded, inp_m = self._pad_grid([])
                out_padded, out_m = self._pad_grid([])
                inputs.append(inp_padded)
                input_masks.append(inp_m)
                outputs.append(out_padded)
                output_masks.append(out_m)
                pair_mask.append(False)

        return {
            "input":       torch.stack(inputs),
            "input_mask":  torch.stack(input_masks),
            "output":      torch.stack(outputs),
            "output_mask": torch.stack(output_masks),
            "pair_mask":   torch.tensor(pair_mask, dtype=torch.bool),
        }

    def _pad_grid(self, grid):
        """
        Pads a 2D grid to (max_size, max_size) using 10 as the pad token.
        Returns:
            padded_grid: LongTensor of shape (max_size, max_size)
            mask:        BoolTensor of shape (max_size, max_size), True for valid cells
        """
        grid = np.array(grid)

        if grid.size == 0:
            return (
                torch.full((self.max_size, self.max_size), 10, dtype=torch.int64),
                torch.zeros((self.max_size, self.max_size), dtype=torch.bool),
            )

        if grid.ndim == 1:
            grid = grid.reshape(1, -1)

        h, w = grid.shape
        h = min(h, self.max_size)
        w = min(w, self.max_size)
        grid = grid[:h, :w]

        padded = np.full((self.max_size, self.max_size), 10, dtype=np.int64)
        mask   = np.zeros((self.max_size, self.max_size), dtype=np.bool_)

        padded[:h, :w] = grid
        mask[:h, :w]   = True

        return torch.tensor(padded), torch.tensor(mask)
