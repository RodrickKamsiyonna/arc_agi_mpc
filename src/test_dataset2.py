from dataset import ARCDataset
from torch.utils.data import DataLoader
import os
import json

# Setup dummy directory
os.makedirs("/app/src/dummy_dir", exist_ok=True)
with open("/app/src/dummy_dir/1.json", "w") as f:
    json.dump({"train": [{"input": [[1,1]], "output": [[1,1]]}]}, f)
with open("/app/src/dummy_dir/2.json", "w") as f:
    json.dump({"train": [{"input": [[2,2], [2,2]], "output": [[2,2], [2,2]]}, {"input": [[3]], "output": [[3]]}]}, f)

ds = ARCDataset("/app/src/dummy_dir", max_pairs=3)
loader = DataLoader(ds, batch_size=2) # we can now batch multiple tasks together!

for batch in loader:
    print("inputs shape:", batch["input"].shape) # should be (batch_size, max_pairs, max_size, max_size)
    print("pair_mask shape:", batch["pair_mask"].shape) # should be (batch_size, max_pairs)
    print("pair_mask values:\n", batch["pair_mask"])
    break
print("Directory reading and fixed batching works!")
