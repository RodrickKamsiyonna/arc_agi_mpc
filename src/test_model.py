import torch
from model import JEPARC

batch_size = 2
num_pairs = 3
max_size = 30
model = JEPARC(hidden_dim=256, num_actions=4, action_dim=64)

input_grid = torch.randint(0, 10, (batch_size, num_pairs, max_size, max_size))
input_mask = torch.ones((batch_size, num_pairs, max_size, max_size), dtype=torch.bool)
output_grid = torch.randint(0, 10, (batch_size, num_pairs, max_size, max_size))
output_mask = torch.ones((batch_size, num_pairs, max_size, max_size), dtype=torch.bool)

z_in, z_out, z_pred, mus, logvars, actions = model(input_grid, input_mask, output_grid, output_mask)

print("z_in shape:", z_in.shape)
print("z_out shape:", z_out.shape)
print("z_pred shape:", z_pred.shape)
print("mus shape:", mus.shape)
print("actions shape:", actions.shape)
