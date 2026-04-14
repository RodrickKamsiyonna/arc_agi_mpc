import torch
from loss import JEPARCLoss

batch_size = 2
num_pairs = 3
hidden_dim = 256
num_actions = 4
action_dim = 64

z_out = torch.randn(batch_size, num_pairs, hidden_dim)
z_pred = torch.randn(batch_size, num_pairs, hidden_dim)
mus = torch.randn(batch_size, num_pairs, num_actions, action_dim)
logvars = torch.randn(batch_size, num_pairs, num_actions, action_dim)
actions = torch.randn(batch_size, num_pairs, num_actions, action_dim)
pair_mask = torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool)

criterion = JEPARCLoss()
total_loss, metrics = criterion(z_out, z_pred, mus, logvars, actions, pair_mask)

print("Total loss:", total_loss.item())
print("Metrics:", metrics)
