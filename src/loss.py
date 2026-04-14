import torch
import torch.nn as nn
import torch.nn.functional as F
import lejepa

class JEPARCLoss(nn.Module):
    def __init__(self, lambda_pred=1.0, lambda_kl=0.1, lambda_consist=1.0, lambda_sigreg=0.1):
        super().__init__()
        self.lambda_pred = lambda_pred
        self.lambda_kl = lambda_kl
        self.lambda_consist = lambda_consist
        self.lambda_sigreg = lambda_sigreg

        # LeJEPA SigReg setup
        univariate_test = lejepa.univariate.EppsPulley(n_points=17)
        self.sigreg_loss_fn = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=1024,
            reduction='mean'
        )

    def forward(self, z_out, z_pred, mus, logvars, actions, pair_mask):
        """
        Calculates the total loss for a batch of tasks, ignoring padded pairs.

        z_out: (batch, num_pairs, hidden_dim)
        z_pred: (batch, num_pairs, hidden_dim)
        mus: (batch, num_pairs, num_actions, action_dim)
        logvars: (batch, num_pairs, num_actions, action_dim)
        actions: (batch, num_pairs, num_actions, action_dim)
        pair_mask: (batch, num_pairs) boolean mask, True for valid pairs
        """
        valid_z_out = z_out[pair_mask]
        valid_z_pred = z_pred[pair_mask]
        valid_mus = mus[pair_mask]
        valid_logvars = logvars[pair_mask]

        # 1. Prediction Loss (MSE between predicted and true latent)
        if valid_z_pred.shape[0] > 0:
            loss_pred = F.mse_loss(valid_z_pred, valid_z_out)
        else:
            loss_pred = torch.tensor(0.0, device=z_out.device)

        # 2. KL Divergence Loss (N(mu, sigma) || N(0, I))
        if valid_mus.shape[0] > 0:
            kl_div = -0.5 * torch.sum(1 + valid_logvars - valid_mus.pow(2) - valid_logvars.exp(), dim=-1)
            loss_kl = kl_div.mean()
        else:
            loss_kl = torch.tensor(0.0, device=z_out.device)

        # 3. Example Consistency Loss
        # Compute variance of actions across the examples for each task individually, then average
        loss_consist = torch.tensor(0.0, device=actions.device)
        consist_count = 0
        b_sz = actions.shape[0]

        for i in range(b_sz):
            task_mask = pair_mask[i]
            task_actions = actions[i][task_mask] # (valid_pairs, num_actions, action_dim)
            if task_actions.shape[0] > 1:
                loss_consist += task_actions.var(dim=0).mean()
                consist_count += 1

        if consist_count > 0:
            loss_consist = loss_consist / consist_count

        # 4. SigReg Regularization
        # Apply to the target latent embeddings (z_out) to prevent collapse
        if valid_z_out.shape[0] > 0:
            loss_sigreg = self.sigreg_loss_fn(valid_z_out)
        else:
            loss_sigreg = torch.tensor(0.0, device=z_out.device)

        # Total Loss
        total_loss = (self.lambda_pred * loss_pred +
                      self.lambda_kl * loss_kl +
                      self.lambda_consist * loss_consist +
                      self.lambda_sigreg * loss_sigreg)

        return total_loss, {
            "loss_pred": loss_pred.item(),
            "loss_kl": loss_kl.item(),
            "loss_consist": loss_consist.item(),
            "loss_sigreg": loss_sigreg.item(),
            "total_loss": total_loss.item()
        }
