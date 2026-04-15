import torch
import torch.nn as nn
import torch.nn.functional as F
import lejepa


class JEPARCLoss(nn.Module):
    def __init__(
        self,
        lambda_pred=1.0,
        lambda_kl=0.1,
        lambda_consist=1.0,
        lambda_sigreg=0.09,
    ):
        super().__init__()
        self.lambda_pred = lambda_pred
        self.lambda_kl = lambda_kl
        self.lambda_consist = lambda_consist
        self.lambda_sigreg = lambda_sigreg

        # SigReg setup (LeJEPA)
        univariate_test = lejepa.univariate.EppsPulley(n_points=17)
        self.sigreg_loss_fn = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=1024,
            reduction="mean",
        )

    def forward(self, z_out, z_pred, mus, logvars, actions, pair_mask):
        """
        z_out:     (B, P, D)
        z_pred:    (B, P, D)
        mus:       (B, P, N, A)
        logvars:   (B, P, N, A)
        actions:   (B, P, N, A)
        pair_mask: (B, P) boolean
        """

        device = z_out.device

        # =========================
        # Mask valid pairs
        # =========================
        valid_z_out = z_out[pair_mask]
        valid_z_pred = z_pred[pair_mask]
        valid_mus = mus[pair_mask]
        valid_logvars = logvars[pair_mask]
        valid_actions = actions[pair_mask]

        # =========================
        # 1. Prediction Loss
        # =========================
        if valid_z_pred.numel() > 0:
            loss_pred = F.mse_loss(valid_z_pred, valid_z_out)
        else:
            loss_pred = torch.zeros(1, device=device, requires_grad=True).squeeze()

        # =========================
        # 2. KL Divergence Loss
        # =========================
        if valid_mus.numel() > 0:
            # KL per dimension
            kl = -0.5 * (
                1 + valid_logvars - valid_mus.pow(2) - valid_logvars.exp()
            )

            # Sum over action_dim
            kl = kl.sum(dim=-1)            # (valid_pairs, N)

            # Mean over actions
            kl = kl.mean(dim=-1)           # (valid_pairs)

            # Mean over batch
            loss_kl = kl.mean()
        else:
            loss_kl = torch.zeros(1, device=device, requires_grad=True).squeeze()

        # =========================
        # 3. Consistency Loss (pairwise MSE - Fixed)
        # =========================
        loss_consist = torch.zeros(1, device=device, requires_grad=True).squeeze()
        consist_count = 0

        B = actions.shape[0]

        for i in range(B):
            task_mask = pair_mask[i]
            task_actions = actions[i][task_mask]  # (num_valid_pairs, N, A)

            V = task_actions.shape[0] # Number of valid pairs
            
            if V > 1:
                # Get indices for unique pairs, excluding self-comparisons (offset=1)
                row_idx, col_idx = torch.triu_indices(V, V, offset=1)

                # Calculate differences ONLY for those unique pairs
                diffs = task_actions[row_idx] - task_actions[col_idx]  # (unique_pairs, N, A)

                loss_consist = loss_consist + diffs.pow(2).mean()
                consist_count += 1

        if consist_count > 0:
            loss_consist = loss_consist / consist_count
        else:
            loss_consist = torch.zeros(1, device=device, requires_grad=True).squeeze()
            
        # =========================
        # 4. SigReg Regularization
        # =========================
        if valid_z_pred.numel() > 0:
            loss_sigreg = self.sigreg_loss_fn(valid_z_pred)
        else:
            loss_sigreg = torch.zeros(1, device=device, requires_grad=True).squeeze()

        # =========================
        # Total Loss
        # =========================
        total_loss = (
            self.lambda_pred * loss_pred
            + self.lambda_kl * loss_kl
            + self.lambda_consist * loss_consist
            + self.lambda_sigreg * loss_sigreg
        )

        return total_loss, {
            "loss_pred": loss_pred.detach().item(),
            "loss_kl": loss_kl.detach().item(),
            "loss_consist": loss_consist.detach().item(),
            "loss_sigreg": loss_sigreg.detach().item(),
            "total_loss": total_loss.detach().item(),
        }
