import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import time
import glob

from dataset import ARCDataset
from model import JEPARC
from loss import JEPARCLoss


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, step, checkpoint_dir, keep_last=3):
    """Atomic save: write to .tmp then rename, so a crash never corrupts the file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
    tmp_path = checkpoint_path + ".tmp"

    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    state = {
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(state, tmp_path)
    os.replace(tmp_path, checkpoint_path)   # atomic on Linux/macOS
    print(f"Saved checkpoint → {checkpoint_path}")

    # Remove old checkpoints, keeping the most recent `keep_last`
    all_ckpts = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt")),
        key=lambda p: int(p.rsplit("_", 1)[-1].replace(".pt", ""))
    )
    for old in all_ckpts[:-keep_last]:
        os.remove(old)
        print(f"  Removed old checkpoint: {old}")


def find_latest_valid_checkpoint(checkpoint_dir):
    """Return the path of the most recent non-corrupt checkpoint, or None."""
    pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
    candidates = sorted(
        glob.glob(pattern),
        key=lambda p: int(p.rsplit("_", 1)[-1].replace(".pt", "")),
        reverse=True,
    )
    for path in candidates:
        try:
            torch.load(path, map_location='cpu', weights_only=True)
            return path
        except Exception as e:
            print(f"Skipping corrupt checkpoint {path}: {e}")
    return None


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path!r}, starting from scratch.")
        return 0

    print(f"Resuming from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    model_to_load = model.module if isinstance(model, nn.DataParallel) else model
    model_to_load.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in state:
        scheduler.load_state_dict(state['scheduler_state_dict'])

    return state.get('step', 0)


# ── Infinite dataloader ───────────────────────────────────────────────────────

def infinite_loader(dataloader):
    """Yield batches forever, restarting the dataloader each epoch."""
    while True:
        yield from dataloader


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    parser = argparse.ArgumentParser(description="Train JEPARC Model")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint, or 'auto' to find the latest valid one.")

    # Model config
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_actions", type=int, default=6)
    parser.add_argument("--action_dim", type=int, default=128)
    parser.add_argument("--max_size", type=int, default=30)
    parser.add_argument("--max_pairs", type=int, default=5)

    # Training config
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--keep_checkpoints", type=int, default=3)

    # Loss config
    parser.add_argument("--lambda_pred", type=float, default=1.0)
    parser.add_argument("--lambda_kl", type=float, default=0.1)
    parser.add_argument("--lambda_consist", type=float, default=1.0)
    parser.add_argument("--lambda_sigreg", type=float, default=0.09)

    args = parser.parse_args()

    # ── Device setup ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"Using device: {device} | Total GPUs: {num_gpus} | AMP dtype: {amp_dtype if use_amp else 'disabled'}")

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = ARCDataset(data_path=args.data_path, max_size=args.max_size, max_pairs=args.max_pairs)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_amp,
    )
    print(f"Number of Files:\n{len(dataset)}")
    print(f"Loaded {len(dataset)} tasks from {args.data_path}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = JEPARC(
        hidden_dim=args.hidden_dim,
        max_size=args.max_size,
        num_actions=args.num_actions,
        action_dim=args.action_dim,
    ).to(device)

    if num_gpus > 1:
        print(f"Activating DataParallel on {num_gpus} GPUs")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = JEPARCLoss(
        lambda_pred=args.lambda_pred,
        lambda_kl=args.lambda_kl,
        lambda_consist=args.lambda_consist,
        lambda_sigreg=args.lambda_sigreg,
    ).to(device)

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_path = args.resume
    if resume_path == 'auto':
        resume_path = find_latest_valid_checkpoint(args.checkpoint_dir)
        if resume_path:
            print(f"Auto-selected checkpoint: {resume_path}")
        else:
            print("No valid checkpoint found; starting from scratch.")

    start_step = load_checkpoint(resume_path, model, optimizer, scheduler=None)

    # ── AMP scaler ────────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler('cuda' if use_amp else 'cpu', enabled=use_amp)

    # ── Loop ──────────────────────────────────────────────────────────────────
    model.train()
    step = start_step
    optimizer.zero_grad()

    print("Starting training...")
    step_start = time.time()

    for batch in infinite_loader(dataloader):
        if step >= args.max_steps:
            break

        input_grid  = batch["input"].to(device)
        input_mask  = batch["input_mask"].to(device)
        output_grid = batch["output"].to(device)
        output_mask = batch["output_mask"].to(device)
        pair_mask   = batch["pair_mask"].to(device)

        with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
            z_in, z_out, z_pred, mus, logvars, actions = model(
                input_grid, input_mask, output_grid, output_mask
            )
            loss, metrics = criterion(z_out, z_pred, mus, logvars, actions, pair_mask)
            # Ensure scalar (DataParallel can return per-device tensors)
            total_loss = loss.mean() if loss.dim() > 0 else loss
            loss_to_backward = total_loss / args.grad_accum_steps

        scaler.scale(loss_to_backward).backward()

        if (step + 1) % args.grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % args.log_every == 0:
            elapsed = time.time() - step_start

            def scalar(v):
                return v.mean().item() if isinstance(v, torch.Tensor) else float(v)

            print(
                f"Step {step:>7} | "
                f"Loss: {scalar(total_loss):.4f} | "          # log the actual backprop'd loss
                f"Pred: {scalar(metrics['loss_pred']):.4f} | "
                f"KL: {scalar(metrics['loss_kl']):.4f} | "
                f"Consist: {scalar(metrics['loss_consist']):.4f} | "
                f"SigReg: {scalar(metrics['loss_sigreg']):.4f} | "
                f"Time: {elapsed:.2f}s"
            )
            step_start = time.time()

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step > 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, None, step,
                            args.checkpoint_dir, keep_last=args.keep_checkpoints)

        step += 1

    save_checkpoint(model, optimizer, None, step,
                    args.checkpoint_dir, keep_last=args.keep_checkpoints)
    print("Training complete!")


if __name__ == "__main__":
    train()
