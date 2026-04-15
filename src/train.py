import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import time

from dataset import ARCDataset
from model import JEPARC
from loss import JEPARCLoss

def save_checkpoint(model, optimizer, scheduler, step, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")

    # If using DataParallel, we must save model.module.state_dict() 
    # to avoid 'module.' prefixes when loading later without DP.
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model

    state = {
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scheduler:
        state['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(state, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return 0

    print(f"Resuming from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    # Load into the underlying model if DataParallel is used
    model_to_load = model.module if isinstance(model, nn.DataParallel) else model
    model_to_load.load_state_dict(state['model_state_dict'])
    
    optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in state:
        scheduler.load_state_dict(state['scheduler_state_dict'])

    return state.get('step', 0)

def train():
    parser = argparse.ArgumentParser(description="Train JEPARC Model (Multi-GPU)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to ARC JSON data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    # Model config
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_actions", type=int, default=4)
    parser.add_argument("--action_dim", type=int, default=128)
    parser.add_argument("--max_size", type=int, default=30)
    parser.add_argument("--max_pairs", type=int, default=5)

    # Training config
    parser.add_argument("--batch_size", type=int, default=6, help="Total batch size across all GPUs")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--grad_accum_steps", type=int, default=4)

    # Loss config
    parser.add_argument("--lambda_pred", type=float, default=1.0)
    parser.add_argument("--lambda_kl", type=float, default=0.1)
    parser.add_argument("--lambda_consist", type=float, default=1.0)
    parser.add_argument("--lambda_sigreg", type=float, default=0.09)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device} | Total GPUs: {num_gpus}")

    # Dataset and DataLoader
    dataset = ARCDataset(data_path=args.data_path, max_size=args.max_size, max_pairs=args.max_pairs)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2, # Scale workers with GPUs
        pin_memory=True
    )

    # Initialize Model
    model = JEPARC(
        hidden_dim=args.hidden_dim,
        max_size=args.max_size,
        num_actions=args.num_actions,
        action_dim=args.action_dim
    ).to(device)

    # Wrap with DataParallel if more than 1 GPU
    if num_gpus > 1:
        print(f"Activating DataParallel on {num_gpus} GPUs")
        model = nn.DataParallel(model)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Loss (Note: criterion usually stays on the main GPU or is used inside model forward)
    criterion = JEPARCLoss(
        lambda_pred=args.lambda_pred,
        lambda_kl=args.lambda_kl,
        lambda_consist=args.lambda_consist,
        lambda_sigreg=args.lambda_sigreg
    ).to(device)

    # Resume logic
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, None)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available())

    model.train()
    step = start_step
    optimizer.zero_grad()

    print("Starting training...")
    start_time = time.time()

    while step < args.max_steps:
        for batch in dataloader:
            if step >= args.max_steps:
                break

            input_grid = batch["input"].to(device)
            input_mask = batch["input_mask"].to(device)
            output_grid = batch["output"].to(device)
            output_mask = batch["output_mask"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            
            # Autocast handles the mixed precision
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', 
                                    enabled=torch.cuda.is_available(), 
                                    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                
                # Model forward pass (distributed across GPUs)
                z_in, z_out, z_pred, mus, logvars, actions = model(input_grid, input_mask, output_grid, output_mask)
                
                # Loss calculation
                loss, metrics = criterion(z_out, z_pred, mus, logvars, actions, pair_mask)
                
                # If using DataParallel, metrics will be returned as tensors with length = num_gpus
                # We need the mean for logging
                total_loss = loss.mean()
                loss_to_backward = total_loss / args.grad_accum_steps

            scaler.scale(loss_to_backward).backward()

            if (step + 1) % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                # Extract mean values from metrics for logging
                log_str = f"Step {step} | Loss: {metrics['total_loss'].mean().item():.4f} | " \
                          f"Pred: {metrics['loss_pred'].mean().item():.4f} | " \
                          f"KL: {metrics['loss_kl'].mean().item():.4f} | " \
                          f"Time: {elapsed:.2f}s"
                print(log_str)
                start_time = time.time()

            if step > 0 and step % args.save_every == 0:
                save_checkpoint(model, optimizer, None, step, args.checkpoint_dir)

            step += 1

    save_checkpoint(model, optimizer, None, step, args.checkpoint_dir)
    print("Training complete!")

if __name__ == "__main__":
    train()
