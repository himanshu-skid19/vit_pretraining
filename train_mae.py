"""
MAE Pretraining Script for Vision Transformer on GASF Images.

Usage:
    python train_mae.py --data_path path/to/data.npz --epochs 200

Features:
- Cosine learning rate schedule with warmup
- Gradient accumulation for larger effective batch sizes
- Automatic mixed precision training
- Checkpointing and resumption
- TensorBoard logging
- Weights & Biases (wandb) logging
- Visualization of reconstructions
"""

import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from mae import MAE, MAEBase, MAESmall
from dataset import create_dataloaders

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Install with: pip install wandb")


def parse_args():
    parser = argparse.ArgumentParser(description='MAE Pretraining for ViT')

    # Data
    parser.add_argument('--data_path', type=str,
                        default=r'/teamspace/studios/this_studio/deepsigndb_asymmetric_gasf.npz',
                        help='Path to .npz dataset file')
    parser.add_argument('--output_dir', type=str, default='./mae_pretrain',
                        help='Output directory for checkpoints and logs')

    # Model
    parser.add_argument('--model', type=str, default='mae_base',
                        choices=['mae_small', 'mae_base', 'mae_custom'],
                        help='Model variant to use')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Patch size')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Number of input channels')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='Masking ratio for MAE')

    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size per GPU')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1.5e-4,
                        help='Base learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping norm')

    # Masking curriculum
    parser.add_argument('--mask_curriculum', action='store_true', default=False,
                        help='Enable masking curriculum (ramp mask_ratio over training)')
    parser.add_argument('--mask_curriculum_start', type=float, default=0.75,
                        help='Starting mask ratio for curriculum')
    parser.add_argument('--mask_curriculum_end', type=float, default=0.90,
                        help='Final mask ratio for curriculum')
    parser.add_argument('--mask_curriculum_warmup', type=int, default=30,
                        help='Epochs to hold at starting mask ratio before ramping')

    # System
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Logging
    parser.add_argument('--log_freq', type=int, default=100,
                        help='Logging frequency (iterations)')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='Checkpoint save frequency (epochs)')
    parser.add_argument('--vis_freq', type=int, default=1,
                        help='Visualization frequency (epochs)')

    # Wandb
    parser.add_argument('--wandb', action='store_true', default=True,
                        help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='mae-gasf-pretrain',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity (username or team)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (auto-generated if not specified)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=['mae', 'pretrain', 'gasf'],
                        help='Wandb tags')
    parser.add_argument('--wandb_offline', action='store_true', default=False,
                        help='Run wandb in offline mode')

    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """
    Create cosine learning rate schedule with linear warmup.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return epoch / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_mask_ratio(epoch, args):
    """
    Get mask ratio for the current epoch based on curriculum schedule.

    Schedule:
      - Epochs [0, warmup): hold at mask_curriculum_start (default 0.75)
      - Epochs [warmup, total): linearly ramp from start to end (default 0.75 -> 0.90)

    If mask_curriculum is disabled, returns args.mask_ratio (constant).
    """
    if not args.mask_curriculum:
        return args.mask_ratio

    if epoch < args.mask_curriculum_warmup:
        return args.mask_curriculum_start

    ramp_epochs = args.epochs - args.mask_curriculum_warmup
    if ramp_epochs <= 0:
        return args.mask_curriculum_end

    progress = (epoch - args.mask_curriculum_warmup) / ramp_epochs
    progress = min(progress, 1.0)
    return args.mask_curriculum_start + progress * (args.mask_curriculum_end - args.mask_curriculum_start)


def create_model(args):
    """Create MAE model based on args."""
    if args.model == 'mae_small':
        model = MAESmall(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=args.in_channels,
            mask_ratio=args.mask_ratio
        )
    elif args.model == 'mae_base':
        model = MAEBase(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=args.in_channels,
            mask_ratio=args.mask_ratio
        )
    else:
        # Custom configuration
        model = MAE(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=args.in_channels,
            mask_ratio=args.mask_ratio
        )

    return model


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, args, path, wandb_run=None):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
        'args': vars(args)
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint['epoch'], checkpoint.get('loss', float('inf'))


def visualize_reconstruction(model, images, epoch, output_dir, device, wandb_run=None):
    """
    Visualize MAE reconstruction.

    Creates a figure showing:
    - Original images
    - Masked inputs (visible patches only)
    - Reconstructed images
    """
    model.eval()

    with torch.no_grad():
        images = images[:8].to(device)  # Take first 8 samples
        orig, recon, masked, mask = model.reconstruct(images)

    # Move to CPU and convert to numpy
    orig = orig.cpu().numpy()
    recon = recon.cpu().numpy()
    masked = masked.cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))

    for i in range(8):
        # Original
        axes[0, i].imshow(orig[i, 0], cmap='viridis', vmin=-1, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12)

        # Masked
        axes[1, i].imshow(masked[i, 0], cmap='viridis', vmin=-1, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Masked (25% visible)', fontsize=12)

        # Reconstructed
        axes[2, i].imshow(recon[i, 0], cmap='viridis', vmin=-1, vmax=1)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Reconstructed', fontsize=12)

    plt.tight_layout()

    # Save figure locally
    vis_path = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_path, exist_ok=True)
    fig_path = os.path.join(vis_path, f'reconstruction_epoch_{epoch:04d}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')

    # Log to wandb
    if wandb_run is not None:
        wandb_run.log({
            'reconstructions': wandb.Image(fig, caption=f'Epoch {epoch}'),
            'epoch': epoch
        })

    plt.close()

    model.train()

    return fig_path


def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, args, device, writer, wandb_run=None, mask_ratio=None):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    num_batches = 0

    start_time = time.time()

    for batch_idx, images in enumerate(train_loader):
        images = images.to(device, non_blocking=True)

        # Forward pass with mixed precision
        if args.amp:
            with autocast():
                loss, pred, mask = model(images, mask_ratio=mask_ratio)
                loss = loss / args.accum_steps
        else:
            loss, pred, mask = model(images, mask_ratio=mask_ratio)
            loss = loss / args.accum_steps

        # Backward pass
        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % args.accum_steps == 0:
            if args.amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

            optimizer.zero_grad()

        total_loss += loss.item() * args.accum_steps
        num_batches += 1

        # Logging
        if batch_idx % args.log_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * args.batch_size / elapsed

            print(f"Epoch [{epoch}/{args.epochs}] "
                  f"Batch [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item() * args.accum_steps:.4f} "
                  f"LR: {current_lr:.2e} "
                  f"Speed: {samples_per_sec:.1f} samples/s")

            global_step = epoch * len(train_loader) + batch_idx

            # TensorBoard logging
            writer.add_scalar('train/loss', loss.item() * args.accum_steps, global_step)
            writer.add_scalar('train/lr', current_lr, global_step)

            # Wandb logging
            if wandb_run is not None:
                wandb_run.log({
                    'train/loss_step': loss.item() * args.accum_steps,
                    'train/lr': current_lr,
                    'train/samples_per_sec': samples_per_sec,
                    'global_step': global_step
                })

    avg_loss = total_loss / num_batches

    return avg_loss


@torch.no_grad()
def validate(model, val_loader, epoch, args, device, writer, wandb_run=None):
    """Validate model."""
    model.eval()

    total_loss = 0
    num_batches = 0

    for images in val_loader:
        images = images.to(device, non_blocking=True)

        if args.amp:
            with autocast():
                loss, pred, mask = model(images)
        else:
            loss, pred, mask = model(images)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    # TensorBoard logging
    writer.add_scalar('val/loss', avg_loss, epoch)

    # Wandb logging
    if wandb_run is not None:
        wandb_run.log({
            'val/loss': avg_loss,
            'epoch': epoch
        })

    return avg_loss


def init_wandb(args):
    """Initialize wandb run."""
    if not WANDB_AVAILABLE:
        print("wandb not available, skipping wandb initialization")
        return None

    if not args.wandb:
        print("wandb logging disabled")
        return None

    # Set offline mode if requested
    if args.wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'

    # Generate run name if not provided
    if args.wandb_run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.wandb_run_name = f'{args.model}_{timestamp}'

    # Initialize wandb
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        config=vars(args),
        reinit=True
    )

    print(f"Wandb initialized: {run.url}")

    return run


def main():
    args = parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print("MAE Pretraining for Vision Transformer")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")

    # Initialize wandb
    wandb_run = init_wandb(args)

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=0.9,
        seed=args.seed,
        genuine_only=True
    )

    # Log dataset info to wandb
    if wandb_run is not None:
        wandb_run.config.update({
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'num_batches_per_epoch': len(train_loader)
        })

    # Create model
    print("\nCreating model...")
    model = create_model(args)
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Log model info to wandb
    if wandb_run is not None:
        wandb_run.config.update({
            'total_params': total_params,
            'trainable_params': trainable_params
        })
        # Watch model gradients and parameters
        wandb.watch(model, log='gradients', log_freq=1000)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=args.min_lr
    )

    # Mixed precision scaler
    scaler = GradScaler() if args.amp else None

    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')

    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch, best_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        start_epoch += 1
        print(f"Resuming from epoch {start_epoch}")

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    # Get a fixed batch for visualization
    vis_images = next(iter(val_loader))

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Compute mask ratio for this epoch (curriculum or constant)
        current_mask_ratio = get_mask_ratio(epoch, args)

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            epoch, args, device, writer, wandb_run, mask_ratio=current_mask_ratio
        )

        # Validate
        val_loss = validate(model, val_loader, epoch, args, device, writer, wandb_run)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Mask Ratio: {current_mask_ratio:.3f}")

        # Log epoch metrics to wandb
        if wandb_run is not None:
            wandb_run.log({
                'train/loss_epoch': train_loss,
                'val/loss_epoch': val_loss,
                'train/mask_ratio': current_mask_ratio,
                'epoch': epoch,
                'epoch_time': epoch_time
            })

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, val_loss, args,
                os.path.join(args.output_dir, 'best_model.pth'),
                wandb_run
            )

            # Log best loss to wandb
            if wandb_run is not None:
                wandb_run.summary['best_val_loss'] = best_loss
                wandb_run.summary['best_epoch'] = epoch

        # Save periodic checkpoint (overwrite previous to save disk space)
        if (epoch + 1) % args.save_freq == 0:
            ckpt_path = os.path.join(args.output_dir, 'latest_checkpoint.pth')
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, val_loss, args,
                ckpt_path
            )

        # Visualization
        if (epoch + 1) % args.vis_freq == 0:
            visualize_reconstruction(model, vis_images, epoch, args.output_dir, device, wandb_run)

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, scaler, args.epochs - 1, val_loss, args,
        os.path.join(args.output_dir, 'final_model.pth')
    )

    # Save encoder weights for fine-tuning
    encoder_state = model.get_encoder_state_dict()
    encoder_path = os.path.join(args.output_dir, 'encoder_weights.pth')
    torch.save(encoder_state, encoder_path)
    print(f"\nEncoder weights saved to {encoder_path}")

    writer.close()

    # Finish wandb run
    if wandb_run is not None:
        wandb_run.finish()

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
