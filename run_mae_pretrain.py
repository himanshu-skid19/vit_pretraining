"""
Quick start script for MAE pretraining with wandb logging.

Run this script to start MAE pretraining with sensible defaults.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mae import MAEBase, MAESmall
from dataset import create_dataloaders
from train_mae import (
    train_one_epoch,
    validate,
    visualize_reconstruction,
    save_checkpoint,
    get_cosine_schedule_with_warmup,
    get_mask_ratio
)

from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Install with: pip install wandb")


def main():
    # ===================== CONFIGURATION =====================
    # Modify these parameters as needed

    DATA_PATH = r"C:\Users\Himanshu Singhal\Desktop\BTP\working\deepsigndb_asymmetric_gasf.npz"
    OUTPUT_DIR = r"C:\Users\Himanshu Singhal\Desktop\BTP\working\mae_pretrain"

    # Model configuration
    MODEL = "mae_small"  # Optimized for GASF (13M params), "mae_base" for larger model (99M params)
    IMG_SIZE = 256
    PATCH_SIZE = 32
    IN_CHANNELS = 1
    MASK_RATIO = 0.75

    # Training configuration
    EPOCHS = 100
    BATCH_SIZE = 32  # Reduce if you run out of GPU memory
    LEARNING_RATE = 1.5e-4
    WARMUP_EPOCHS = 10
    WEIGHT_DECAY = 0.05

    # Masking curriculum
    MASK_CURRICULUM = True        # Set to False for constant mask ratio
    MASK_CURRICULUM_START = 0.75  # Hold at this ratio during warmup
    MASK_CURRICULUM_END = 0.95    # Ramp up to this ratio
    MASK_CURRICULUM_WARMUP = 30   # Epochs to hold before ramping

    # System
    NUM_WORKERS = 4
    USE_AMP = True  # Mixed precision training
    SEED = 42

    # Logging
    VIS_FREQ = 10  # Visualize reconstruction every N epochs
    SAVE_FREQ = 20  # Save checkpoint every N epochs
    LOG_FREQ = 50   # Log metrics every N batches

    # Wandb configuration
    USE_WANDB = True  # Set to False to disable wandb
    WANDB_PROJECT = "mae-gasf-pretrain"
    WANDB_ENTITY = None  # Your wandb username or team name
    WANDB_TAGS = ["mae", "pretrain", "gasf", "deepsigndb"]

    # =========================================================

    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(OUTPUT_DIR, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MAE Pretraining for GASF Images")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Mask ratio: {MASK_RATIO}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Output: {output_dir}")

    # Initialize wandb
    wandb_run = None
    if USE_WANDB and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f'{MODEL}_{timestamp}',
            tags=WANDB_TAGS,
            config={
                'model': MODEL,
                'img_size': IMG_SIZE,
                'patch_size': PATCH_SIZE,
                'in_channels': IN_CHANNELS,
                'mask_ratio': MASK_RATIO,
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'warmup_epochs': WARMUP_EPOCHS,
                'weight_decay': WEIGHT_DECAY,
                'seed': SEED,
                'amp': USE_AMP
            }
        )
        print(f"\nWandb initialized: {wandb_run.url}")
    elif USE_WANDB and not WANDB_AVAILABLE:
        print("\nWandb requested but not installed. Continuing without wandb.")

    # Set seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        DATA_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_ratio=0.9,
        seed=SEED,
        genuine_only=True
    )

    # Log dataset info to wandb
    if wandb_run is not None:
        wandb_run.config.update({
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset)
        })

    # Create model
    print("\nCreating model...")
    if MODEL == "mae_small":
        model = MAESmall(
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            in_channels=IN_CHANNELS,
            mask_ratio=MASK_RATIO
        )
    else:
        model = MAEBase(
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            in_channels=IN_CHANNELS,
            mask_ratio=MASK_RATIO
        )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Watch model with wandb
    if wandb_run is not None:
        wandb_run.config.update({'total_params': total_params})
        wandb.watch(model, log='gradients', log_freq=LOG_FREQ * 10)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95)
    )

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=EPOCHS
    )

    # Mixed precision
    scaler = GradScaler() if USE_AMP else None

    # TensorBoard
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))

    # Get visualization batch
    vis_batch = next(iter(val_loader))

    # Training args (for compatibility with train_mae functions)
    class Args:
        amp = USE_AMP
        accum_steps = 1
        clip_grad = 1.0
        log_freq = LOG_FREQ
        epochs = EPOCHS
        batch_size = BATCH_SIZE
        mask_ratio = MASK_RATIO
        mask_curriculum = MASK_CURRICULUM
        mask_curriculum_start = MASK_CURRICULUM_START
        mask_curriculum_end = MASK_CURRICULUM_END
        mask_curriculum_warmup = MASK_CURRICULUM_WARMUP

    args = Args()

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    if MASK_CURRICULUM:
        print(f"Mask curriculum: {MASK_CURRICULUM_START:.0%} for {MASK_CURRICULUM_WARMUP} epochs, "
              f"then ramp to {MASK_CURRICULUM_END:.0%}")
    print("=" * 60)

    best_loss = float('inf')

    for epoch in range(EPOCHS):
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

        print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Mask Ratio: {current_mask_ratio:.3f}")

        # Log epoch metrics to wandb
        if wandb_run is not None:
            wandb_run.log({
                'train/loss_epoch': train_loss,
                'val/loss_epoch': val_loss,
                'train/mask_ratio': current_mask_ratio,
                'epoch': epoch
            })

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, val_loss, args,
                os.path.join(output_dir, 'best_model.pth'),
                wandb_run
            )

            if wandb_run is not None:
                wandb_run.summary['best_val_loss'] = best_loss
                wandb_run.summary['best_epoch'] = epoch

        # Periodic checkpoint (overwrite previous to save disk space)
        if (epoch + 1) % SAVE_FREQ == 0:
            ckpt_path = os.path.join(output_dir, 'latest_checkpoint.pth')
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, val_loss, args,
                ckpt_path
            )

        # Visualization
        if (epoch + 1) % VIS_FREQ == 0:
            visualize_reconstruction(model, vis_batch, epoch, output_dir, device, wandb_run)

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, scaler, EPOCHS - 1, val_loss, args,
        os.path.join(output_dir, 'final_model.pth')
    )

    # Save encoder weights for fine-tuning
    encoder_state = model.get_encoder_state_dict()
    encoder_path = os.path.join(output_dir, 'encoder_weights.pth')
    torch.save(encoder_state, encoder_path)

    writer.close()

    # Finish wandb run
    if wandb_run is not None:
        wandb_run.finish()

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Encoder weights saved to: {encoder_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
