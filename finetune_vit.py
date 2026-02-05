"""
Fine-tuning script for Vision Transformer after MAE pretraining.

This script loads pretrained encoder weights and fine-tunes for classification
tasks (e.g., signature verification - genuine vs forgery).

Features:
- Layer-wise learning rate decay
- Label smoothing
- Wandb logging
- Comprehensive metrics (accuracy, precision, recall, F1, AUC)
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from vit import ViTClassifier

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Install with: pip install wandb")


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune ViT after MAE pretraining')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to .npz dataset file')
    parser.add_argument('--output_dir', type=str, default='./vit_finetune',
                        help='Output directory')

    # Pretrained weights
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained encoder weights (from MAE)')

    # Model
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (2 for genuine/forgery)')
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--mlp_dim', type=int, default=3072)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # Fine-tuning strategy
    parser.add_argument('--freeze_epochs', type=int, default=5,
                        help='Epochs to freeze encoder before unfreezing')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='Layer-wise learning rate decay')

    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action='store_true', default=True)

    # Wandb
    parser.add_argument('--wandb', action='store_true', default=True,
                        help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='vit-gasf-finetune',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity (username or team)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')
    parser.add_argument('--wandb_tags', type=str, nargs='+',
                        default=['vit', 'finetune', 'gasf', 'classification'],
                        help='Wandb tags')

    return parser.parse_args()


class ClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset for classification with labels.

    Expects .npz file with:
    - 'gasf_data': (N, C, H, W) images
    - 'labels': (N,) labels (0 for genuine, 1 for forgery)
    - 'file_names': (N,) file names
    """

    def __init__(self, npz_path, augment=True):
        data = np.load(npz_path, allow_pickle=True)

        self.images = data['gasf_data'].astype(np.float32)
        self.labels = data.get('labels', None)
        self.file_names = data['file_names']

        # If no labels, infer from file names (genuine/forgery)
        if self.labels is None:
            self.labels = self._infer_labels()

        # Normalize to [-1, 1]
        data_min = self.images.min()
        data_max = self.images.max()
        if data_max - data_min > 1e-6:
            self.images = 2 * (self.images - data_min) / (data_max - data_min) - 1

        self.augment = augment

        print(f"Loaded {len(self.images)} samples")
        print(f"Class distribution: {np.bincount(self.labels.astype(int))}")

    def _infer_labels(self):
        """Infer labels from file names (assumes 'genuine' or 'forgery' in name)."""
        labels = []
        for name in self.file_names:
            name_lower = str(name).lower()
            if 'genuine' in name_lower or 'g_' in name_lower:
                labels.append(0)
            elif 'forgery' in name_lower or 'f_' in name_lower or 'skilled' in name_lower:
                labels.append(1)
            else:
                # Default heuristic: files with numbers 1-20 are genuine, 21+ are forgeries
                # This is dataset-specific and may need adjustment
                labels.append(0)
        return np.array(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        label = self.labels[idx]

        if self.augment:
            image = self._augment(image)

        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)

    def _augment(self, image):
        import random
        if random.random() < 0.5:
            k = random.randint(0, 3)
            image = np.rot90(image, k, axes=(1, 2)).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=2).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
        return image


def get_layer_wise_lr_groups(model, base_lr, layer_decay):
    """
    Create parameter groups with layer-wise learning rate decay.

    Later layers get higher learning rates.
    """
    # Get number of encoder layers
    num_layers = len(model.encoder.blocks)

    # Create parameter groups
    param_groups = []

    # Patch embedding and position embedding - lowest LR
    patch_params = list(model.encoder.patch_embed.parameters())
    patch_params.append(model.encoder.cls_token)
    patch_params.append(model.encoder.pos_embed)

    param_groups.append({
        'params': patch_params,
        'lr': base_lr * (layer_decay ** (num_layers + 1)),
        'name': 'patch_embed'
    })

    # Transformer blocks - increasing LR
    for i, block in enumerate(model.encoder.blocks):
        layer_lr = base_lr * (layer_decay ** (num_layers - i))
        param_groups.append({
            'params': list(block.parameters()),
            'lr': layer_lr,
            'name': f'block_{i}'
        })

    # Final norm and classifier head - highest LR
    param_groups.append({
        'params': list(model.encoder.norm.parameters()) + list(model.classifier.parameters()),
        'lr': base_lr,
        'name': 'head'
    })

    return param_groups


def plot_confusion_matrix(y_true, y_pred, classes=['Genuine', 'Forgery']):
    """Create confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig


def train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch, args, device, wandb_run=None):
    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if args.amp:
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Log batch metrics to wandb
        if wandb_run is not None and batch_idx % 50 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            wandb_run.log({
                'train/loss_step': loss.item(),
                'global_step': global_step
            })

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


@torch.no_grad()
def validate(model, val_loader, criterion, args, device):
    model.eval()

    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if args.amp:
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'preds': all_preds,
        'labels': all_labels
    }


def init_wandb(args):
    """Initialize wandb run."""
    if not WANDB_AVAILABLE:
        print("wandb not available, skipping wandb initialization")
        return None

    if not args.wandb:
        print("wandb logging disabled")
        return None

    # Generate run name if not provided
    if args.wandb_run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pretrained_tag = 'pretrained' if args.pretrained else 'scratch'
        args.wandb_run_name = f'vit_{pretrained_tag}_{timestamp}'

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

    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print("ViT Fine-tuning for Classification")
    print("=" * 60)

    # Initialize wandb
    wandb_run = init_wandb(args)

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = ViTClassifier(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout
    )

    # Load pretrained weights
    if args.pretrained:
        print(f"\nLoading pretrained weights from {args.pretrained}")
        pretrained_state = torch.load(args.pretrained, map_location='cpu')
        model.load_pretrained_encoder(pretrained_state)
        print("Pretrained encoder weights loaded!")

    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Log to wandb
    if wandb_run is not None:
        wandb_run.config.update({'total_params': total_params})
        wandb.watch(model, log='all', log_freq=100)

    # Data (using simple split for demo - in practice use proper train/val/test splits)
    full_dataset = ClassificationDataset(args.data_path, augment=False)

    # Simple split
    n = len(full_dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Log dataset info
    if wandb_run is not None:
        wandb_run.config.update({
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset)
        })

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optimizer with layer-wise LR decay
    param_groups = get_layer_wise_lr_groups(model, args.lr, args.layer_decay)
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    # Mixed precision
    scaler = GradScaler() if args.amp else None

    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    # Training
    best_f1 = 0
    best_epoch = 0

    print("\nStarting training...")

    for epoch in range(args.epochs):
        # Freeze encoder for initial epochs
        if epoch < args.freeze_epochs:
            for param in model.encoder.parameters():
                param.requires_grad = False
            model.classifier.weight.requires_grad = True
            model.classifier.bias.requires_grad = True
            phase = "frozen"
        else:
            for param in model.encoder.parameters():
                param.requires_grad = True
            phase = "unfrozen"

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, epoch, args, device, wandb_run
        )

        val_metrics = validate(model, val_loader, criterion, args, device)

        scheduler.step()

        # Logging
        print(f"Epoch {epoch} [{phase}]: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.4f}, "
              f"Val F1={val_metrics['f1']:.4f}, Val AUC={val_metrics['auc']:.4f}")

        # TensorBoard logging
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/accuracy', train_acc, epoch)
        for k, v in val_metrics.items():
            if k not in ['preds', 'labels']:
                writer.add_scalar(f'val/{k}', v, epoch)

        # Wandb logging
        if wandb_run is not None:
            log_dict = {
                'train/loss_epoch': train_loss,
                'train/accuracy': train_acc,
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/f1': val_metrics['f1'],
                'val/auc': val_metrics['auc'],
                'epoch': epoch,
                'phase': phase
            }
            wandb_run.log(log_dict)

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))

            # Log best metrics to wandb summary
            if wandb_run is not None:
                wandb_run.summary['best_f1'] = best_f1
                wandb_run.summary['best_epoch'] = best_epoch
                wandb_run.summary['best_accuracy'] = val_metrics['accuracy']
                wandb_run.summary['best_auc'] = val_metrics['auc']

                # Log confusion matrix
                cm_fig = plot_confusion_matrix(val_metrics['labels'], val_metrics['preds'])
                wandb_run.log({
                    'confusion_matrix': wandb.Image(cm_fig, caption=f'Epoch {epoch}')
                })
                plt.close(cm_fig)

                # Save model artifact
                artifact = wandb.Artifact(
                    name='vit-classifier-best',
                    type='model',
                    description=f'Best ViT classifier at epoch {epoch}'
                )
                artifact.add_file(os.path.join(args.output_dir, 'best_model.pth'))
                wandb_run.log_artifact(artifact)

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))

    writer.close()

    # Finish wandb
    if wandb_run is not None:
        wandb_run.finish()

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best F1 score: {best_f1:.4f} at epoch {best_epoch}")
    print("=" * 60)


if __name__ == '__main__':
    main()
