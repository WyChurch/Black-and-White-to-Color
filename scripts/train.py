"""
Training script for colorization model
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.colorization_model import ColorizationModel
from utils.data_utils import create_dataloader, lab_to_rgb


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for l_channel, ab_channels in tqdm(dataloader, desc="Training"):
        l_channel = l_channel.to(device)
        ab_channels = ab_channels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predicted_ab = model(l_channel)
        
        # Calculate loss
        loss = criterion(predicted_ab, ab_channels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for l_channel, ab_channels in tqdm(dataloader, desc="Validating"):
            l_channel = l_channel.to(device)
            ab_channels = ab_channels.to(device)
            
            predicted_ab = model(l_channel)
            loss = criterion(predicted_ab, ab_channels)
            
            running_loss += loss.item()
    
    return running_loss / len(dataloader)


def save_sample_images(model, dataloader, device, save_dir, epoch, num_samples=4):
    """Save sample colorized images"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (l_channel, ab_channels) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            l_channel = l_channel.to(device)
            ab_channels = ab_channels.to(device)
            
            predicted_ab = model(l_channel)
            
            # Convert to RGB and save
            for j in range(min(l_channel.shape[0], num_samples)):
                l = l_channel[j:j+1]
                ab_pred = predicted_ab[j:j+1]
                ab_true = ab_channels[j:j+1]
                
                # Denormalize predicted AB
                ab_pred_denorm = torch.clamp(ab_pred, 0, 1)
                
                rgb_pred = lab_to_rgb(l, ab_pred_denorm)
                rgb_true = lab_to_rgb(l, ab_true)
                
                # Create comparison image
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(l.squeeze().cpu().numpy(), cmap='gray')
                axes[0].set_title('Input (Grayscale)')
                axes[0].axis('off')
                
                axes[1].imshow(rgb_pred)
                axes[1].set_title('Predicted Color')
                axes[1].axis('off')
                
                axes[2].imshow(rgb_true)
                axes[2].set_title('Ground Truth')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_sample_{j}.png'))
                plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train colorization model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing training images')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Directory containing validation images')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size (square)')
    parser.add_argument('--save_dir', type=str, default='outputs',
                        help='Directory to save model checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    
    # Create data loaders
    print("Loading training data...")
    train_loader = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        shuffle=True,
        size=(args.image_size, args.image_size)
    )
    
    val_loader = None
    if args.val_dir:
        print("Loading validation data...")
        val_loader = create_dataloader(
            args.val_dir,
            batch_size=args.batch_size,
            shuffle=False,
            size=(args.image_size, args.image_size)
        )
    
    # Initialize model
    model = ColorizationModel(input_channels=1, output_channels=2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        writer.add_scalar('Loss/Train', train_loss, epoch)
        
        # Validate
        if val_loader:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.4f}")
            writer.add_scalar('Loss/Val', val_loss, epoch)
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, os.path.join(args.save_dir, 'checkpoints', 'best_model.pth'))
        else:
            scheduler.step(train_loss)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, os.path.join(args.save_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save sample images
        if (epoch + 1) % 5 == 0:
            save_sample_images(
                model,
                train_loader,
                device,
                os.path.join(args.save_dir, 'samples'),
                epoch + 1
            )
    
    writer.close()
    print("\nTraining completed!")


if __name__ == '__main__':
    main()

