import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json
import numpy as np  

from FastKAN_model.DenseNetFastKAN import DenseNetFastKAN 

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class CheckpointManager:
    """Manages model checkpoints and training state."""
    def __init__(self, checkpoint_dir, model_name="densenet_fastkan"):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, epoch, model, optimizer, scheduler, train_losses, val_losses, 
                    train_accs, val_accs, best_val_loss, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_loss': best_val_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model at epoch {epoch} with val_loss: {best_val_loss:.4f}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None):
        """Load checkpoint and resume training."""
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return {
            'epoch': checkpoint['epoch'],
            'train_losses': checkpoint['train_losses'],
            'val_losses': checkpoint['val_losses'],
            'train_accs': checkpoint['train_accs'],
            'val_accs': checkpoint['val_accs'],
            'best_val_loss': checkpoint['best_val_loss']
        }
    
    def get_latest_checkpoint(self):
        """Get the latest checkpoint file."""
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                        if f.startswith(f'{self.model_name}_checkpoint_epoch_')]
        if not checkpoint_files:
            return None
        
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.pth')[0]))
        return os.path.join(self.checkpoint_dir, checkpoint_files[-1])


# Training function
def train(model, train_loader, criterion, optimizer, device, epoch, config):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(train_loader)
    num_epochs = int(config['num_epochs'])

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        batch_progress = (batch_idx + 1) / num_batches * 100
        print(f"\rEpoch {epoch+1} / {num_epochs} - Batch {batch_idx+1}/{num_batches} - Running Loss: {running_loss/total:.4f}, Running Acc: {100.*correct/total:.2f}% [{batch_progress:.1f}%]", end='', flush=True)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    epoch_progress = (epoch + 1) / config['num_epochs'] * 100
    print(f"Epoch {epoch+1} / {num_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% [{epoch_progress:.1f}%]", end='\r', flush=True)
    return epoch_loss, epoch_acc
    
    
def validate(model, val_loader, criterion, device, epoch, config):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(val_loader)
    num_epochs = int(config['num_epochs'])

    with torch.no_grad():  # Disable gradient tracking for validation
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Calculate batch progress percentage
            batch_progress = (batch_idx + 1) / num_batches * 100
            print(f"\rEpoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{num_batches} - Val Running Loss: {running_loss/total:.4f}, Val Running Acc: {100.*correct/total:.2f}% [{batch_progress:.1f}%]", end='', flush=True)

    # Calculate epoch metrics
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = 100. * correct / total
    epoch_progress = (epoch + 1) / num_epochs * 100
    print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.2f}% [{epoch_progress:.1f}%]", end='\r', flush=True)
    
    return epoch_loss, epoch_acc

def save_training_config(config_path, config):
    """Save training configuration."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save training history."""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, np.array(val_losses) - np.array(train_losses), 'g-', label='Overfitting Gap')
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss - Train Loss')
    plt.legend()
    plt.title('Overfitting Monitor')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Configuration
    config = {
        'batch_size': 64,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.001,
        'scheduler_patience': 3,
        'scheduler_factor': 0.5,
        'checkpoint_every': 5,  # Save checkpoint every N epochs
        'data_train': r'C:\Users\HP\Documents\Dang\CourseFile\Luận Văn\code\data\train',
        'data_val': r'C:\Users\HP\Documents\Dang\CourseFile\Luận Văn\code\data\val',
        'checkpoint_dir': r'C:\Users\HP\Documents\Dang\CourseFile\Luận Văn\code\checkpoints',
        'results_dir': r'C:\Users\HP\Documents\Dang\CourseFile\Luận Văn\code\training_result'
    }
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config['results_dir'], 'training_config.json')
    save_training_config(config_path, config)

    # Transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(root=config['data_train'], transform=train_transform)
    val_dataset = datasets.ImageFolder(root=config['data_val'], transform=val_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    print(f"Dataset classes: {class_names} (total: {num_classes})")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model, Loss, and Optimizer
    model = DenseNetFastKAN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', 
        factor=config['scheduler_factor'], 
        patience=config['scheduler_patience'], 
        verbose=True
    )

    # Initialize training components
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'], 
        min_delta=config['early_stopping_min_delta']
    )
    checkpoint_manager = CheckpointManager(config['checkpoint_dir'], "densenet_fastkan")
    
    # Check for existing checkpoint
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    if latest_checkpoint:
        print(f"Found existing checkpoint: {latest_checkpoint}")
        resume = input("Resume training from checkpoint? (y/n): ").lower().strip()
        if resume == 'y':
            checkpoint_data = checkpoint_manager.load_checkpoint(
                latest_checkpoint, model, optimizer, scheduler
            )
            start_epoch = checkpoint_data['epoch'] + 1
            best_val_loss = checkpoint_data['best_val_loss']
            train_losses = checkpoint_data['train_losses']
            val_losses = checkpoint_data['val_losses']
            train_accs = checkpoint_data['train_accs']
            val_accs = checkpoint_data['val_accs']
            print(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

    # Training loop
    print(f"Starting training from epoch {start_epoch + 1}")
    
    for epoch in range(start_epoch, config['num_epochs']):
        print(f'\n=== Epoch {epoch+1}/{config["num_epochs"]} ===')
        
        # Train
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch, config)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, config)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Check for best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        # Save checkpoint
        if (epoch + 1) % config['checkpoint_every'] == 0 or is_best:
            checkpoint_manager.save_checkpoint(
                epoch, model, optimizer, scheduler, 
                train_losses, val_losses, train_accs, val_accs, 
                best_val_loss, is_best
            )
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss: {early_stopping.best_loss:.4f}")
            break
    
    # Final checkpoint save
    checkpoint_manager.save_checkpoint(
        epoch, model, optimizer, scheduler, 
        train_losses, val_losses, train_accs, val_accs, 
        best_val_loss, is_best=True
    )
    
    # Plot training history
    plot_path = os.path.join(config['results_dir'], 'DenseNetFastKAN_training_history.png')
    plot_training_history(train_losses, val_losses, train_accs, val_accs, plot_path)
    
    # Save final metrics
    final_metrics = {
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'best_val_loss': best_val_loss,
        'total_epochs': len(train_losses),
        'early_stopped': len(train_losses) < config['num_epochs']
    }
    
    metrics_path = os.path.join(config['results_dir'], 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"Total epochs: {len(train_losses)}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation accuracy: {val_accs[-1]:.2f}%")
    print(f"Results saved to: {config['results_dir']}")


if __name__ == "__main__":
    main()