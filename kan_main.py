import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import json
import time
import argparse

from KAN_model.ResNetKAN import ResNetKAN
from KAN_model.BasicCNNKAN import BasicCNNKAN
from KAN_model.ConvnextKAN import ConvNeXtKAN
from KAN_model.DenseNetKAN import DenseNetKAN
from KAN_model.EfficientNetV2KAN import EfficientNetV2KAN

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
    def __init__(self, checkpoint_dir, model_name="model"):
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
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
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
        checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.pth')[0]))
        return os.path.join(self.checkpoint_dir, checkpoint_files[-1])

def train(model, train_loader, criterion, optimizer, device, epoch, config):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(train_loader)
    num_epochs = int(config['num_epochs'])

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        batch_progress = (batch_idx + 1) / num_batches * 100
        print(f"\rEpoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{num_batches} - Train Running Loss: {running_loss/total:.4f}, Train Running Acc: {correct/total:.4f} - Progress: [{batch_progress:.1f}%]", end='', flush=True)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    epoch_progress = (epoch + 1) / num_epochs * 100
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} - Progress: [{epoch_progress:.1f}%]")
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device, epoch, config):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(val_loader)
    num_epochs = int(config['num_epochs'])

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            batch_progress = (batch_idx + 1) / num_batches * 100
            print(f"\rEpoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{num_batches} - Val Running Loss: {running_loss/total:.4f}, Val Running Acc: {correct/total:.4f} - Progress: [{batch_progress:.1f}%]", end='', flush=True)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = correct / total
    epoch_progress = (epoch + 1) / num_epochs * 100
    print(f"\rEpoch {epoch+1}/{num_epochs} - Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f} - Progress: [{epoch_progress:.1f}%]")
    return epoch_loss, epoch_acc

def save_training_config(config_path, config):
    """Save config to JSON, excluding non-serializable objects."""
    # Create a copy of config without the model classes
    serializable_config = config.copy()
    
    # Convert models list to only include names (remove class objects)
    if 'models' in serializable_config:
        serializable_config['models'] = [{'name': m['name']} for m in serializable_config['models']]
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
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

def train_model(model_name, model_class, config, device, train_loader, val_loader, num_classes):
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['scheduler_factor'], patience=config['scheduler_patience'], verbose=True
    )
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'], min_delta=config['early_stopping_min_delta']
    )
    checkpoint_manager = CheckpointManager(config['checkpoint_dir'], model_name)
    
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    if latest_checkpoint:
        print(f"\nFound existing checkpoint for {model_name}: {latest_checkpoint}")
        resume = input(f"Resume training {model_name} from checkpoint? (y/n): ").lower().strip()
        if resume == 'y':
            checkpoint_data = checkpoint_manager.load_checkpoint(latest_checkpoint, model, optimizer, scheduler)
            start_epoch = checkpoint_data['epoch'] + 1
            best_val_loss = checkpoint_data['best_val_loss']
            train_losses = checkpoint_data['train_losses']
            val_losses = checkpoint_data['val_losses']
            train_accs = checkpoint_data['train_accs']
            val_accs = checkpoint_data['val_accs']
            print(f"Resumed {model_name} from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
    
    print(f"\nStarting training {model_name} from epoch {start_epoch + 1}")
    start_time = time.time()
    
    for epoch in range(start_epoch, config['num_epochs']):
        print(f'\n=== {model_name} Epoch {epoch+1}/{config["num_epochs"]} ===')
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch, config)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, config)
        print()  # New line after epoch progress
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if (epoch + 1) % config['checkpoint_every'] == 0 or is_best:
            checkpoint_manager.save_checkpoint(
                epoch, model, optimizer, scheduler, train_losses, val_losses, train_accs, val_accs, best_val_loss, is_best
            )
        
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered for {model_name} at epoch {epoch+1}")
            print(f"Best validation loss: {early_stopping.best_loss:.4f}")
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining {model_name} completed in {total_time:.2f} seconds")
    
    checkpoint_manager.save_checkpoint(
        epoch, model, optimizer, scheduler, train_losses, val_losses, train_accs, val_accs, best_val_loss, is_best=True
    )
    
    plot_path = os.path.join(config['results_dir'], f'{model_name}_training_history.png')
    plot_training_history(train_losses, val_losses, train_accs, val_accs, plot_path)
    
    final_metrics = {
        'model_name': model_name,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'best_val_loss': best_val_loss,
        'total_epochs': len(train_losses),
        'early_stopped': len(train_losses) < config['num_epochs']
    }
    metrics_path = os.path.join(config['results_dir'], f'{model_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\n=== {model_name} Training Complete ===")
    print(f"Total epochs: {len(train_losses)}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation accuracy: {val_accs[-1]:.2f}%")
    print(f"Results saved to: {config['results_dir']}")

def get_model_registry():
    """Return a dictionary mapping model names to their classes."""
    return {
        'resnet_kan': ResNetKAN,
        'basic_cnn_kan': BasicCNNKAN,
        'convnext_kan': ConvNeXtKAN,
        'densenet_kan': DenseNetKAN,
        'efficientnetv2_kan': EfficientNetV2KAN
    }

def main():
    parser = argparse.ArgumentParser(description="Train multiple KAN base models")
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--model', type=str, help='Specific model to train (optional)')
    parser.add_argument('--data_train', type=str, default=r'C:\Users\HP\Documents\Dang\CourseFile\Luận Văn\code\data\train', help='Path to training data')
    parser.add_argument('--data_val', type=str, default=r'C:\Users\HP\Documents\Dang\CourseFile\Luận Văn\code\data\val', help='Path to validation data')
    args = parser.parse_args()
    
    # Get model registry
    model_registry = get_model_registry()
    
    # Default configuration - store models as name-class pairs
    config = {
        'batch_size': 32,  # KAN models might need smaller batch sizes due to memory requirements
        'learning_rate': 0.0005,  # Slightly lower learning rate for KAN models
        'num_epochs': 100,
        'early_stopping_patience': 15,  # Increased patience for KAN models
        'early_stopping_min_delta': 0.001,
        'scheduler_patience': 5,  # Increased patience for learning rate scheduler
        'scheduler_factor': 0.5,
        'checkpoint_every': 5,
        'data_train': args.data_train,
        'data_val': args.data_val,
        'checkpoint_dir': r'C:\Users\HP\Documents\Dang\CourseFile\Luận Văn\code\kan_checkpoints',
        'results_dir': r'C:\Users\HP\Documents\Dang\CourseFile\Luận Văn\code\kan_training_result',
        'models': [
            {'name': 'resnet_kan', 'class': ResNetKAN},
            {'name': 'basic_cnn_kan', 'class': BasicCNNKAN},
            {'name': 'convnext_kan', 'class': ConvNeXtKAN},
            {'name': 'densenet_kan', 'class': DenseNetKAN},
            {'name': 'efficientnetv2_kan', 'class': EfficientNetV2KAN}
        ]
    }
    
    # Load config from file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)
            
            # Reconstruct model classes from names if loading from JSON
            if 'models' in loaded_config:
                config['models'] = []
                for model_info in loaded_config['models']:
                    model_name = model_info['name']
                    if model_name in model_registry:
                        config['models'].append({
                            'name': model_name,
                            'class': model_registry[model_name]
                        })
    
    # Override dataset paths if provided via command line
    if args.data_train:
        config['data_train'] = args.data_train
    if args.data_val:
        config['data_val'] = args.data_val
    
    # Filter models if specific model is requested
    if args.model:
        config['models'] = [m for m in config['models'] if m['name'] == args.model]
        if not config['models']:
            raise ValueError(f"Model {args.model} not found in configuration")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    model_names = [m['name'] for m in config['models']]
    
    # Save configuration (this will now work without the JSON serialization error)
    config_path = os.path.join(config['results_dir'], f'{"_".join(model_names)}_training_config.json')
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
    
    # Train each model
    for model_config in config['models']:
        model_name = model_config['name']
        model_class = model_config['class']
        print(f"\nTraining KAN model: {model_name}")
        train_model(model_name, model_class, config, device, train_loader, val_loader, num_classes)

if __name__ == "__main__":
    main()