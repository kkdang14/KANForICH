import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from FastKAN_model.DenseNetFastKAN import DenseNetFastKAN 

def plot_confusion_matrix(cm, class_names, filename='confusion_matrix_fast.png'):
    """Plots and saves the confusion matrix as an image."""
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
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

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    print(f'Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    return epoch_loss, epoch_acc
    
def validate(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient tracking for validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Calculate epoch metrics
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = 100. * correct / total
    print(f'Validation Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return epoch_loss, epoch_acc

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformations for training and testing data
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Standard normalization for pre-trained models
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),     
        transforms.CenterCrop(224),         
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Datasets - use a more generic path approach
    data_dir = os.path.join(os.getcwd(), 'data')  # Assuming data folder is in current directory
    print(f"Looking for data in: {data_dir}")
    
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=val_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    print(f"Dataset classes: {class_names} (total: {num_classes})")

    # Model, Loss, and Optimizer
    # Make sure the model's output matches the number of classes in your dataset
    model = DenseNetFastKAN(num_classes = num_classes).to(device)
    
    # Update the FastKAN architecture to match your number of classes if needed
    if hasattr(model, 'fastkan') and model.fastkan.layers[-1].output_dim != num_classes:
        print(f"Warning: Model output dimension ({model.fastkan.layers[-1].output_dim})" 
            f"doesn't match number of classes ({num_classes}). Please update the model.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Training loop
    num_epochs = 5
    best_val_loss = float('inf')
    best_model_path = r'C:\Users\HP\OneDrive\Documents\Dang\CourseFile\Luận Văn\code\model\densenet_fastkan_best_model.pth'
    
    # Metrics tracking
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")

    # Load best model for testing
    model.load_state_dict(torch.load(best_model_path))

    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accs, label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    training_history_path = r'C:\Users\HP\OneDrive\Documents\Dang\CourseFile\Luận Văn\code\training_result\DenseNet_FastKAN_training_history.png'
    plt.savefig(training_history_path)
    plt.close()
    
    print(f"Best model saved as '{best_model_path}'")


if __name__ == "__main__":
    main()