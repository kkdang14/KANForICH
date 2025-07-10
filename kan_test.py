import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import argparse
import gc
from sklearn.metrics import classification_report, confusion_matrix
from KAN_model.ResNetKAN import ResNetKAN
from KAN_model.BasicCNNKAN import BasicCNNKAN
from KAN_model.ConvnextKAN import ConvNeXtKAN
from KAN_model.DenseNetKAN import DenseNetKAN
from KAN_model.EfficientNetV2KAN import EfficientNetV2KAN

def print_parameter_details(model):
    total_params = 0
    trainable_params = 0
    
    print("Layer-wise parameter count:")
    print("-" * 60)
    
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params
        
        if parameter.requires_grad:
            trainable_params += params
            print(f"{name}: {params:,} (trainable)")
        else:
            print(f"{name}: {params:,} (frozen)")
    
    print("-" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

def count_model_size(model):
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / 1024 / 1024

def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def save_test_config(config_path, config):
    serializable_config = config.copy()
    serializable_config['models'] = [{'name': m['name']} for m in serializable_config['models']]
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=4)

def plot_confusion_matrix(conf_matrix, class_names, model_name, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def test_model(model_name, model_class, config, device, test_loader, num_classes):
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    
    checkpoint_path = os.path.join(config['checkpoint_dir'], f'{model_name}_best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found for {model_name} at {checkpoint_path}")
    
    model = load_checkpoint(checkpoint_path, model, device)
    model.eval()
    
    print(f"\nTesting KAN-based model: {model_name}")
    print_parameter_details(model)
    print(f"Model size: {count_model_size(model):.2f} MB")
    
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    num_batches = len(test_loader)
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            batch_progress = (batch_idx + 1) / num_batches * 100
            print(f"\rBatch {batch_idx+1}/{num_batches} - Test Running Loss: {running_loss/total:.4f}, Test Running Acc: {correct/total:.4f} - [{batch_progress:.1f}%]", end='', flush=True)
    
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = correct / total
    
    class_names = test_loader.dataset.classes
    clf_report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'model_name': model_name,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'per_class_metrics': {
            cls: {
                'precision': clf_report[cls]['precision'],
                'recall': clf_report[cls]['recall'],
                'f1_score': clf_report[cls]['f1-score'],
                'support': clf_report[cls]['support']
            } for cls in class_names
        },
        'average_metrics': {
            'macro_precision': clf_report['macro avg']['precision'],
            'macro_recall': clf_report['macro avg']['recall'],
            'macro_f1_score': clf_report['macro avg']['f1-score'],
            'weighted_precision': clf_report['weighted avg']['precision'],
            'weighted_recall': clf_report['weighted avg']['recall'],
            'weighted_f1_score': clf_report['weighted avg']['f1-score']
        },
        'confusion_matrix': conf_matrix.tolist(),
        'model_size_mb': count_model_size(model),
        'num_classes': num_classes,
        'class_names': class_names
    }
    
    results_dir = config['results_dir']
    metrics_path = os.path.join(results_dir, f'{model_name}_test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    cm_path = os.path.join(results_dir, f'{model_name}_confusion_matrix.png')
    plot_confusion_matrix(conf_matrix, class_names, model_name, cm_path)
    
    print(f"\n=== {model_name} Test Complete ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("\nAverage Metrics:")
    print(f"Macro Precision: {metrics['average_metrics']['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['average_metrics']['macro_recall']:.4f}")
    print(f"Macro F1-Score: {metrics['average_metrics']['macro_f1_score']:.4f}")
    print(f"Weighted Precision: {metrics['average_metrics']['weighted_precision']:.4f}")
    print(f"Weighted Recall: {metrics['average_metrics']['weighted_recall']:.4f}")
    print(f"Weighted F1-Score: {metrics['average_metrics']['weighted_f1_score']:.4f}")
    print(f"Results saved to: {results_dir}")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return metrics

def get_model_registry():
    return {
        'resnet_kan': ResNetKAN,
        'basic_cnn_kan': BasicCNNKAN,
        'convnext_kan': ConvNeXtKAN,
        'densenet_kan': DenseNetKAN,
        'efficientnetv2_kan': EfficientNetV2KAN
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate KAN-based models on test data")
    parser.add_argument('--config', type=str, help='Path to config JSON file (from training)')
    parser.add_argument('--model', type=str, help='Specific model to test (optional)')
    parser.add_argument('--data_test', type=str, default=r'C:\Users\HP\Documents\Dang\CourseFile\Luận Văn\code\data\test', help='Path to test data')
    args = parser.parse_args()
    
    model_registry = get_model_registry()
    
    config = {
        'batch_size': 32,
        'data_test': args.data_test,
        'checkpoint_dir': r'C:\Users\HP\Documents\Dang\CourseFile\Luận Văn\code\checkpoints_kan',
        'results_dir': r'C:\Users\HP\Documents\Dang\CourseFile\Luận Văn\code\testing_result_kan',
        'models': [
            {'name': 'resnet_kan', 'class': ResNetKAN},
            {'name': 'basic_cnn_kan', 'class': BasicCNNKAN},
            {'name': 'convnext_kan', 'class': ConvNeXtKAN},
            {'name': 'densenet_kan', 'class': DenseNetKAN},
            {'name': 'efficientnetv2_kan', 'class': EfficientNetV2KAN}
        ]
    }
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)
            if 'models' in loaded_config:
                config['models'] = []
                for model_info in loaded_config['models']:
                    model_name = model_info['name']
                    if model_name in model_registry:
                        config['models'].append({
                            'name': model_name,
                            'class': model_registry[model_name]
                        })
    
    if args.data_test:
        config['data_test'] = args.data_test
    
    if args.model:
        config['models'] = [m for m in config['models'] if m['name'] == args.model]
        if not config['models']:
            raise ValueError(f"Model {args.model} not found in configuration")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(config['results_dir'], exist_ok=True)
    
    config_path = os.path.join(config['results_dir'], f'{model_name}_test_config.json')
    save_test_config(config_path, config)
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(root=config['data_test'], transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    class_names = test_dataset.classes
    num_classes = len(class_names)
    
    print(f"Test dataset classes: {class_names} (total: {num_classes})")
    print(f"Test samples: {len(test_dataset)}")
    
    for model_config in config['models']:
        model_name = model_config['name']
        model_class = model_config['class']
        test_model(model_name, model_class, config, device, test_loader, num_classes)

if __name__ == "__main__":
    main()