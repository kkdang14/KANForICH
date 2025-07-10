import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from PIL import Image
from datetime import datetime
import glob

from FastKAN_model.ConvnextFastKAN import ConvNeXtFastKAN 
from FastKAN_model.DenseNetFastKAN import DenseNetFastKAN
from FastKAN_model.EfficientNetV2FastKAN import EfficientNetV2FastKAN
from FastKAN_model.BasicCNNFastKAN import BasicCNNFastKAN

class FlexibleDataset(Dataset):
    """Dataset that can handle both structured (with labels) and single folder (without labels)"""
    def __init__(self, folder_path, has_ground_truth=False, transform=None):
        self.folder_path = folder_path
        self.has_ground_truth = has_ground_truth
        self.transform = transform
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        if has_ground_truth:
            self._load_structured_data()
        else:
            self._load_single_folder_data()
        
        print(f"Found {len(self.image_paths)} images in {folder_path}")
        if has_ground_truth:
            print(f"Classes: {list(self.class_to_idx.keys())}")
    
    def _load_structured_data(self):
        """Load data from structured folders (class/images)"""
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Get all class folders
        class_folders = [d for d in os.listdir(self.folder_path) 
                        if os.path.isdir(os.path.join(self.folder_path, d))]
        class_folders.sort()
        
        for idx, class_name in enumerate(class_folders):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            
            class_path = os.path.join(self.folder_path, class_name)
            for ext in self.image_extensions:
                class_images = glob.glob(os.path.join(class_path, f"*{ext}"))
                class_images.extend(glob.glob(os.path.join(class_path, f"*{ext.upper()}")))
                
                self.image_paths.extend(class_images)
                self.labels.extend([idx] * len(class_images))
    
    def _load_single_folder_data(self):
        """Load data from single folder (no labels)"""
        self.image_paths = []
        
        for ext in self.image_extensions:
            self.image_paths.extend(glob.glob(os.path.join(self.folder_path, f"*{ext}")))
            self.image_paths.extend(glob.glob(os.path.join(self.folder_path, f"*{ext.upper()}")))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            if self.has_ground_truth:
                label = self.labels[idx]
                return image, label, image_path
            else:
                return image, image_path
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            dummy_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            
            if self.has_ground_truth:
                label = self.labels[idx] if idx < len(self.labels) else 0
                return dummy_image, label, image_path
            else:
                return dummy_image, image_path


class SimpleImageEvaluator:
    def __init__(self, model_path, data_dir, output_dir, class_names=None, 
                has_ground_truth=True, device=None, batch_size=32):
        
        self.model_path = model_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.has_ground_truth = has_ground_truth
        self.batch_size = batch_size
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup class names
        if has_ground_truth:
            self.class_names = None 
        else:
            if class_names is None:
                raise ValueError("class_names must be provided when has_ground_truth=False")
            self.class_names = class_names
        
        print(f"Using device: {self.device}")
        print(f"Model: {model_path}")
        print(f"Data folder: {data_dir}")
        print(f"Output: {output_dir}")
        
    def prepare_data(self):
        """Prepare data loader"""
        transform =  transforms.Compose([
            transforms.Resize((224, 224)),       # Cố định
            transforms.ToTensor(),               # Không flip
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.dataset = FlexibleDataset(
            self.data_dir, 
            has_ground_truth=self.has_ground_truth, 
            transform=transform
        )
        
        if self.has_ground_truth:
            self.class_names = [self.dataset.idx_to_class[i] for i in range(len(self.dataset.class_to_idx))]
        
        self.num_classes = len(self.class_names)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"Classes: {self.class_names}")
        print(f"Loaded {len(self.dataset)} images")
        
    def load_model(self):
        """Load trained model"""
        self.model = EfficientNetV2FastKAN(num_classes=11).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print("Model loaded successfully")
    
    def predict(self):
        """Run model predictions"""
        all_predictions = []
        all_true_labels = []
        
        print("Starting prediction...")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.dataloader):
                images, true_labels, image_paths = batch_data
                all_true_labels.extend(true_labels.cpu().numpy())
                
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                
                # Progress update
                if (batch_idx + 1) % 10 == 0:
                    processed = min((batch_idx + 1) * self.batch_size, len(self.dataset))
                    print(f"Processed {processed}/{len(self.dataset)} images...")
        
        self.all_predictions = np.array(all_predictions)
        self.all_true_labels = np.array(all_true_labels)
        
        print("Prediction completed!")
    
    def calculate_metrics(self):
        """Calculate core evaluation metrics"""
        print(f"\n{'='*70}")
        print("EVALUATION METRICS")
        print(f"{'='*70}")
        
        # Core metrics
        self.accuracy = accuracy_score(self.all_true_labels, self.all_predictions)
        
        # Precision
        self.precision_macro = precision_score(self.all_true_labels, self.all_predictions, average='macro', zero_division=0)
        self.precision_micro = precision_score(self.all_true_labels, self.all_predictions, average='micro', zero_division=0)
        self.precision_weighted = precision_score(self.all_true_labels, self.all_predictions, average='weighted', zero_division=0)
        self.precision_per_class = precision_score(self.all_true_labels, self.all_predictions, average=None, zero_division=0)
        
        # Recall
        self.recall_macro = recall_score(self.all_true_labels, self.all_predictions, average='macro', zero_division=0)
        self.recall_micro = recall_score(self.all_true_labels, self.all_predictions, average='micro', zero_division=0)
        self.recall_weighted = recall_score(self.all_true_labels, self.all_predictions, average='weighted', zero_division=0)
        self.recall_per_class = recall_score(self.all_true_labels, self.all_predictions, average=None, zero_division=0)
        
        # F1-Score
        self.f1_macro = f1_score(self.all_true_labels, self.all_predictions, average='macro', zero_division=0)
        self.f1_micro = f1_score(self.all_true_labels, self.all_predictions, average='micro', zero_division=0)
        self.f1_weighted = f1_score(self.all_true_labels, self.all_predictions, average='weighted', zero_division=0)
        self.f1_per_class = f1_score(self.all_true_labels, self.all_predictions, average=None, zero_division=0)
        
        # Confusion matrix
        self.cm = confusion_matrix(self.all_true_labels, self.all_predictions)
        
        # Print results
        print(f"Overall Metrics:")
        print(f"{'─'*50}")
        print(f"Accuracy:             {self.accuracy:.4f}")
        print()
        print(f"Precision (Macro):    {self.precision_macro:.4f}")
        print(f"Precision (Micro):    {self.precision_micro:.4f}")
        print(f"Precision (Weighted): {self.precision_weighted:.4f}")
        print()
        print(f"Recall (Macro):       {self.recall_macro:.4f}")
        print(f"Recall (Micro):       {self.recall_micro:.4f}")
        print(f"Recall (Weighted):    {self.recall_weighted:.4f}")
        print()
        print(f"F1-Score (Macro):     {self.f1_macro:.4f}")
        print(f"F1-Score (Micro):     {self.f1_micro:.4f}")
        print(f"F1-Score (Weighted):  {self.f1_weighted:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'─'*80}")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print(f"{'─'*80}")
        for i, class_name in enumerate(self.class_names):
            support = np.sum(self.all_true_labels == i)
            print(f"{class_name:<20} {self.precision_per_class[i]:<12.4f} "
                f"{self.recall_per_class[i]:<12.4f} {self.f1_per_class[i]:<12.4f} {support:<10}")
        
        print(f"\nConfusion Matrix:")
        print(f"{'─'*50}")
        print("Rows: True labels, Columns: Predicted labels")
        print(self.cm)
    
    def create_confusion_matrix_plot(self):
        """Create and save confusion matrix visualization"""
        plt.figure(figsize=(10, 8))
        
        # Calculate percentages for better readability
        cm_percent = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both count and percentage
        annot = np.empty_like(self.cm).astype(str)
        for i in range(self.cm.shape[0]):
            for j in range(self.cm.shape[1]):
                count = self.cm[i, j]
                percent = cm_percent[i, j]
                annot[i, j] = f'{count}\n({percent:.1f}%)'
        
        # Create heatmap
        sns.heatmap(self.cm, 
                    annot=annot, 
                    fmt='', 
                    cmap='Blues',
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Number of samples'})
        
        plt.title('Confusion Matrix\n(Count and Percentage)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Confusion matrix plot saved: {plot_file}")
    
    def create_metrics_comparison_plot(self):
        """Create metrics comparison visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Overall metrics comparison
        metrics_data = {
            'Macro': [self.precision_macro, self.recall_macro, self.f1_macro],
            'Micro': [self.precision_micro, self.recall_micro, self.f1_micro],
            'Weighted': [self.precision_weighted, self.recall_weighted, self.f1_weighted]
        }
        
        x = np.arange(3)
        width = 0.25
        
        ax1.bar(x - width, metrics_data['Macro'], width, label='Macro', alpha=0.8)
        ax1.bar(x, metrics_data['Micro'], width, label='Micro', alpha=0.8)
        ax1.bar(x + width, metrics_data['Weighted'], width, label='Weighted', alpha=0.8)
        
        ax1.set_xlabel('Metrics', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Overall Metrics Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Per-class metrics
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax2.bar(x - width, self.precision_per_class, width, label='Precision', alpha=0.8)
        ax2.bar(x, self.recall_per_class, width, label='Recall', alpha=0.8)
        ax2.bar(x + width, self.f1_per_class, width, label='F1-Score', alpha=0.8)
        
        ax2.set_xlabel('Classes', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.output_dir, 'metrics_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Metrics comparison plot saved: {plot_file}")
    
    def save_results(self):
        """Save evaluation results to text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f'evaluation_results_{timestamp}.txt')
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("MODEL EVALUATION RESULTS\n")
            f.write("=" * 70 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Data Folder: {self.data_dir}\n")
            f.write(f"Total Images: {len(self.all_predictions)}\n")
            f.write(f"Classes: {self.class_names}\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write("─" * 50 + "\n")
            f.write(f"Accuracy:             {self.accuracy:.4f}\n\n")
            f.write(f"Precision (Macro):    {self.precision_macro:.4f}\n")
            f.write(f"Precision (Micro):    {self.precision_micro:.4f}\n")
            f.write(f"Precision (Weighted): {self.precision_weighted:.4f}\n\n")
            f.write(f"Recall (Macro):       {self.recall_macro:.4f}\n")
            f.write(f"Recall (Micro):       {self.recall_micro:.4f}\n")
            f.write(f"Recall (Weighted):    {self.recall_weighted:.4f}\n\n")
            f.write(f"F1-Score (Macro):     {self.f1_macro:.4f}\n")
            f.write(f"F1-Score (Micro):     {self.f1_micro:.4f}\n")
            f.write(f"F1-Score (Weighted):  {self.f1_weighted:.4f}\n\n")
            
            f.write("PER-CLASS METRICS:\n")
            f.write("─" * 80 + "\n")
            f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
            f.write("─" * 80 + "\n")
            for i, class_name in enumerate(self.class_names):
                support = np.sum(self.all_true_labels == i)
                f.write(f"{class_name:<20} {self.precision_per_class[i]:<12.4f} "
                        f"{self.recall_per_class[i]:<12.4f} {self.f1_per_class[i]:<12.4f} {support:<10}\n")
            
            f.write(f"\nCONFUSION MATRIX:\n")
            f.write("─" * 50 + "\n")
            f.write("Rows: True labels, Columns: Predicted labels\n")
            f.write(f"Classes: {self.class_names}\n\n")
            
            # Write confusion matrix with class names
            f.write("     ")
            for class_name in self.class_names:
                f.write(f"{class_name[:8]:<10}")
            f.write("\n")
            
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name[:8]:<5}")
                for j in range(len(self.class_names)):
                    f.write(f"{self.cm[i][j]:<10}")
                f.write("\n")
        
        print(f"Results saved to: {results_file}")
        return results_file
    
    def run_evaluation(self):
        """Run the complete evaluation pipeline"""
        print("Starting model evaluation...")
        print("=" * 70)
        
        try:
            # Step 1: Prepare data
            print("Step 1: Preparing data...")
            self.prepare_data()
            
            # Step 2: Load model
            print("\nStep 2: Loading model...")
            self.load_model()
            
            # Step 3: Run predictions
            print("\nStep 3: Running predictions...")
            self.predict()
            
            # Step 4: Calculate metrics
            print("\nStep 4: Calculating metrics...")
            self.calculate_metrics()
            
            # Step 5: Create visualizations
            print("\nStep 5: Creating visualizations...")
            self.create_confusion_matrix_plot()
            self.create_metrics_comparison_plot()
            
            # Step 6: Save results
            print("\nStep 6: Saving results...")
            results_file = self.save_results()
            
            print("\n" + "=" * 70)
            print("EVALUATION COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            
            return {
                'accuracy': self.accuracy,
                'precision': {
                    'macro': self.precision_macro,
                    'micro': self.precision_micro,
                    'weighted': self.precision_weighted,
                    'per_class': self.precision_per_class.tolist()
                },
                'recall': {
                    'macro': self.recall_macro,
                    'micro': self.recall_micro, 
                    'weighted': self.recall_weighted,
                    'per_class': self.recall_per_class.tolist()
                },
                'f1_score': {
                    'macro': self.f1_macro,
                    'micro': self.f1_micro,
                    'weighted': self.f1_weighted,
                    'per_class': self.f1_per_class.tolist()
                },
                'confusion_matrix': self.cm.tolist(),
                'results_file': results_file
            }
            
        except Exception as e:
            print(f"\nError during evaluation: {str(e)}")
            raise e


def main():
    evaluator = SimpleImageEvaluator(
        model_path=r"C:\Users\HP\OneDrive\Documents\Dang\CourseFile\Luận Văn\code\model\efficientnetv2_fastkan_best_model.pth",
        data_dir=r"C:\Users\HP\OneDrive\Documents\Dang\CourseFile\Luận Văn\code\data\test",
        output_dir=r"C:\Users\HP\OneDrive\Documents\Dang\CourseFile\Luận Văn\code\test\efficientnetv2_fastkan_evaluation_results",
        has_ground_truth=True,
        batch_size=32
    )
    
    results = evaluator.run_evaluation()
    print(f"\nFinal Results Summary:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['f1_score']['macro']:.4f}")
    print(f"Weighted F1: {results['f1_score']['weighted']:.4f}")


if __name__ == "__main__":
    main()