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
import shutil
from PIL import Image
import json
from datetime import datetime
import glob

from kcn import ConvNeXtKAN  # Import your model

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


class ImageClassifier:
    def __init__(self, model_path, data_dir, output_dir, class_names=None, 
                 has_ground_truth=False, device=None, batch_size=32):
        """
        Args:
            model_path: path to trained model
            data_dir: folder containing images
            output_dir: folder to save results
            class_names: list of class names (required if has_ground_truth=False)
            has_ground_truth: True if data has structured folders with labels
            device: computing device
            batch_size: batch size for processing
        """
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
            
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.classified_dir = os.path.join(output_dir, "classified_images")
        os.makedirs(self.classified_dir, exist_ok=True)
        
        # Setup class names
        if has_ground_truth:
            # Will be set when loading data
            self.class_names = None
        else:
            if class_names is None:
                raise ValueError("class_names must be provided when has_ground_truth=False")
            self.class_names = class_names
        
        print(f"Using device: {self.device}")
        print(f"Model: {model_path}")
        print(f"Data folder: {data_dir}")
        print(f"Output: {output_dir}")
        print(f"Has ground truth: {has_ground_truth}")
        
    def prepare_data(self):
        """Prepare data loader"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
        
        # Create folders for each class
        for class_name in self.class_names:
            class_dir = os.path.join(self.classified_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
        
        print(f"Classes: {self.class_names}")
        print(f"Loaded {len(self.dataset)} images")
        
    def load_model(self):
        """Load trained model"""
        self.model = ConvNeXtKAN().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print("Model loaded successfully")
    
    def predict_and_evaluate(self):
        """Predict and evaluate if ground truth is available"""
        all_predictions = []
        all_probabilities = []
        all_confidences = []
        all_image_paths = []
        all_true_labels = [] if self.has_ground_truth else None
        
        print("Starting prediction...")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.has_ground_truth:
                    images, true_labels, image_paths = batch_data
                    all_true_labels.extend(true_labels.cpu().numpy())
                else:
                    images, image_paths = batch_data
                
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probs, 1)
                
                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_image_paths.extend(image_paths)
                
                # Progress update
                if (batch_idx + 1) % 10 == 0:
                    processed = min((batch_idx + 1) * self.batch_size, len(self.dataset))
                    print(f"Processed {processed}/{len(self.dataset)} images...")
        
        self.all_predictions = np.array(all_predictions)
        self.all_probabilities = np.array(all_probabilities)
        self.all_confidences = np.array(all_confidences)
        self.all_image_paths = all_image_paths
        self.all_true_labels = np.array(all_true_labels) if all_true_labels else None
        
        print("Prediction completed!")
        
        # Calculate metrics if ground truth available
        if self.has_ground_truth:
            self.calculate_metrics()
        
        # Organize images
        self.organize_images()
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        if not self.has_ground_truth:
            print("Cannot calculate metrics without ground truth labels")
            return
        
        print(f"\n{'='*60}")
        print("EVALUATION METRICS")
        print(f"{'='*60}")
        
        # Overall metrics
        self.accuracy = accuracy_score(self.all_true_labels, self.all_predictions)
        self.precision_macro = precision_score(self.all_true_labels, self.all_predictions, average='macro', zero_division=0)
        self.precision_micro = precision_score(self.all_true_labels, self.all_predictions, average='micro', zero_division=0)
        self.precision_weighted = precision_score(self.all_true_labels, self.all_predictions, average='weighted', zero_division=0)
        
        self.recall_macro = recall_score(self.all_true_labels, self.all_predictions, average='macro', zero_division=0)
        self.recall_micro = recall_score(self.all_true_labels, self.all_predictions, average='micro', zero_division=0)
        self.recall_weighted = recall_score(self.all_true_labels, self.all_predictions, average='weighted', zero_division=0)
        
        self.f1_macro = f1_score(self.all_true_labels, self.all_predictions, average='macro', zero_division=0)
        self.f1_micro = f1_score(self.all_true_labels, self.all_predictions, average='micro', zero_division=0)
        self.f1_weighted = f1_score(self.all_true_labels, self.all_predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        self.precision_per_class = precision_score(self.all_true_labels, self.all_predictions, average=None, zero_division=0)
        self.recall_per_class = recall_score(self.all_true_labels, self.all_predictions, average=None, zero_division=0)
        self.f1_per_class = f1_score(self.all_true_labels, self.all_predictions, average=None, zero_division=0)
        
        # Confusion matrix
        self.cm = confusion_matrix(self.all_true_labels, self.all_predictions)
        
        # Classification report
        self.classification_rep = classification_report(
            self.all_true_labels, self.all_predictions, 
            target_names=self.class_names, zero_division=0, output_dict=True
        )
        
        # Print results
        print(f"Overall Metrics:")
        print(f"{'─'*50}")
        print(f"Accuracy:             {self.accuracy:.4f}")
        print(f"Precision (Macro):    {self.precision_macro:.4f}")
        print(f"Precision (Micro):    {self.precision_micro:.4f}")
        print(f"Precision (Weighted): {self.precision_weighted:.4f}")
        print(f"Recall (Macro):       {self.recall_macro:.4f}")
        print(f"Recall (Micro):       {self.recall_micro:.4f}")
        print(f"Recall (Weighted):    {self.recall_weighted:.4f}")
        print(f"F1-Score (Macro):     {self.f1_macro:.4f}")
        print(f"F1-Score (Micro):     {self.f1_micro:.4f}")
        print(f"F1-Score (Weighted):  {self.f1_weighted:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'─'*70}")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print(f"{'─'*70}")
        for i, class_name in enumerate(self.class_names):
            support = np.sum(self.all_true_labels == i)
            print(f"{class_name:<15} {self.precision_per_class[i]:<10.4f} "
                  f"{self.recall_per_class[i]:<10.4f} {self.f1_per_class[i]:<10.4f} {support:<10}")
        
        # Store metrics
        self.metrics = {
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
            'classification_report': self.classification_rep
        }
    
    def organize_images(self):
        """Organize images into folders by predicted class"""
        organization_log = []
        organized_count = 0
        error_count = 0
        
        # Statistics
        class_counts = {class_name: 0 for class_name in self.class_names}
        correct_predictions = 0
        
        for i, (image_path, pred_label, confidence) in enumerate(
            zip(self.all_image_paths, self.all_predictions, self.all_confidences)
        ):
            try:
                # Get predicted class name
                pred_class = self.class_names[pred_label]
                class_counts[pred_class] += 1
                
                # Check if prediction is correct (if ground truth available)
                is_correct = None
                true_class = None
                if self.has_ground_truth:
                    true_label = self.all_true_labels[i]
                    true_class = self.class_names[true_label]
                    is_correct = (pred_label == true_label)
                    if is_correct:
                        correct_predictions += 1
                
                # Create new filename with prediction info
                filename = os.path.basename(image_path)
                base_name, ext = os.path.splitext(filename)
                
                if self.has_ground_truth:
                    correct_str = "correct" if is_correct else "wrong"
                    new_filename = f"{base_name}_pred-{pred_class}_true-{true_class}_{correct_str}_conf-{confidence:.3f}{ext}"
                else:
                    new_filename = f"{base_name}_pred-{pred_class}_conf-{confidence:.3f}{ext}"
                
                # Destination path
                dest_path = os.path.join(self.classified_dir, pred_class, new_filename)
                
                # Copy image
                shutil.copy2(image_path, dest_path)
                organized_count += 1
                
                # Log entry
                log_entry = {
                    'original_path': image_path,
                    'new_path': dest_path,
                    'predicted_class': pred_class,
                    'confidence': float(confidence),
                    'prediction_index': int(pred_label)
                }
                
                if self.has_ground_truth:
                    log_entry.update({
                        'true_class': true_class,
                        'true_index': int(self.all_true_labels[i]),
                        'is_correct': is_correct
                    })
                
                organization_log.append(log_entry)
                
            except Exception as e:
                print(f"Error organizing image {image_path}: {str(e)}")
                error_count += 1
        
        # Print summary
        print(f"\n{'='*60}")
        print("IMAGE ORGANIZATION SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully organized: {organized_count} images")
        if error_count > 0:
            print(f"Errors encountered: {error_count} images")
        
        if self.has_ground_truth:
            accuracy_org = correct_predictions / len(self.all_predictions)
            print(f"Correct predictions: {correct_predictions}/{len(self.all_predictions)} ({accuracy_org:.1%})")
        
        print(f"\nImages organized by predicted class:")
        print(f"{'─'*40}")
        for class_name, count in class_counts.items():
            percentage = (count / len(self.all_predictions)) * 100
            print(f"{class_name:<20} {count:<8} ({percentage:.1f}%)")
        
        # # Save organization log
        # log_file = os.path.join(self.output_dir, "organization_log.json")
        # with open(log_file, 'w', encoding='utf-8') as f:
        #     json.dump(organization_log, f, indent=2, ensure_ascii=False)
        
        # self.organization_log = organization_log
        self.class_counts = class_counts
        
        return class_counts
    
    def create_visualizations(self):
        """Create analysis charts"""
        plt.style.use('default')
        sns.set_palette("husl")
        
        if self.has_ground_truth:
            # With ground truth: show confusion matrix and metrics
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Confusion Matrix
            sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[0,0])
            axes[0,0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            axes[0,0].set_xlabel('Predicted')
            axes[0,0].set_ylabel('True')
            
            # Overall metrics comparison
            metrics_data = {
                'Precision': [self.precision_macro, self.precision_micro, self.precision_weighted],
                'Recall': [self.recall_macro, self.recall_micro, self.recall_weighted],
                'F1-Score': [self.f1_macro, self.f1_micro, self.f1_weighted]
            }
            
            x = np.arange(3)
            width = 0.25
            
            axes[0,1].bar(x - width, metrics_data['Precision'], width, label='Precision', alpha=0.8)
            axes[0,1].bar(x, metrics_data['Recall'], width, label='Recall', alpha=0.8)
            axes[0,1].bar(x + width, metrics_data['F1-Score'], width, label='F1-Score', alpha=0.8)
            
            axes[0,1].set_xlabel('Averaging Method')
            axes[0,1].set_ylabel('Score')
            axes[0,1].set_title('Overall Metrics Comparison')
            axes[0,1].set_xticks(x)
            axes[0,1].set_xticklabels(['Macro', 'Micro', 'Weighted'])
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # Per-class metrics
            x = np.arange(len(self.class_names))
            width = 0.25
            
            axes[0,2].bar(x - width, self.precision_per_class, width, label='Precision', alpha=0.8)
            axes[0,2].bar(x, self.recall_per_class, width, label='Recall', alpha=0.8)
            axes[0,2].bar(x + width, self.f1_per_class, width, label='F1-Score', alpha=0.8)
            
            axes[0,2].set_xlabel('Classes')
            axes[0,2].set_ylabel('Score')
            axes[0,2].set_title('Per-Class Metrics')
            axes[0,2].set_xticks(x)
            axes[0,2].set_xticklabels(self.class_names, rotation=45, ha='right')
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)
            
        else:
            # Without ground truth: focus on distribution and confidence
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Class Distribution (both cases)
        subplot_idx = (1, 0) if self.has_ground_truth else (0, 0)
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        percentages = [count/sum(counts)*100 for count in counts]
        
        bars = axes[subplot_idx].bar(classes, counts, alpha=0.7, color=sns.color_palette("husl", len(classes)))
        axes[subplot_idx].set_xlabel('Predicted Classes', fontsize=12)
        axes[subplot_idx].set_ylabel('Number of Images', fontsize=12)
        axes[subplot_idx].set_title('Distribution of Predicted Classes', fontsize=14, fontweight='bold')
        axes[subplot_idx].tick_params(axis='x', rotation=45)
        
        # Add percentage labels on bars
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            axes[subplot_idx].text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{percentage:.1f}%', ha='center', va='bottom')
        
        # Confidence Distribution
        subplot_idx = (1, 1) if self.has_ground_truth else (0, 1)
        axes[subplot_idx].hist(self.all_confidences, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
        axes[subplot_idx].axvline(np.mean(self.all_confidences), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(self.all_confidences):.3f}')
        axes[subplot_idx].axvline(np.median(self.all_confidences), color='orange', linestyle='--', 
                    label=f'Median: {np.median(self.all_confidences):.3f}')
        axes[subplot_idx].set_xlabel('Prediction Confidence')
        axes[subplot_idx].set_ylabel('Frequency')
        axes[subplot_idx].set_title('Overall Confidence Distribution')
        axes[subplot_idx].legend()
        axes[subplot_idx].grid(True, alpha=0.3)
        
        # Confidence by Class
        subplot_idx = (1, 2) if self.has_ground_truth else (0, 2)
        class_confidences = []
        class_labels = []
        
        for i, class_name in enumerate(self.class_names):
            class_mask = self.all_predictions == i
            if np.sum(class_mask) > 0:
                class_conf = self.all_confidences[class_mask]
                class_confidences.extend(class_conf)
                class_labels.extend([class_name] * len(class_conf))
        
        if class_confidences:
            conf_df = pd.DataFrame({
                'confidence': class_confidences,
                'class': class_labels
            })
            sns.boxplot(data=conf_df, y='class', x='confidence', ax=axes[subplot_idx])
            axes[subplot_idx].set_title('Confidence Distribution by Class')
            axes[subplot_idx].set_xlabel('Confidence')
        
        # Confidence Ranges
        if not self.has_ground_truth:
            confidence_ranges = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
            range_counts = [
                np.sum((self.all_confidences >= 0.0) & (self.all_confidences < 0.2)),
                np.sum((self.all_confidences >= 0.2) & (self.all_confidences < 0.4)),
                np.sum((self.all_confidences >= 0.4) & (self.all_confidences < 0.6)),
                np.sum((self.all_confidences >= 0.6) & (self.all_confidences < 0.8)),
                np.sum((self.all_confidences >= 0.8) & (self.all_confidences <= 1.0))
            ]
            
            colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
            bars = axes[1,0].bar(confidence_ranges, range_counts, color=colors, alpha=0.7)
            axes[1,0].set_xlabel('Confidence Range')
            axes[1,0].set_ylabel('Number of Images')
            axes[1,0].set_title('Images by Confidence Range')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[1,0].text(bar.get_x() + bar.get_width()/2., height + max(range_counts)*0.01,
                        f'{int(height)}', ha='center', va='bottom')
            
            # Hide unused subplots
            axes[1,1].axis('off')
            axes[1,2].axis('off')
        
        plt.tight_layout()
        
        filename = 'evaluation_analysis.png' if self.has_ground_truth else 'prediction_analysis.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Save detailed results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Text summary
        prefix = 'evaluation' if self.has_ground_truth else 'prediction'
        results_file = os.path.join(self.output_dir, f'{prefix}_results_{timestamp}.txt')
        
        with open(results_file, 'w', encoding='utf-8') as f:
            title = "MODEL EVALUATION RESULTS" if self.has_ground_truth else "IMAGE CLASSIFICATION RESULTS"
            f.write(f"{title}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Data Folder: {self.data_dir}\n")
            f.write(f"Total Images: {len(self.all_predictions)}\n")
            f.write(f"Classes: {self.class_names}\n")
            f.write(f"Has Ground Truth: {self.has_ground_truth}\n\n")
            
            if self.has_ground_truth:
                f.write(f"EVALUATION METRICS:\n")
                f.write(f"{'─'*50}\n")
                f.write(f"Accuracy:             {self.accuracy:.4f}\n")
                f.write(f"Precision (Macro):    {self.precision_macro:.4f}\n")
                f.write(f"Precision (Micro):    {self.precision_micro:.4f}\n")
                f.write(f"Precision (Weighted): {self.precision_weighted:.4f}\n")
                f.write(f"Recall (Macro):       {self.recall_macro:.4f}\n")
                f.write(f"Recall (Micro):       {self.recall_micro:.4f}\n")
                f.write(f"Recall (Weighted):    {self.recall_weighted:.4f}\n")
                f.write(f"F1-Score (Macro):     {self.f1_macro:.4f}\n")
                f.write(f"F1-Score (Micro):     {self.f1_micro:.4f}\n")
                f.write(f"F1-Score (Weighted):  {self.f1_weighted:.4f}\n\n")
                
                f.write(f"PER-CLASS METRICS:\n")
                f.write(f"{'─'*70}\n")
                for i, class_name in enumerate(self.class_names):
                    support = np.sum(self.all_true_labels == i)
                    f.write(f"{class_name}: Precision={self.precision_per_class[i]:.4f}, "
                            f"Recall={self.recall_per_class[i]:.4f}, F1={self.f1_per_class[i]:.4f}, "
                            f"Support={support}\n")
            
            f.write(f"\nCLASS DISTRIBUTION:\n")
            f.write(f"{'─'*40}\n")
            for class_name, count in self.class_counts.items():
                percentage = count / len(self.all_predictions) * 100
                f.write(f"{class_name:<20} {count:<8} ({percentage:.1f}%)\n")
            
            f.write(f"\nCONFIDENCE STATISTICS:\n")
            f.write(f"{'─'*40}\n")
            f.write(f"Mean Confidence:      {np.mean(self.all_confidences):.4f}\n")
            f.write(f"Median Confidence:    {np.median(self.all_confidences):.4f}\n")
            f.write(f"Std Confidence:       {np.std(self.all_confidences):.4f}\n")
            f.write(f"Min Confidence:       {np.min(self.all_confidences):.4f}\n")
            f.write(f"Max Confidence:       {np.max(self.all_confidences):.4f}\n")
            
            # Confidence ranges
            conf_ranges = [
                (0.0, 0.2, "Very Low"),
                (0.2, 0.4, "Low"),
                (0.4, 0.6, "Medium"),
                (0.6, 0.8, "High"),
                (0.8, 1.0, "Very High")
            ]
            
            f.write(f"\nCONFIDENCE RANGES:\n")
            f.write(f"{'─'*40}\n")
            for low, high, label in conf_ranges:
                count = np.sum((self.all_confidences >= low) & (self.all_confidences < high if high < 1.0 else self.all_confidences <= high))
                percentage = count / len(self.all_confidences) * 100
                f.write(f"{label} ({low:.1f}-{high:.1f}): {count} ({percentage:.1f}%)\n")
        
        # # 2. JSON summary
        # json_file = os.path.join(self.output_dir, f'{prefix}_summary_{timestamp}.json')
        
        # summary_data = {
        #     'metadata': {
        #         'timestamp': datetime.now().isoformat(),
        #         'model_path': self.model_path,
        #         'data_directory': self.data_dir,
        #         'output_directory': self.output_dir,
        #         'has_ground_truth': self.has_ground_truth,
        #         'total_images': len(self.all_predictions),
        #         'class_names': self.class_names,
        #         'batch_size': self.batch_size,
        #         'device': str(self.device)
        #     },
        #     'class_distribution': self.class_counts,
        #     'confidence_statistics': {
        #         'mean': float(np.mean(self.all_confidences)),
        #         'median': float(np.median(self.all_confidences)),
        #         'std': float(np.std(self.all_confidences)),
        #         'min': float(np.min(self.all_confidences)),
        #         'max': float(np.max(self.all_confidences)),
        #         'quartiles': {
        #             'q25': float(np.percentile(self.all_confidences, 25)),
        #             'q50': float(np.percentile(self.all_confidences, 50)),
        #             'q75': float(np.percentile(self.all_confidences, 75))
        #         }
        #     }
        # }
        
        # if self.has_ground_truth:
        #     summary_data['evaluation_metrics'] = self.metrics
        
        # with open(json_file, 'w', encoding='utf-8') as f:
        #     json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # # 3. CSV with detailed predictions
        # csv_file = os.path.join(self.output_dir, f'{prefix}_predictions_{timestamp}.csv')
        
        # # Prepare data for CSV
        # csv_data = []
        # for i in range(len(self.all_predictions)):
        #     row = {
        #         'image_path': self.all_image_paths[i],
        #         'predicted_class': self.class_names[self.all_predictions[i]],
        #         'predicted_index': int(self.all_predictions[i]),
        #         'confidence': float(self.all_confidences[i])
        #     }
            
        #     # Add probability for each class
        #     for j, class_name in enumerate(self.class_names):
        #         row[f'prob_{class_name}'] = float(self.all_probabilities[i][j])
            
        #     if self.has_ground_truth:
        #         row.update({
        #             'true_class': self.class_names[self.all_true_labels[i]],
        #             'true_index': int(self.all_true_labels[i]),
        #             'is_correct': bool(self.all_predictions[i] == self.all_true_labels[i])
        #         })
            
        #     csv_data.append(row)
        
        # # Save CSV
        # df = pd.DataFrame(csv_data)
        # df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # print(f"\nResults saved:")
        # print(f"- Text summary: {results_file}")
        # print(f"- JSON summary: {json_file}")
        # print(f"- Detailed CSV: {csv_file}")
        # print(f"- Organization log: organization_log.json")
        
        # return {
        #     'text_file': results_file,
        #     'json_file': json_file,
        #     'csv_file': csv_file,
        #     'summary_data': summary_data
        # }
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting complete image classification analysis...")
        print("=" * 60)
        
        try:
            # Step 1: Prepare data
            print("Step 1: Preparing data...")
            self.prepare_data()
            
            # Step 2: Load model
            print("\nStep 2: Loading model...")
            self.load_model()
            
            # Step 3: Run predictions and evaluation
            print("\nStep 3: Running predictions...")
            self.predict_and_evaluate()
            
            # Step 4: Create visualizations
            print("\nStep 4: Creating visualizations...")
            self.create_visualizations()
            
            # Step 5: Save results
            print("\nStep 5: Saving results...")
            results = self.save_results()
            
            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            return results
            
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            raise e


def main():
    class_names = ["dieu_mua_truyen_thong", "don_ca_tai_tu", "dua_bo_bay_nui", "le_hoi_nghinh_ong"]  # Define your class names
    
    classifier_pred = ImageClassifier(
        model_path=r"C:\Users\HP\OneDrive\Documents\Dang\CourseFile\Luận Văn\code\model\best_convnext_kan.pth",
        data_dir=r"C:\Users\HP\OneDrive\Documents\Dang\CourseFile\Luận Văn\code\data\test2",  # Single folder with images
        output_dir=r"C:\Users\HP\OneDrive\Documents\Dang\CourseFile\Luận Văn\code\test\prediction_results_for_modify_data",
        class_names=class_names,
        has_ground_truth=True,
        batch_size=32
    )
    
    results_eval = classifier_pred.run_complete_analysis()
    print("\nResults for Example 1:")
    print(results_eval)


if __name__ == "__main__":
    main()