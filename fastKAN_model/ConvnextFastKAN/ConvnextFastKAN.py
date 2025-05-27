import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gc
import math

from version.fastkan.fastkan import FastKAN  # Assuming FastKAN is defined in fastkan.py
from sklearn.metrics import classification_report, confusion_matrix

class ConvNeXtFastKAN(nn.Module):
    def __init__(self, hidden_dims=None, num_classes=2, pretrained=True, freeze_backbone=True):
        super(ConvNeXtFastKAN, self).__init__()
        
        # Load pre-trained ConvNeXt model
        self.convnext = models.convnext_tiny(pretrained=pretrained)

        # Freeze ConvNeXt layers if specified
        if freeze_backbone:
            for param in self.convnext.parameters():
                param.requires_grad = False

        # Get the feature dimension from ConvNeXt
        num_features = self.convnext.classifier[2].in_features
        self.convnext.classifier = nn.Identity()  # Remove the classifier
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # Create the complete layers list including input and output dimensions
        layers_hidden = [num_features] + hidden_dims + [num_classes]
        
        # Create FastKAN network with the specified architecture
        self.fastkan = FastKAN(
            layers_hidden=layers_hidden,
            grid_min=-2.0,
            grid_max=2.0,
            num_grids=8,
            use_base_update=True
        )

    def forward(self, x):
        x = self.convnext(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fastkan(x)
        return x

def print_parameter_details(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            params = parameter.numel()  # Number of elements in the tensor
            total_params += params
            print(f"{name}: {params}")
    print(f"Total trainable parameters: {total_params}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNeXtFastKAN().to(device)
print(model)
print_parameter_details(model)

# Note: torchsummary might not work correctly with the complex FastKAN architecture
# You can use a simpler approach to print the model summary
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_parameters(model)}")