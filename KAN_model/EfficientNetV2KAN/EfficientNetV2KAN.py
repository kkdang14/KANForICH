import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchsummary import summary
import gc
import matplotlib.pyplot as plt
import math

from version.kan.kan import KANLinear

class EfficientNetV2KAN(nn.Module):
    def __init__(self):
        super(EfficientNetV2KAN, self).__init__()
        # Load pre-trained EfficientNetV2 model
        self.efficientnet = models.efficientnet_v2_s(pretrained=True)

        # Freeze EfficientNetV2 layers (if required)
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # Modify the classifier part of EfficientNetV2
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()

        self.kan1 = KANLinear(num_features, 256)
        self.kan2 = KANLinear(256, 10)

    def forward(self, x):
        x = self.efficientnet(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.kan1(x)
        x = self.kan2(x)
        return x

def print_parameter_details(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            params = parameter.numel()  # Number of elements in the tensor
            total_params += params
            print(f"{name}: {params}")
    print(f"Total trainable parameters: {total_params}")
    

def count_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetV2KAN().to(device)
print(model)
print_parameter_details(model)
count_model_size(model)
print(f"Model size: {count_model_size(model):.2f} MB")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_parameters(model)}")

# Clean up
print("\nCleaning up...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
print("Done! EfficientNetV2 + Regular KAN implementation complete.")