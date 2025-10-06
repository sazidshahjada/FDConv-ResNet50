"""
ResNet-50 with Frequency Domain Convolution (FDConv) Implementation

This module implements ResNet-50 architecture using FDConv layers instead of standard 
convolutions. FDConv performs convolution operations in the frequency domain using FFT,
providing enhanced feature learning capabilities with adaptive kernel generation.

Key Features:
- ResNet-50 architecture with 50 layers
- FDConv replaces standard Conv2d for 1x1 and 3x3 convolutions  
- Frequency domain processing with attention mechanisms
- Adaptive kernel generation based on input features
- Support for pretrained weights (partial loading for conv1 only)

Architecture Details:
- Input: 224x224 RGB images
- Layers: [3, 4, 6, 3] blocks in layers 1-4
- FDConv parameters: kernel_num=4, reduction=0.0625
- Output: 1000 classes (ImageNet)

Performance Benefits:
- Enhanced feature learning through frequency domain processing
- Adaptive convolution kernels based on input characteristics
- Multi-scale attention mechanisms (global, local, spatial, kernel)
- Frequency band modulation for better feature decomposition

Author: Based on FDConv implementation from https://github.com/Linwei-Chen/FDConv.git
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from FDConv import FDConv
from utils import print_model_summary


# =============================================================================
# Bottleneck Block with FDConv
# =============================================================================

class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet with FDConv layers.
    
    This is the fundamental building block of ResNet-50, using the bottleneck design
    with 1x1 -> 3x3 -> 1x1 convolutions. All convolutions are replaced with FDConv
    for enhanced feature learning through frequency domain processing.

    Args:
        inplanes (int): Number of input channels
        planes (int): Number of base channels (before expansion)
        stride (int): Stride for the 3x3 convolution (default: 1)
        downsample (nn.Module): Downsampling layer for residual connection
        groups (int): Number of groups for grouped convolution (default: 1)
        base_width (int): Base width for calculating actual width (default: 64)
        dilation (int): Dilation rate for convolution (default: 1)
        norm_layer (nn.Module): Normalization layer (default: BatchNorm2d)
    
    Attributes:
        expansion (int): Channel expansion factor (4 for bottleneck)
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # Calculate the actual width for grouped convolution
        width = int(planes * (base_width / 64.)) * groups
        
        # Bottleneck architecture with FDConv layers
        # 1x1 convolution for channel reduction
        self.conv1 = FDConv(inplanes, width, kernel_size=1, stride=1, bias=False, kernel_num=4, reduction=0.0625)
        self.bn1 = norm_layer(width)
        
        # 3x3 convolution for spatial processing
        self.conv2 = FDConv(width, width, kernel_size=3, stride=stride, padding=dilation, bias=False, groups=groups, dilation=dilation, kernel_num=4, reduction=0.0625)
        self.bn2 = norm_layer(width)
        
        # 1x1 convolution for channel expansion
        self.conv3 = FDConv(width, planes * self.expansion, kernel_size=1, stride=1, bias=False, kernel_num=4, reduction=0.0625)
        self.bn3 = norm_layer(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass through the bottleneck block.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor with residual connection applied
        """
        identity = x

        # Main path: 1x1 -> 3x3 -> 1x1 convolutions
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply downsampling to identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual connection
        out += identity
        out = self.relu(out)

        return out


# =============================================================================
# ResNet Architecture with FDConv
# =============================================================================

class FDResNet(nn.Module):
    """
    ResNet architecture with FDConv layers.
    
    This implementation replaces standard convolutions with FDConv (Frequency Domain 
    Convolution) layers, which perform convolution operations in the frequency domain
    using FFT for enhanced feature learning and adaptive kernel generation.
    
    Args:
        block (nn.Module): Block type to use (Bottleneck for ResNet-50)
        layers (list): Number of blocks in each layer [3, 4, 6, 3] for ResNet-50
        num_classes (int): Number of output classes (default: 1000)
        zero_init_residual (bool): Zero-initialize last BN in each block (default: False)
        groups (int): Number of groups for grouped convolution (default: 1)
        width_per_group (int): Width per group for grouped convolution (default: 64)
        replace_stride_with_dilation (list): Replace stride with dilation (default: None)
        norm_layer (nn.Module): Normalization layer (default: BatchNorm2d)
    
    Attributes:
        inplanes (int): Current number of input channels for layer construction
        dilation (int): Current dilation rate
        groups (int): Number of groups for convolution
        base_width (int): Base width for width calculation
    """
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, 
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        # Initialize network parameters
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        
        # Stem layers (initial feature extraction)
        # Note: Keep 7x7 conv as standard Conv2d since FDConv only supports kernel_size in [1, 3]
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Build ResNet layers with progressively increasing channels
        self.layer1 = self._make_layer(block, 64, layers[0])  # 64 -> 256 channels
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])  # 128 -> 512 channels
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])  # 256 -> 1024 channels
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])  # 512 -> 2048 channels
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights(zero_init_residual)

    def _initialize_weights(self, zero_init_residual):
        """
        Initialize network weights using appropriate initialization schemes.
        
        Args:
            zero_init_residual (bool): Whether to zero-initialize residual connections
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Standard Conv2d layers (like conv1) - apply kaiming initialization
                # FDConv layers handle their own initialization internally
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch for better training
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        Build a ResNet layer consisting of multiple blocks.
        
        This method constructs a sequence of blocks with the specified configuration.
        The first block may have a different stride and/or downsample layer for 
        spatial reduction and channel adjustment.
        
        Args:
            block (nn.Module): Block class to instantiate (e.g., Bottleneck)
            planes (int): Number of base channels for this layer
            blocks (int): Number of blocks in this layer
            stride (int): Stride for the first block (default: 1)
            dilate (bool): Whether to use dilation instead of stride (default: False)
            
        Returns:
            nn.Sequential: A sequential container of blocks forming one layer
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        # Handle dilation vs stride trade-off
        if dilate:
            self.dilation *= stride
            stride = 1
            
        # Create downsampling layer if spatial or channel dimensions change
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                FDConv(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, kernel_num=4, reduction=0.0625),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # First block may have stride and/or downsample
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, 
                           self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        
        # Remaining blocks have no stride or downsample
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, 
                               base_width=self.base_width, dilation=self.dilation, 
                               norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the ResNet with FDConv.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, 3, 224, 224)
            
        Returns:
            torch.Tensor: Class logits with shape (B, num_classes)
        """
        # Stem: Initial feature extraction
        x = self.conv1(x)      # (B, 64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # (B, 64, 56, 56)

        # Progressive feature learning through ResNet layers
        x = self.layer1(x)     # (B, 256, 56, 56)
        x = self.layer2(x)     # (B, 512, 28, 28)
        x = self.layer3(x)     # (B, 1024, 14, 14)
        x = self.layer4(x)     # (B, 2048, 7, 7)

        # Classification head
        x = self.avgpool(x)    # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)  # (B, 2048)
        x = self.fc(x)         # (B, num_classes)

        return x


# =============================================================================
# Model Factory Functions
# =============================================================================

def resnet50_fdconv(pretrained=False, **kwargs):
    """
    Constructs a ResNet-50 model with FDConv layers.
    
    This function creates a ResNet-50 architecture where standard convolutions
    are replaced with FDConv (Frequency Domain Convolution) layers for enhanced
    feature learning through frequency domain processing.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
                          Note: Only the initial conv1 layer weights are loaded
                          as FDConv layers have different parameter structure.
        **kwargs: Additional keyword arguments passed to ResNet constructor
        
    Returns:
        ResNet: A ResNet-50 model with FDConv layers
        
    Example:
        >>> model = resnet50_fdconv(num_classes=1000)
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> output = model(input_tensor)
        >>> print(output.shape)  # torch.Size([1, 1000])
    """
    model = FDResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        # Load pretrained weights from standard ResNet-50
        state_dict = torch.hub.load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth', 
            progress=True
        )
        
        # Only load weights for conv1 (standard Conv2d layer)
        # FDConv layers have different parameter structure (dft_weight vs weight)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() 
                     if k in model_dict and 'conv1' in k}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict, strict=False)
        
        print("Loaded pretrained weights for conv1 layer only.")
        print("FDConv layers are randomly initialized.")
    
    return model





# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == '__main__':
    print("Initializing ResNet-50 with FDConv...")
    
    # Create model with standard ImageNet configuration
    model = resnet50_fdconv(num_classes=1000)
    
    # Print comprehensive model summary
    print_model_summary(model)
    