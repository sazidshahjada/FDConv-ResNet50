import torch
import torch.nn as nn
import time
import warnings
from collections import defaultdict, OrderedDict
from typing import Tuple, Dict, List, Optional, Any, Union


# =============================================================================
# Core Analysis Functions
# =============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> Dict[str, int]:
    """
    Count model parameters with detailed breakdown.
    
    Args:
        model (nn.Module): PyTorch model
        trainable_only (bool): If True, count only trainable parameters
        
    Returns:
        Dict[str, int]: Dictionary with parameter counts
    """
    if trainable_only:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    return {
        'trainable': trainable,
        'non_trainable': non_trainable,
        'total': trainable + non_trainable
    }


def analyze_layer_types(model: nn.Module) -> Dict[str, int]:
    """
    Analyze and count different layer types in the model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        Dict[str, int]: Dictionary with layer type counts
    """
    layer_counts = defaultdict(int)
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        # Skip the root model itself
        if name == '':
            continue
            
        # Count important layer types regardless of having children
        important_layers = ['FDConv', 'Conv2d', 'Conv1d', 'Linear', 'LSTM', 'GRU', 'TransformerEncoder']
        
        if any(layer in module_type for layer in important_layers):
            layer_counts[module_type] += 1
        elif len(list(module.children())) == 0:  # Count leaf modules for other types
            layer_counts[module_type] += 1
    
    return dict(layer_counts)


def get_model_size_mb(model: nn.Module, precision: str = 'float32') -> float:
    """
    Calculate model size in megabytes.
    
    Args:
        model (nn.Module): PyTorch model
        precision (str): Parameter precision ('float32', 'float16', 'int8')
        
    Returns:
        float: Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    precision_bytes = {
        'float32': 4,
        'float16': 2,
        'float64': 8,
        'int8': 1,
        'int32': 4
    }
    
    bytes_per_param = precision_bytes.get(precision, 4)
    
    for param in model.parameters():
        param_size += param.numel() * bytes_per_param
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * bytes_per_param
    
    return (param_size + buffer_size) / (1024 ** 2)


def estimate_inference_memory(model: nn.Module, input_size: Tuple[int, ...], 
                             batch_size: int = 1, precision: str = 'float32') -> Dict[str, float]:
    """
    Estimate memory usage during inference.
    
    Args:
        model (nn.Module): PyTorch model
        input_size (tuple): Input tensor size (C, H, W) or (sequence_length, features)
        batch_size (int): Batch size for estimation
        precision (str): Computation precision
        
    Returns:
        Dict[str, float]: Memory usage breakdown in MB
    """
    precision_bytes = {'float32': 4, 'float16': 2, 'float64': 8}
    bytes_per_element = precision_bytes.get(precision, 4)
    
    # Model parameters
    model_size = get_model_size_mb(model, precision)
    
    # Input tensor
    input_elements = batch_size * torch.prod(torch.tensor(input_size))
    input_memory = input_elements * bytes_per_element / (1024 ** 2)
    
    # Rough estimation of intermediate activations (2x input size is conservative)
    activation_memory = input_memory * 2
    
    return {
        'model_parameters': model_size,
        'input_tensor': input_memory,
        'estimated_activations': activation_memory,
        'total_estimated': model_size + input_memory + activation_memory
    }


def profile_inference_time(model: nn.Module, input_tensor: torch.Tensor, 
                          num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
    """
    Profile model inference time.
    
    Args:
        model (nn.Module): PyTorch model
        input_tensor (torch.Tensor): Input tensor for profiling
        num_runs (int): Number of inference runs for averaging
        warmup_runs (int): Number of warmup runs (not counted)
        
    Returns:
        Dict[str, float]: Timing statistics in milliseconds
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Profile
    torch.cuda.synchronize() if device.type == 'cuda' else None
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        'mean_ms': sum(times) / len(times),
        'std_ms': torch.tensor(times).std().item(),
        'min_ms': min(times),
        'max_ms': max(times),
        'median_ms': sorted(times)[len(times) // 2]
    }


def detect_model_architecture(model: nn.Module) -> Dict[str, Any]:
    """
    Automatically detect model architecture characteristics.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        Dict[str, Any]: Architecture information
    """
    layer_counts = analyze_layer_types(model)
    total_layers = sum(layer_counts.values())
    
    # Detect model family based on common patterns
    model_family = "Unknown"
    if "Bottleneck" in str(type(model)):
        model_family = "ResNet-style"
    elif "BasicBlock" in str(type(model)):
        model_family = "ResNet-style (Basic)"
    elif "TransformerBlock" in str(type(model)) or "Attention" in str(type(model)):
        model_family = "Transformer-style"
    elif "InvertedResidual" in str(type(model)):
        model_family = "MobileNet-style"
    
    # Detect special layers
    has_fdconv = layer_counts.get('FDConv', 0) > 0
    has_attention = any('Attention' in layer for layer in layer_counts.keys())
    has_normalization = any(norm in layer_counts for norm in ['BatchNorm2d', 'LayerNorm', 'GroupNorm'])
    
    return {
        'model_family': model_family,
        'total_layers': total_layers,
        'has_fdconv': has_fdconv,
        'has_attention': has_attention,
        'has_normalization': has_normalization,
        'layer_distribution': layer_counts
    }


# =============================================================================
# Generalized Model Summary Functions
# =============================================================================

def print_generalized_model_summary(model: nn.Module, 
                                   input_size: Tuple[int, ...] = (3, 224, 224),
                                   batch_size: int = 1,
                                   device: str = 'auto',
                                   include_timing: bool = False,
                                   include_detailed_layers: bool = False,
                                   precision: str = 'float32',
                                   model_name: str = None) -> None:
    """
    Print a comprehensive, generalized model summary that works with any PyTorch model.
    
    Args:
        model (nn.Module): PyTorch model to analyze
        input_size (tuple): Input tensor size (C, H, W) for image models or appropriate dims
        batch_size (int): Batch size for analysis
        device (str): Device to run analysis on ('auto', 'cpu', 'cuda')
        include_timing (bool): Whether to include inference timing analysis
        include_detailed_layers (bool): Whether to include detailed layer breakdown
        precision (str): Computation precision for memory estimation
        model_name (str): Custom name for the model (auto-detected if None)
    """
    # Device setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
        device = 'cpu'
    
    # Auto-detect model name if not provided
    if model_name is None:
        model_name = type(model).__name__
    
    # Header
    print("=" * 100)
    print(f"{'Generalized Model Summary':^100}")
    print("=" * 100)
    
    # Basic model information
    param_info = count_parameters(model)
    arch_info = detect_model_architecture(model)
    layer_counts = arch_info['layer_distribution']
    
    print(f"Model Name: {model_name}")
    print(f"Model Family: {arch_info['model_family']}")
    print(f"Device: {device.upper()}")
    print(f"Precision: {precision}")
    
    # Parameter information
    print(f"\nParameter Analysis:")
    print(f"  ├─ Trainable Parameters: {param_info['trainable']:,}")
    print(f"  ├─ Non-trainable Parameters: {param_info['non_trainable']:,}")
    print(f"  └─ Total Parameters: {param_info['total']:,}")
    
    # Memory analysis
    memory_info = estimate_inference_memory(model, input_size, batch_size, precision)
    model_size = get_model_size_mb(model, precision)
    
    print(f"\nMemory Analysis:")
    print(f"  ├─ Model Size: {model_size:.2f} MB")
    print(f"  ├─ Input Tensor: {memory_info['input_tensor']:.2f} MB")
    print(f"  ├─ Estimated Activations: {memory_info['estimated_activations']:.2f} MB")
    print(f"  └─ Total Estimated: {memory_info['total_estimated']:.2f} MB")
    
    # Test forward pass
    try:
        model.eval()
        test_input = torch.randn(batch_size, *input_size).to(device)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"\nInput/Output Analysis:")
        print(f"  ├─ Input Shape: {tuple(test_input.shape)}")
        print(f"  ├─ Output Shape: {tuple(output.shape)}")
        if len(output.shape) == 2:  # Classification model
            print(f"  └─ Number of Classes: {output.shape[-1]}")
        else:
            print(f"  └─ Output Dimensions: {output.shape[1:]} (C, H, W) or similar")
        
        # Timing analysis
        if include_timing:
            timing_info = profile_inference_time(model, test_input)
            print(f"\nInference Timing (averaged over 100 runs):")
            print(f"  ├─ Mean: {timing_info['mean_ms']:.2f} ms")
            print(f"  ├─ Std Dev: {timing_info['std_ms']:.2f} ms")
            print(f"  ├─ Min/Max: {timing_info['min_ms']:.2f} / {timing_info['max_ms']:.2f} ms")
            print(f"  └─ Median: {timing_info['median_ms']:.2f} ms")
            
    except Exception as e:
        print(f"\nInput/Output Analysis: Failed - {str(e)}")
    
    # Layer analysis
    print(f"\nArchitecture Analysis:")
    print(f"  ├─ Total Layers: {arch_info['total_layers']}")
    print(f"  ├─ Has FDConv: {'Yes' if arch_info['has_fdconv'] else 'No'}")
    print(f"  ├─ Has Attention: {'Yes' if arch_info['has_attention'] else 'No'}")
    print(f"  └─ Has Normalization: {'Yes' if arch_info['has_normalization'] else 'No'}")
    
    # Layer type breakdown
    if layer_counts:
        print(f"\nLayer Type Distribution:")
        sorted_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (layer_type, count) in enumerate(sorted_layers[:10]):  # Top 10 layer types
            prefix = "├─" if i < len(sorted_layers[:10]) - 1 else "└─"
            print(f"  {prefix} {layer_type}: {count}")
        
        if len(sorted_layers) > 10:
            print(f"  └─ ... and {len(sorted_layers) - 10} more layer types")
    
    # Detailed layer breakdown (optional)
    if include_detailed_layers:
        print(f"\nDetailed Layer Information:")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                param_count = sum(p.numel() for p in module.parameters())
                print(f"  {name}: {type(module).__name__} ({param_count:,} params)")
    
    # Special features detection
    special_features = []
    if arch_info['has_fdconv']:
        fdconv_count = layer_counts.get('FDConv', 0)
        special_features.append(f"FDConv layers ({fdconv_count})")
    
    if special_features:
        print(f"\nSpecial Features:")
        for feature in special_features:
            print(f"  ├─ {feature}")
            if 'FDConv' in feature:
                print(f"  │  ├─ Frequency domain convolution")
                print(f"  │  ├─ Adaptive kernel generation")
                print(f"  │  └─ Multi-scale attention mechanisms")
    
    print("=" * 100)




