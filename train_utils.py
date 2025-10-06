"""
Advanced Training and Evaluation Utilities for ResNet and FDResNet

This module provides comprehensive training and evaluation functions with adaptive features
including early stopping, tensorboard support, learning rate scheduling, and comprehensive
evaluation metrics for classification tasks.

Author: Created for FDConv-ResNet50 project
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
from collections import defaultdict
from tqdm import tqdm
import warnings

# For evaluation metrics
try:
    from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
    import seaborn as sns
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn and seaborn not available. Some evaluation features will be limited.")


# =============================================================================
# Early Stopping Implementation
# =============================================================================

class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    
    Args:
        patience (int): Number of epochs to wait for improvement
        min_delta (float): Minimum change to qualify as improvement
        mode (str): 'min' for loss, 'max' for accuracy
        restore_best_weights (bool): Whether to restore best weights when stopping
        verbose (bool): Whether to print early stopping messages
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = 'min', 
                 restore_best_weights: bool = True, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f"Mode {mode} is unknown, please use 'min' or 'max'")
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            score (float): Current validation score
            model (nn.Module): Model to save best weights
            
        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.epochs_no_improve = 0
            self.save_checkpoint(model)
        else:
            self.epochs_no_improve += 1
            
        if self.epochs_no_improve >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Early stopping triggered after {self.epochs_no_improve} epochs without improvement")
            
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print("Restored best weights")
        
        return self.early_stop
    
    def save_checkpoint(self, model: nn.Module):
        """Save model weights"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


# =============================================================================
# Learning Rate Scheduler Factory
# =============================================================================

def create_lr_scheduler(optimizer: optim.Optimizer, scheduler_type: str, **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler based on type.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type (str): Type of scheduler ('step', 'cosine', 'plateau', 'exponential')
        **kwargs: Additional arguments for scheduler
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 50),
            eta_min=kwargs.get('eta_min', 0)
        )
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10)
        )
    elif scheduler_type == 'exponential':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# =============================================================================
# Training Metrics Tracker
# =============================================================================

class MetricsTracker:
    """Track training and validation metrics during training."""
    
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []
        
    def update(self, train_loss: float, train_acc: float, val_loss: float, val_acc: float, 
               lr: float, epoch_time: float):
        """Update metrics for current epoch"""
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
    
    def get_best_epoch(self, metric: str = 'val_acc') -> Tuple[int, float]:
        """Get best epoch based on specified metric"""
        if metric == 'val_acc':
            best_idx = np.argmax(self.val_accuracies)
            return best_idx, self.val_accuracies[best_idx]
        elif metric == 'val_loss':
            best_idx = np.argmin(self.val_losses)
            return best_idx, self.val_losses[best_idx]
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.train_accuracies, label='Train Acc', color='blue')
        axes[0, 1].plot(self.val_accuracies, label='Val Acc', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.learning_rates, color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Epoch time plot
        axes[1, 1].plot(self.epoch_times, color='orange')
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plot saved to {save_path}")
        
        plt.close(fig)  # Close figure instead of showing


# =============================================================================
# Custom Training Function
# =============================================================================

def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    optimizer_type: str = 'adam',
    scheduler_type: str = 'step',
    early_stopping_patience: int = 10,
    device: str = 'auto',
    save_dir: str = './checkpoints',
    model_name: str = 'model',
    tensorboard_log_dir: str = './runs',
    print_freq: int = 100,
    save_best_only: bool = True,
    mixed_precision: bool = False,
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 1e-4,
    **scheduler_kwargs
) -> tuple[nn.Module, MetricsTracker]:
    """
    Advanced training function with adaptive features and interactive progress bar for ResNet models.

    Args:
        model (nn.Module): Model to train (e.g., ResNet)
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Maximum number of epochs
        learning_rate (float): Initial learning rate
        optimizer_type (str): Optimizer type ('adam', 'sgd', 'adamw')
        scheduler_type (str): LR scheduler type ('step', 'cosine', 'plateau', 'exponential')
        early_stopping_patience (int): Patience for early stopping
        device (str): Device to train on ('auto', 'cpu', 'cuda')
        save_dir (str): Directory to save checkpoints
        model_name (str): Name for saving model
        tensorboard_log_dir (str): TensorBoard log directory
        print_freq (int): Frequency to print training progress
        save_best_only (bool): Whether to save only the best model
        mixed_precision (bool): Whether to use mixed precision training
        gradient_accumulation_steps (int): Steps for gradient accumulation
        weight_decay (float): Weight decay for regularization
        **scheduler_kwargs: Additional arguments for learning rate scheduler

    Returns:
        Tuple[nn.Module, MetricsTracker]: Trained model and metrics tracker
    """
    # Device setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    model = model.to(device)
    print(f"Training on device: {device}")
    print(f"Model: {type(model).__name__}")
    
    if device == 'cuda':
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    # Setup optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Setup learning rate scheduler
    scheduler = create_lr_scheduler(optimizer, scheduler_type, **scheduler_kwargs)
    
    # Setup loss function
    criterion = nn.CrossEntropyLoss()
    
    # Setup early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
    
    # Setup metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Setup TensorBoard
    writer = SummaryWriter(tensorboard_log_dir)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler() if mixed_precision and device == 'cuda' else None
    
    print(f"Starting training for {num_epochs} epochs")
    print(f"Optimizer: {optimizer_type.title()}")
    print(f"Scheduler: {scheduler_type.title()}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Mixed precision: {mixed_precision}")
    print("-" * 80)
    
    best_val_acc = 0.0
    start_time = time.time()
    global_step = 0  # Add global step counter for detailed TensorBoard logging
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Initialize tqdm progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            if mixed_precision and scaler is not None:
                with torch.amp.autocast(device_type=device):
                    output = model(data)
                    loss = criterion(output, target) / gradient_accumulation_steps
                scaler.scale(loss).backward()
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                output = model(data)
                loss = criterion(output, target) / gradient_accumulation_steps
                loss.backward()
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * gradient_accumulation_steps * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
            
            # TensorBoard logging for every step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                global_step += 1
                step_loss = loss.item() * gradient_accumulation_steps
                step_acc = 100. * train_correct / train_total
                
                # Log training metrics every step
                writer.add_scalar('Training/Step_Loss', step_loss, global_step)
                writer.add_scalar('Training/Step_Accuracy', step_acc, global_step)
                writer.add_scalar('Training/Learning_Rate_Step', optimizer.param_groups[0]['lr'], global_step)
                
                # Log gradient norms every 10 steps
                if global_step % 10 == 0:
                    total_grad_norm = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            total_grad_norm += param.grad.data.norm(2).item() ** 2
                    total_grad_norm = total_grad_norm ** 0.5
                    writer.add_scalar('Training/Gradient_Norm', total_grad_norm, global_step)
                
                # Log model weights statistics every 50 steps
                if global_step % 50 == 0:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            writer.add_histogram(f'Weights/{name}', param.data, global_step)
                            if param.grad is not None:
                                writer.add_histogram(f'Gradients/{name}', param.grad.data, global_step)
            
            # Update progress bar
            if batch_idx % print_freq == 0 or batch_idx == len(train_loader) - 1:
                current_loss = train_loss / train_total
                current_acc = 100. * train_correct / train_total
                train_pbar.set_postfix({
                    'Batch Loss': f'{loss.item() * gradient_accumulation_steps:.6f}',
                    'Avg Loss': f'{current_loss:.6f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        train_loss /= train_total
        train_acc = 100. * train_correct / train_total
        train_pbar.close()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Initialize tqdm progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                if mixed_precision and scaler is not None:
                    with torch.amp.autocast(device_type=device):
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                
                val_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
                
                # Update validation progress bar
                val_pbar.set_postfix({
                    'Val Loss': f'{val_loss / val_total:.6f}',
                    'Val Acc': f'{100. * val_correct / val_total:.2f}%'
                })
        
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        val_pbar.close()
        
        # Update learning rate scheduler
        if scheduler_type.lower() == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time
        
        # Update metrics tracker
        metrics_tracker.update(train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time)
        
        # Enhanced TensorBoard logging for epochs
        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Validation_Loss', val_loss, epoch)
        writer.add_scalar('Epoch/Train_Accuracy', train_acc, epoch)
        writer.add_scalar('Epoch/Validation_Accuracy', val_acc, epoch)
        writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
        writer.add_scalar('Epoch/Training_Time', epoch_time, epoch)
        
        # Log epoch-level statistics
        writer.add_scalar('Epoch/Global_Step', global_step, epoch)
        
        # Log memory usage if on CUDA
        if device == 'cuda':
            writer.add_scalar('Memory/GPU_Allocated_GB', torch.cuda.memory_allocated() / 1024**3, epoch)
            writer.add_scalar('Memory/GPU_Reserved_GB', torch.cuda.memory_reserved() / 1024**3, epoch)
        
        # Legacy logging for backward compatibility
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%')
        print(f'  LR: {current_lr:.8f}, Time: {epoch_time:.2f}s')
        print('-' * 50)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_best_only:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, os.path.join(save_dir, f'{model_name}_best.pth'))
        
        # Save regular checkpoint
        if not save_best_only:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(save_dir, f'{model_name}_epoch_{epoch+1}.pth'))
        
        # Check early stopping
        if early_stopping(val_acc, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Close TensorBoard writer
    writer.close()
    
    # Plot and save metrics
    metrics_tracker.plot_metrics(os.path.join(save_dir, f'{model_name}_metrics.png'))
    
    return model, metrics_tracker


# =============================================================================
# Model Evaluation Function
# =============================================================================

def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'auto',
    num_classes: int = None,
    class_names: List[str] = None,
    save_dir: str = './evaluation',
    model_name: str = 'model',
    verbose: bool = True
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Comprehensive evaluation function for classification models.
    
    Args:
        model (nn.Module): Trained model to evaluate
        test_loader (DataLoader): Test data loader
        device (str): Device to run evaluation on
        num_classes (int): Number of classes (auto-detected if None)
        class_names (List[str]): Names of classes for reporting
        save_dir (str): Directory to save evaluation results
        model_name (str): Name for saving results
        verbose (bool): Whether to print detailed results
        
    Returns:
        Dict containing evaluation metrics and results
    """
    
    # Device setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    model.eval()
    
    if verbose:
        print(f"Evaluating model on device: {device}")
        print(f"Model: {type(model).__name__}")
        print("-" * 50)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect predictions and ground truth
    all_predictions = []
    all_probabilities = []
    all_targets = []
    test_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            # Get probabilities and predictions
            probabilities = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            # Store results
            all_predictions.extend(pred.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Calculate accuracy
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if verbose and batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(test_loader)} processed')
    
    evaluation_time = time.time() - start_time
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    # Auto-detect number of classes if not provided
    if num_classes is None:
        num_classes = len(np.unique(all_targets))
    
    # Generate class names if not provided
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    # Calculate metrics
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    # Top-k accuracy (if applicable)
    top5_accuracy = None
    if num_classes >= 5 and HAS_SKLEARN:
        top5_accuracy = top_k_accuracy_score(all_targets, all_probabilities, k=5) * 100
    
    # Classification report
    if HAS_SKLEARN:
        class_report = classification_report(
            all_targets, all_predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_targets, all_predictions)
    else:
        class_report = None
        conf_matrix = None
    
    # Results dictionary
    results = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'targets': all_targets,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'evaluation_time': evaluation_time,
        'num_samples': total,
        'num_classes': num_classes
    }
    
    # Print results
    if verbose:
        print(f"\nEvaluation Results:")
        print(f"Test Loss: {test_loss:.6f}")
        print(f"Accuracy: {accuracy:.2f}%")
        if top5_accuracy is not None:
            print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
        print(f"Evaluation Time: {evaluation_time:.2f} seconds")
        print(f"Samples Evaluated: {total}")
        
        # Print classification report
        if HAS_SKLEARN and class_report is not None:
            print(f"\nClassification Report:")
            print(classification_report(all_targets, all_predictions, target_names=class_names))
    
    # Save results
    results_file = os.path.join(save_dir, f'{model_name}_evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Model Evaluation Results\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Model: {type(model).__name__}\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        if top5_accuracy is not None:
            f.write(f"Top-5 Accuracy: {top5_accuracy:.2f}%\n")
        f.write(f"Evaluation Time: {evaluation_time:.2f} seconds\n")
        f.write(f"Samples: {total}\n")
        f.write(f"Classes: {num_classes}\n\n")
        
        if HAS_SKLEARN and class_report is not None:
            f.write("Classification Report:\n")
            f.write(classification_report(all_targets, all_predictions, target_names=class_names))
    
    # Plot and save confusion matrix
    if HAS_SKLEARN and conf_matrix is not None:
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        conf_matrix_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Confusion matrix saved to {conf_matrix_path}")
        plt.close()  # Close instead of show
    
    # Plot class-wise accuracy
    class_accuracies = []
    for i in range(num_classes):
        class_mask = all_targets == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(all_predictions[class_mask] == all_targets[class_mask]) * 100
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    plt.figure(figsize=(15, 6))
    bars = plt.bar(range(num_classes), class_accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Per-Class Accuracy - {model_name}')
    plt.xticks(range(num_classes), class_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    class_acc_path = os.path.join(save_dir, f'{model_name}_class_accuracies.png')
    plt.savefig(class_acc_path, dpi=300, bbox_inches='tight')
    if verbose:
        print(f"Class accuracies plot saved to {class_acc_path}")
    plt.close()  # Close instead of show
    
    return results


# =============================================================================
# Utility Functions
# =============================================================================

def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer = None, 
                   scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model (nn.Module): Model to load weights into
        optimizer (optim.Optimizer): Optimizer to load state
        scheduler: Learning rate scheduler to load state
        
    Returns:
        Dict: Checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'val_acc' in checkpoint:
        print(f"Validation Accuracy: {checkpoint['val_acc']:.2f}%")
    
    return checkpoint


def print_training_summary(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                          test_loader: DataLoader = None):
    """
    Print comprehensive training setup summary.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader  
        test_loader (DataLoader): Test data loader (optional)
    """
    from utils import count_parameters
    
    print("=" * 80)
    print("TRAINING SETUP SUMMARY")
    print("=" * 80)
    
    # Model information
    param_info = count_parameters(model)
    print(f"Model: {type(model).__name__}")
    print(f"Total Parameters: {param_info['total']:,}")
    print(f"Trainable Parameters: {param_info['trainable']:,}")
    
    # Dataset information
    print(f"\nDataset Information:")
    print(f"Training Samples: {len(train_loader.dataset):,}")
    print(f"Validation Samples: {len(val_loader.dataset):,}")
    if test_loader is not None:
        print(f"Test Samples: {len(test_loader.dataset):,}")
    
    print(f"Training Batches: {len(train_loader)}")
    print(f"Validation Batches: {len(val_loader)}")
    if test_loader is not None:
        print(f"Test Batches: {len(test_loader)}")
    
    print(f"Batch Size: {train_loader.batch_size}")
    
    # Sample batch information
    sample_batch = next(iter(train_loader))
    print(f"Input Shape: {sample_batch[0].shape}")
    print(f"Target Shape: {sample_batch[1].shape}")
    
    print("=" * 80)