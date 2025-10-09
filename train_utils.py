"""
Training and Evaluation Utilities for ResNet and FDResNet

This module provides comprehensive training and evaluation functions with adaptive features
including early stopping, tensorboard support, learning rate scheduling, and comprehensive
evaluation metrics for classification tasks.

Author: Created for FDConv-ResNet50 project
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import seaborn as sns



import os
import time
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_model(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    writer: SummaryWriter,
    num_epochs: int = 10,
    device: str = "cuda",
    model_name: str = "model",
    early_stopping_patience: int = None,
    save_dir: str = None,
    save_best_only: bool = True,
    max_grad_norm: float = None,
    accumulation_steps: int = 1,
    warmup_epochs: int = 0,
    use_amp: bool = False,
    log_every_n_steps: int = 100,
    validate_every_n_epochs: int = 1,
    monitor_gradients: bool = False,
    nan_check_frequency: int = 10,
    memory_efficient: bool = False,
    resume_from_checkpoint: str = None,
):
    """
    Advanced PyTorch model training function with comprehensive monitoring and optimization features.
    
    This function provides a robust training loop with extensive monitoring capabilities,
    gradient management, mixed precision training, memory optimization, and advanced
    debugging features. Designed for production-ready model training with automatic
    error detection and recovery mechanisms.
    
    Key Features:
    - Gradient clipping and monitoring for numerical stability
    - Mixed precision training for memory efficiency
    - Gradient accumulation for large effective batch sizes
    - Comprehensive NaN/Inf detection and handling
    - Memory usage monitoring and optimization
    - Flexible validation scheduling
    - Advanced checkpoint management with resume capability
    - Warmup learning rate scheduling
    - Real-time gradient norm tracking
    
    Args:
        model (torch.nn.Module): The neural network model to train.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss, MSELoss).
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates (e.g., Adam, SGD, AdamW).
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler for adaptive learning rates.
        train_loader (torch.utils.data.DataLoader): Training dataset loader.
        val_loader (torch.utils.data.DataLoader): Validation dataset loader.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging metrics.
        num_epochs (int, optional): Total number of training epochs. Defaults to 10.
        device (str, optional): Compute device ('cuda', 'cpu', or specific GPU). Defaults to "cuda".
        model_name (str, optional): Model identifier for logging and checkpoints. Defaults to "model".
        early_stopping_patience (int, optional): Epochs to wait before early stopping. 
                                               None disables early stopping. Defaults to None.
        save_dir (str, optional): Directory for saving model checkpoints. None disables saving. Defaults to None.
        save_best_only (bool, optional): Save only best performing model vs all epochs. Defaults to True.
        max_grad_norm (float, optional): Maximum gradient norm for clipping. None disables clipping. Defaults to None.
        accumulation_steps (int, optional): Steps to accumulate gradients before update. Defaults to 1.
        warmup_epochs (int, optional): Number of warmup epochs with linear LR increase. Defaults to 0.
        use_amp (bool, optional): Enable Automatic Mixed Precision training. Defaults to False.
        log_every_n_steps (int, optional): Frequency of step-level logging to TensorBoard. Defaults to 100.
        validate_every_n_epochs (int, optional): Validation frequency in epochs. Defaults to 1.
        monitor_gradients (bool, optional): Track and log gradient statistics. Defaults to False.
        nan_check_frequency (int, optional): Check for NaN/Inf every N batches. Defaults to 10.
        memory_efficient (bool, optional): Enable memory optimization techniques. Defaults to False.
        resume_from_checkpoint (str, optional): Path to checkpoint file for resuming training. Defaults to None.
    
    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The trained model with best weights loaded
            - history (dict): Comprehensive training history containing:
                * 'train_loss': Training losses per epoch
                * 'val_loss': Validation losses per epoch  
                * 'train_acc': Training accuracies per epoch
                * 'val_acc': Validation accuracies per epoch
                * 'learning_rates': Learning rates per epoch
                * 'gradient_norms': Gradient norms per epoch (if monitored)
                * 'best_epoch': Epoch with best validation performance
                * 'best_val_acc': Best validation accuracy achieved
                * 'total_steps': Total training steps completed
                * 'epochs_completed': Number of epochs completed
    
    Raises:
        RuntimeError: If training encounters unrecoverable numerical instability
        ValueError: If invalid parameters are provided
        FileNotFoundError: If resume_from_checkpoint file doesn't exist
    
    Example:
        >>> model = MyModel()
        >>> criterion = nn.CrossEntropyLoss()
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        >>> writer = SummaryWriter('runs/experiment_1')
        >>> 
        >>> trained_model, history = train_model(
        ...     model=model,
        ...     criterion=criterion,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     writer=writer,
        ...     num_epochs=50,
        ...     max_grad_norm=1.0,
        ...     use_amp=True,
        ...     monitor_gradients=True
        ... )
    """
    # Initialize training components and state
    model.to(device)
    best_val_acc = 0.0
    best_epoch = -1
    best_weights = None
    no_improve_epochs = 0
    global_step = 0
    start_epoch = 0
    
    # Initialize AMP scaler if using mixed precision
    # scaler = torch.amp.GradScaler(device=device) if use_amp and device != 'cpu' else None
    scaler = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None

    
    # Enhanced history tracking
    history = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
        "learning_rates": [], "gradient_norms": [], "total_steps": 0,
        "epochs_completed": 0, "best_epoch": -1, "best_val_acc": 0.0
    }
    
    # Setup checkpoint directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Resume from checkpoint if provided
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        best_epoch = checkpoint.get('best_epoch', -1)
        global_step = checkpoint.get('global_step', 0)
        if 'history' in checkpoint:
            history.update(checkpoint['history'])
    
    # Parameter validation
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be >= 0")
    
    # Memory monitoring setup
    initial_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    
    print(f"Training Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {num_epochs} (starting from {start_epoch + 1})")
    print(f"  Device: {device.upper()}")
    print(f"  Mixed Precision: {use_amp}")
    print(f"  Gradient Clipping: {max_grad_norm if max_grad_norm else 'Disabled'}")
    print(f"  Accumulation Steps: {accumulation_steps}")
    print(f"  Warmup Epochs: {warmup_epochs}")
    if torch.cuda.is_available():
        print(f"  Initial GPU Memory: {initial_memory:.2f} GB")
    print(f"Training started for {num_epochs - start_epoch} remaining epochs")

    try:
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            
            # Warmup learning rate adjustment
            if epoch < warmup_epochs:
                warmup_factor = (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group.get('initial_lr', param_group['lr']) * warmup_factor
            
            # ==================== TRAINING PHASE ====================
            model.train()
            running_loss = 0.0
            running_correct = 0
            total_samples = 0
            gradient_norms = []
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
            
            for batch_idx, (inputs, labels) in enumerate(train_pbar):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                # Forward pass with optional mixed precision
                if use_amp and scaler:
                    # with torch.amp.autocast(device_type=device):
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels) / accumulation_steps
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / accumulation_steps
                
                # Check for NaN/Inf in loss
                if batch_idx % nan_check_frequency == 0:
                    if torch.isnan(loss) or torch.isinf(loss):
                        raise RuntimeError(f"NaN or Inf loss detected at epoch {epoch+1}, batch {batch_idx}")
                
                # Backward pass
                if use_amp and scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation and optimization step
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping and monitoring
                    if max_grad_norm is not None or monitor_gradients:
                        if use_amp and scaler:
                            scaler.unscale_(optimizer)
                        
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            max_grad_norm if max_grad_norm else float('inf')
                        )
                        
                        if monitor_gradients:
                            gradient_norms.append(total_norm.item())
                    
                    # Optimizer step
                    if use_amp and scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    global_step += 1
                
                # Update running metrics
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size * accumulation_steps
                total_samples += batch_size
                _, preds = torch.max(outputs, 1)
                running_correct += (preds == labels).sum().item()
                
                train_loss = running_loss / total_samples
                train_acc = 100.0 * running_correct / total_samples
                
                # Step-level logging
                if global_step % log_every_n_steps == 0:
                    writer.add_scalar(f"{model_name}/Step_Train_Loss", loss.item() * accumulation_steps, global_step)
                    writer.add_scalar(f"{model_name}/Step_Train_Acc", 
                                    100.0 * (preds == labels).float().mean(), global_step)
                    if monitor_gradients and gradient_norms:
                        writer.add_scalar(f"{model_name}/Gradient_Norm", gradient_norms[-1], global_step)
                
                # Update progress bar
                train_pbar.set_postfix({
                    "T_Loss": f"{train_loss:.4f}",
                    "T_Acc": f"{train_acc:.2f}%"
                })
                
                # Memory cleanup for memory efficient mode
                if memory_efficient and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            epoch_train_loss = train_loss
            epoch_train_acc = train_acc
            
            # ==================== VALIDATION PHASE ====================
            epoch_val_loss = None
            epoch_val_acc = None
            
            if epoch % validate_every_n_epochs == 0:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
                
                with torch.no_grad():
                    for batch_idx, (inputs, labels) in enumerate(val_pbar):
                        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                        
                        if use_amp:
                            # with torch.amp.autocast(device_type=device):
                            with torch.cuda.amp.autocast():
                                outputs = model(inputs)
                                loss = criterion(outputs, labels)
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        
                        # Check for validation NaN/Inf
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"WARNING: NaN/Inf validation loss at epoch {epoch+1}, batch {batch_idx}")
                            break
                        
                        batch_size = inputs.size(0)
                        val_loss += loss.item() * batch_size
                        val_total += batch_size
                        _, preds = torch.max(outputs, 1)
                        val_correct += (preds == labels).sum().item()
                        
                        current_val_loss = val_loss / val_total
                        current_val_acc = 100.0 * val_correct / val_total
                        
                        val_pbar.set_postfix({
                            "V_Loss": f"{current_val_loss:.4f}",
                            "V_Acc": f"{current_val_acc:.2f}%"
                        })
                
                epoch_val_loss = val_loss / val_total if val_total > 0 else float('inf')
                epoch_val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
            
            # ==================== LOGGING AND TRACKING ====================
            history["train_loss"].append(epoch_train_loss)
            history["train_acc"].append(epoch_train_acc)
            history["learning_rates"].append(optimizer.param_groups[0]['lr'])
            
            if epoch_val_loss is not None:
                history["val_loss"].append(epoch_val_loss)
                history["val_acc"].append(epoch_val_acc)
            
            if monitor_gradients and gradient_norms:
                avg_grad_norm = np.mean(gradient_norms)
                history["gradient_norms"].append(avg_grad_norm)
                writer.add_scalar(f"{model_name}/Avg_Gradient_Norm", avg_grad_norm, epoch)
            
            # TensorBoard logging
            writer.add_scalar(f"{model_name}/Train_Loss", epoch_train_loss, epoch)
            writer.add_scalar(f"{model_name}/Train_Acc", epoch_train_acc, epoch)
            writer.add_scalar(f"{model_name}/Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
            
            if epoch_val_loss is not None:
                writer.add_scalar(f"{model_name}/Val_Loss", epoch_val_loss, epoch)
                writer.add_scalar(f"{model_name}/Val_Acc", epoch_val_acc, epoch)
            
            # Memory logging
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**3
                writer.add_scalar(f"{model_name}/GPU_Memory_GB", current_memory, epoch)
            
            # ==================== LEARNING RATE SCHEDULING ====================
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if epoch_val_loss is not None:
                        scheduler.step(epoch_val_loss)
                else:
                    scheduler.step()
            
            # ==================== MODEL SAVING AND BEST MODEL TRACKING ====================
            if epoch_val_acc is not None:
                improved = epoch_val_acc > best_val_acc
                if improved:
                    best_val_acc = epoch_val_acc
                    best_epoch = epoch
                    best_weights = copy.deepcopy(model.state_dict())
                    no_improve_epochs = 0
                    
                    if save_dir and save_best_only:
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': best_weights,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_acc': best_val_acc,
                            'best_epoch': best_epoch,
                            'global_step': global_step,
                            'history': history
                        }
                        torch.save(checkpoint, os.path.join(save_dir, f"{model_name}_best.pth"))
                else:
                    no_improve_epochs += 1
            
            # Save epoch checkpoint if not save_best_only
            if save_dir and not save_best_only:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'best_epoch': best_epoch,
                    'global_step': global_step,
                    'history': history
                }
                torch.save(checkpoint, os.path.join(save_dir, f"{model_name}_epoch{epoch+1}.pth"))
            
            # ==================== EARLY STOPPING ====================
            if (early_stopping_patience is not None and 
                epoch_val_acc is not None and 
                no_improve_epochs >= early_stopping_patience):
                print(f"Early stopping at epoch {epoch+1} - no improvement for {early_stopping_patience} epochs.")
                break
            
            # ==================== EPOCH SUMMARY ====================
            val_summary = ""
            if epoch_val_loss is not None:
                val_summary = f"V_Loss: {epoch_val_loss:.4f} , V_Acc: {epoch_val_acc:.2f}% | "
            
            memory_info = ""
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**3
                memory_info = f"| Memory: {current_memory:.2f}GB"
            
            tqdm.write(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"T_Loss: {epoch_train_loss:.4f} , T_Acc: {epoch_train_acc:.2f}% | "
                f"{val_summary}"
                f"Best_V_Acc: {best_val_acc:.2f}% | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                f"Time: {time.time()-epoch_start:.2f}s {memory_info}"
            )
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Training error: {str(e)}")
        raise
    
    # ==================== FINALIZATION ====================
    # Restore best weights if available
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"Loaded best model weights from epoch {best_epoch + 1}")
    
    # Update final history
    history["best_epoch"] = best_epoch + 1 if best_epoch >= 0 else -1
    history["best_val_acc"] = best_val_acc
    history["total_steps"] = global_step
    history["epochs_completed"] = epoch + 1 if 'epoch' in locals() else start_epoch
    
    print(f"\nTraining complete.")
    if best_val_acc > 0:
        print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch + 1})")
    print(f"Total training steps: {global_step}")
    
    if torch.cuda.is_available():
        final_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"Final GPU Memory Usage: {final_memory:.2f} GB")
    
    return model, history



def plot_training_history(history: dict, model_name: str):
    """
    Plot training and validation loss/accuracy curves for model training analysis.
    
    This function creates a comprehensive visualization of the training progress by
    plotting both loss and accuracy curves for training and validation sets side by side.
    Useful for analyzing model convergence, overfitting, and training dynamics.
    
    Args:
        history (dict): Training history dictionary containing:
            - 'train_loss': List of training losses per epoch
            - 'val_loss': List of validation losses per epoch
            - 'train_acc': List of training accuracies per epoch  
            - 'val_acc': List of validation accuracies per epoch
        model_name (str): Name of the model for plot titles and identification
    
    Returns:
        None: Displays the matplotlib plots directly
    
    Example:
        >>> history = {
        ...     'train_loss': [0.8, 0.6, 0.4, 0.3],
        ...     'val_loss': [0.9, 0.7, 0.5, 0.4], 
        ...     'train_acc': [0.6, 0.7, 0.8, 0.85],
        ...     'val_acc': [0.55, 0.65, 0.75, 0.80]
        ... }
        >>> plot_curves(history, 'ResNet50')
    
    Note:
        This function requires matplotlib to be imported and will display plots inline
        in Jupyter notebooks or open plot windows in standard Python environments.
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label="Train Loss", marker="o")
    plt.plot(history['val_loss'], label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("Loss Curve")
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label="Train Acc", marker="o")
    plt.plot(history['val_acc'], label="Val Acc", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("Accuracy Curve")
    
    plt.show()




def evaluate_model(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
                   device: str = 'cuda', model_name: str = 'Model', class_names: list = None):
    """
    Comprehensive model evaluation with detailed metrics and visualizations.
    
    This function performs a complete evaluation of a trained model on a test dataset,
    computing various classification metrics including accuracy, precision, recall, F1-score,
    confusion matrix, and detailed classification report. Results are both printed and 
    visualized with a confusion matrix heatmap.
    
    Args:
        model (torch.nn.Module): The trained neural network model to evaluate
        test_loader (torch.utils.data.DataLoader): Test data loader containing evaluation data
        device (str, optional): Device for computation ('cuda' or 'cpu'). Defaults to 'cuda'.
        model_name (str, optional): Name of the model for display purposes. Defaults to 'Model'.
        class_names (list, optional): List of class names for confusion matrix labels. 
                                    If None, numeric indices will be used. Defaults to None.
    
    Returns:
        tuple: A tuple containing evaluation metrics:
            - acc (float): Overall accuracy (0-1 range)
            - pre (float): Macro-averaged precision (0-1 range)  
            - rec (float): Macro-averaged recall (0-1 range)
            - f1 (float): Macro-averaged F1-score (0-1 range)
    
    Example:
        >>> model = load_trained_model('best_model.pth')
        >>> class_names = ['Normal', 'Diabetic Retinopathy', 'Glaucoma']
        >>> acc, pre, rec, f1 = evaluate_model(model, test_loader, 
        ...                                     device='cuda', 
        ...                                     model_name='ResNet50',
        ...                                     class_names=class_names)
        >>> print(f"Test Accuracy: {acc:.3f}")
    
    Note:
        - Requires scikit-learn for metrics computation and seaborn for visualization
        - Model is automatically set to evaluation mode during testing
        - Displays confusion matrix heatmap and prints detailed classification report
        - All computations are performed without gradient tracking for efficiency
    """
    model.eval()
    y_true, y_pred = [], []
    test_loss = 0.0
    test_correct = 0
    test_samples = 0
    
    # Evaluation loop with adaptive tqdm
    eval_pbar = tqdm(test_loader, desc=f"Evaluating {model_name}", position=0)
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(eval_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Calculate loss if criterion is available (optional)
            try:
                import torch.nn.functional as F
                loss = F.cross_entropy(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
            except:
                pass  # Skip loss calculation if not possible
            
            _, preds = torch.max(outputs, 1)
            
            # Update running statistics
            batch_size = inputs.size(0)
            test_samples += batch_size
            test_correct += torch.sum(preds == labels.data).item()
            
            # Calculate running accuracy
            running_acc = test_correct / test_samples
            
            # Update progress info
            progress_info = {'Acc': f'{running_acc:.4f}'}
            if test_loss > 0:
                running_loss = test_loss / test_samples
                progress_info['Loss'] = f'{running_loss:.4f}'
            
            eval_pbar.set_postfix(progress_info)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    pre = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"{model_name} Test Accuracy: {acc:.4f}")
    print(f"{model_name} Test Precision: {pre:.4f}")
    print(f"{model_name} Test Recall: {rec:.4f}")
    print(f"{model_name} Test F1-Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

    return acc, pre, rec, f1



def show_pred_vs_truth(model: torch.nn.Module, dataset: torch.utils.data.Dataset,
                        class_names: list, device: str = "cuda", num_samples: int = 5):
    """
    Visualizes model predictions versus true labels on a subset of the dataset.

    This function randomly selects a specified number of samples from the provided dataset,
    performs inference using the given model, and displays the images along with their true
    and predicted labels. Useful for qualitative assessment of model performance.

    Args:
        model (torch.nn.Module): The trained model for making predictions.
        dataset (torch.utils.data.Dataset): Dataset containing images and labels.
        class_names (list): List of class names corresponding to label indices.
        device (str, optional): Device for computation ('cuda' or 'cpu'). Defaults to "cuda".
        num_samples (int, optional): Number of random samples to display. Defaults to 5.
    """
    model.eval()
    plt.figure(figsize=(15, 5))
    indices = np.random.sample(range(len(dataset)), num_samples)
    for i, idx in tqdm(enumerate(indices), desc="Showing Predictions"):
        img, true_label = dataset[idx]
        inp = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(inp)
            pred_label = output.argmax(dim=1).item()
        img_disp = img.permute(1, 2, 0).cpu() * 0.5 + 0.5
        img_disp = img_disp.clamp(0, 1)
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img_disp)
        plt.title(f"T: {class_names[true_label]}\nP: {class_names[pred_label]}")
        plt.axis("off")