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
):
    """
    Train a PyTorch model with validation monitoring, TensorBoard logging, and early stopping.
    
    This function implements a complete training loop with epoch-wise progress tracking,
    automatic validation evaluation, learning rate scheduling, early stopping functionality,
    and comprehensive logging to TensorBoard for monitoring training progress. Includes
    automatic best model saving based on validation performance.
    
    Args:
        model (torch.nn.Module): The neural network model to train.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates (e.g., Adam, SGD).
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler for adaptive LR.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging.
        num_epochs (int, optional): Number of training epochs. Defaults to 10.
        device (str, optional): Device to run training on ('cuda' or 'cpu'). Defaults to "cuda".
        model_name (str, optional): Name prefix for TensorBoard logging. Defaults to "model".
        early_stopping_patience (int, optional): Number of epochs to wait for improvement 
                                               before stopping. If None, no early stopping. Defaults to None.
        save_dir (str, optional): Directory to save model checkpoints. If None, no saving. Defaults to None.
        save_best_only (bool, optional): If True, only save the best model. If False, save every epoch.
                                        Defaults to True.
    
    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The trained model (loaded with best weights if early stopping used)
            - history (dict): Dictionary with training history containing:
                * 'train_loss': List of training losses per epoch
                * 'val_loss': List of validation losses per epoch  
                * 'train_acc': List of training accuracies per epoch
                * 'val_acc': List of validation accuracies per epoch
                * 'best_epoch': Epoch number with best validation performance
                * 'best_val_acc': Best validation accuracy achieved
    """
    model.to(device)
    best_val_acc = 0.0
    best_epoch = -1
    best_weights = None
    no_improve_epochs = 0
    global_step = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"Training started for {num_epochs} epochs on {device.upper()}")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # ==================== TRAIN ====================
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update running metrics
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()

            train_loss = running_loss / total_samples
            train_acc = 100.0 * running_correct / total_samples

            # Per-step logging (TensorBoard only)
            global_step += 1
            writer.add_scalar(f"{model_name}/Step_Train_Loss", loss.item(), global_step)
            writer.add_scalar(f"{model_name}/Step_Train_Acc", 100.0 * (preds == labels).float().mean(), global_step)

            # Update progress bar
            train_pbar.set_postfix({
                "T_Loss": f"{train_loss:.4f}",
                "T_Acc": f"{train_acc:.2f}%"
            })

        epoch_train_loss = train_loss
        epoch_train_acc = train_acc

        # ==================== VALIDATION ====================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

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

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total

        # ==================== LOGGING (Epoch) ====================
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        writer.add_scalar(f"{model_name}/Train_Loss", epoch_train_loss, epoch)
        writer.add_scalar(f"{model_name}/Train_Acc", epoch_train_acc, epoch)
        writer.add_scalar(f"{model_name}/Val_Loss", epoch_val_loss, epoch)
        writer.add_scalar(f"{model_name}/Val_Acc", epoch_val_acc, epoch)

        # ==================== SCHEDULER ====================
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        # ==================== MODEL SAVING ====================
        improved = epoch_val_acc > best_val_acc
        if improved:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            best_weights = model.state_dict()
            no_improve_epochs = 0
            if save_dir and save_best_only:
                torch.save(best_weights, os.path.join(save_dir, f"{model_name}_best.pth"))
        else:
            no_improve_epochs += 1

        if save_dir and not save_best_only:
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_epoch{epoch+1}.pth"))

        # ==================== EARLY STOPPING ====================
        if early_stopping_patience is not None and no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1} — no improvement for {early_stopping_patience} epochs.")
            break

        # ==================== SUMMARY ====================
        tqdm.write(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"T_Loss: {epoch_train_loss:.4f} , T_Acc: {epoch_train_acc:.2f}% | "
            f"V_Loss: {epoch_val_loss:.4f} , V_Acc: {epoch_val_acc:.2f}% | "
            f"Best_V_Acc: {best_val_acc:.2f}% | Time: {time.time()-epoch_start:.2f}s\n"
        )

    # Restore best weights if available
    if best_weights is not None:
        model.load_state_dict(best_weights)

    history["best_epoch"] = best_epoch + 1
    history["best_val_acc"] = best_val_acc

    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch+1})")

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