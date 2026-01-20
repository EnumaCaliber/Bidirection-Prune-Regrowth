"""
Class-wise accuracy analysis utilities.

This module provides functions to compute and visualize:
- Per-class top-1 and top-5 accuracy
- Confusion matrices
- Class-wise performance comparisons between models
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os


def compute_classwise_accuracy(model, data_loader, device='cuda', num_classes=10, 
                               class_names=None, top_k=(1, 5)):
    """Compute per-class top-1 and top-5 accuracy.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader yielding (inputs, labels) pairs
        device: Device to run inference on ('cuda' or 'cpu')
        num_classes: Number of classes in the dataset
        class_names: Optional list/dict of class names for display
        top_k: Tuple of k values for top-k accuracy (default: (1, 5))
        
    Returns:
        Dictionary containing:
        - 'overall_top1': Overall top-1 accuracy (%)
        - 'overall_top5': Overall top-5 accuracy (%)
        - 'classwise_top1': Dict mapping class_id -> top-1 accuracy (%)
        - 'classwise_top5': Dict mapping class_id -> top-5 accuracy (%)
        - 'class_counts': Dict mapping class_id -> number of samples
        - 'predictions': List of all predictions
        - 'labels': List of all ground truth labels
        - 'confusion_matrix': Confusion matrix (num_classes x num_classes)
    """
    model.eval()
    model = model.to(device)
    
    # Initialize counters
    class_correct_top1 = {i: 0 for i in range(num_classes)}
    class_correct_top5 = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}
    
    all_predictions = []
    all_labels = []
    
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            # Top-1 accuracy
            _, pred_top1 = outputs.max(1)
            
            # Top-5 accuracy
            _, pred_top5 = outputs.topk(min(5, num_classes), 1, True, True)
            pred_top5 = pred_top5.t()
            
            # Overall accuracy
            total_samples += labels.size(0)
            total_correct_top1 += pred_top1.eq(labels).sum().item()
            
            # Top-5: check if true label is in top-5 predictions
            correct_top5 = pred_top5.eq(labels.view(1, -1).expand_as(pred_top5))
            total_correct_top5 += correct_top5.any(dim=0).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                
                # Top-1
                if pred_top1[i] == label:
                    class_correct_top1[label] += 1
                
                # Top-5
                if label in pred_top5[:, i]:
                    class_correct_top5[label] += 1
            
            # Store predictions and labels for confusion matrix
            all_predictions.extend(pred_top1.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute overall accuracy
    overall_top1 = 100.0 * total_correct_top1 / total_samples
    overall_top5 = 100.0 * total_correct_top5 / total_samples
    
    # Compute per-class accuracy
    classwise_top1 = {}
    classwise_top5 = {}
    
    for class_id in range(num_classes):
        if class_total[class_id] > 0:
            classwise_top1[class_id] = 100.0 * class_correct_top1[class_id] / class_total[class_id]
            classwise_top5[class_id] = 100.0 * class_correct_top5[class_id] / class_total[class_id]
        else:
            classwise_top1[class_id] = 0.0
            classwise_top5[class_id] = 0.0
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(num_classes)))
    
    results = {
        'overall_top1': overall_top1,
        'overall_top5': overall_top5,
        'classwise_top1': classwise_top1,
        'classwise_top5': classwise_top5,
        'class_counts': class_total,
        'predictions': all_predictions,
        'labels': all_labels,
        'confusion_matrix': cm,
    }
    
    return results


def print_classwise_accuracy(results, class_names=None, show_top5=True):
    """Pretty print class-wise accuracy results.
    
    Args:
        results: Output from compute_classwise_accuracy()
        class_names: Optional list/dict of class names
        show_top5: Whether to show top-5 accuracy (default: True)
    """
    num_classes = len(results['classwise_top1'])
    
    print("\n" + "="*80)
    print("ACCURACY ANALYSIS")
    print("="*80)
    
    # Overall accuracy
    print(f"\nOverall Top-1 Accuracy: {results['overall_top1']:.2f}%")
    if show_top5:
        print(f"Overall Top-5 Accuracy: {results['overall_top5']:.2f}%")
    
    # Per-class accuracy
    print("\n" + "-"*80)
    print("Per-Class Accuracy:")
    print("-"*80)
    
    if show_top5:
        print(f"{'Class':<20} {'Count':>8} {'Top-1 Acc':>12} {'Top-5 Acc':>12}")
        print("-"*80)
    else:
        print(f"{'Class':<20} {'Count':>8} {'Top-1 Acc':>12}")
        print("-"*80)
    
    for class_id in sorted(results['classwise_top1'].keys()):
        if class_names is not None:
            if isinstance(class_names, dict):
                class_name = class_names.get(class_id, f"Class {class_id}")
            else:
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        else:
            class_name = f"Class {class_id}"
        
        count = results['class_counts'][class_id]
        top1_acc = results['classwise_top1'][class_id]
        
        if show_top5:
            top5_acc = results['classwise_top5'][class_id]
            print(f"{class_name:<20} {count:>8} {top1_acc:>11.2f}% {top5_acc:>11.2f}%")
        else:
            print(f"{class_name:<20} {count:>8} {top1_acc:>11.2f}%")
    
    print("="*80 + "\n")


def plot_classwise_accuracy(results, class_names=None, title="Class-wise Accuracy", 
                           save_path=None, show_top5=True, figsize=(12, 6)):
    """Plot class-wise accuracy as bar charts.
    
    Args:
        results: Output from compute_classwise_accuracy()
        class_names: Optional list/dict of class names
        title: Title for the plot
        save_path: If provided, save plot to this path
        show_top5: Whether to show top-5 accuracy bars
        figsize: Figure size (width, height)
    """
    num_classes = len(results['classwise_top1'])
    class_ids = sorted(results['classwise_top1'].keys())
    
    # Prepare class names
    if class_names is not None:
        if isinstance(class_names, dict):
            labels = [class_names.get(i, f"Class {i}") for i in class_ids]
        else:
            labels = [class_names[i] if i < len(class_names) else f"Class {i}" for i in class_ids]
    else:
        labels = [f"Class {i}" for i in class_ids]
    
    # Get accuracy values
    top1_accs = [results['classwise_top1'][i] for i in class_ids]
    
    if show_top5:
        top5_accs = [results['classwise_top5'][i] for i in class_ids]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Top-1 accuracy
        ax1.bar(labels, top1_accs, color='steelblue', alpha=0.8)
        ax1.axhline(y=results['overall_top1'], color='red', linestyle='--', 
                   label=f"Overall: {results['overall_top1']:.2f}%")
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Top-1 Accuracy', fontsize=14)
        ax1.set_ylim([0, 105])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Top-5 accuracy
        ax2.bar(labels, top5_accs, color='seagreen', alpha=0.8)
        ax2.axhline(y=results['overall_top5'], color='red', linestyle='--',
                   label=f"Overall: {results['overall_top5']:.2f}%")
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Top-5 Accuracy', fontsize=14)
        ax2.set_ylim([0, 105])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.suptitle(title, fontsize=16, y=1.02)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.bar(labels, top1_accs, color='steelblue', alpha=0.8)
        ax.axhline(y=results['overall_top1'], color='red', linestyle='--',
                  label=f"Overall: {results['overall_top1']:.2f}%")
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_ylim([0, 105])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved class-wise accuracy plot to {save_path}")
    
    plt.show()


def plot_confusion_matrix(results, class_names=None, title="Confusion Matrix",
                         save_path=None, figsize=(10, 8), normalize=False):
    """Plot confusion matrix as a heatmap.
    
    Args:
        results: Output from compute_classwise_accuracy()
        class_names: Optional list/dict of class names
        title: Title for the plot
        save_path: If provided, save plot to this path
        figsize: Figure size (width, height)
        normalize: If True, normalize confusion matrix by row (true labels)
    """
    cm = results['confusion_matrix']
    num_classes = cm.shape[0]
    
    # Prepare class names
    if class_names is not None:
        if isinstance(class_names, dict):
            labels = [class_names.get(i, f"Class {i}") for i in range(num_classes)]
        else:
            labels = [class_names[i] if i < len(class_names) else f"Class {i}" 
                     for i in range(num_classes)]
    else:
        labels = [f"Class {i}" for i in range(num_classes)]
    
    # Normalize if requested
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
        title = f"{title} (Normalized)"
    else:
        cm_display = cm
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def compare_classwise_accuracy(results_list, model_names, class_names=None,
                               title="Model Comparison - Class-wise Accuracy",
                               save_path=None, show_top5=False, figsize=(14, 6)):
    """Compare class-wise accuracy across multiple models.
    
    Args:
        results_list: List of results from compute_classwise_accuracy()
        model_names: List of model names corresponding to results_list
        class_names: Optional list/dict of class names
        title: Title for the plot
        save_path: If provided, save plot to this path
        show_top5: Whether to show top-5 accuracy comparison
        figsize: Figure size (width, height)
    """
    if len(results_list) != len(model_names):
        raise ValueError("Length of results_list must match length of model_names")
    
    num_classes = len(results_list[0]['classwise_top1'])
    class_ids = sorted(results_list[0]['classwise_top1'].keys())
    
    # Prepare class names
    if class_names is not None:
        if isinstance(class_names, dict):
            labels = [class_names.get(i, f"Class {i}") for i in class_ids]
        else:
            labels = [class_names[i] if i < len(class_names) else f"Class {i}" 
                     for i in class_ids]
    else:
        labels = [f"Class {i}" for i in class_ids]
    
    # Set up bar positions
    x = np.arange(num_classes)
    width = 0.8 / len(results_list)
    
    if show_top5:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Top-1 comparison
        for i, (results, name) in enumerate(zip(results_list, model_names)):
            top1_accs = [results['classwise_top1'][c] for c in class_ids]
            offset = width * (i - len(results_list)/2 + 0.5)
            ax1.bar(x + offset, top1_accs, width, label=name, alpha=0.8)
        
        ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        ax1.set_title('Top-1 Accuracy Comparison', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 105])
        
        # Top-5 comparison
        for i, (results, name) in enumerate(zip(results_list, model_names)):
            top5_accs = [results['classwise_top5'][c] for c in class_ids]
            offset = width * (i - len(results_list)/2 + 0.5)
            ax2.bar(x + offset, top5_accs, width, label=name, alpha=0.8)
        
        ax2.set_ylabel('Top-5 Accuracy (%)', fontsize=12)
        ax2.set_title('Top-5 Accuracy Comparison', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0, 105])
        
        fig.suptitle(title, fontsize=16, y=1.02)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, (results, name) in enumerate(zip(results_list, model_names)):
            top1_accs = [results['classwise_top1'][c] for c in class_ids]
            offset = width * (i - len(results_list)/2 + 0.5)
            ax.bar(x + offset, top1_accs, width, label=name, alpha=0.8)
        
        ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved model comparison to {save_path}")
    
    plt.show()


def get_cifar10_class_names():
    """Return CIFAR-10 class names."""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']


def get_cifar100_class_names():
    """Return CIFAR-100 class names."""
    return [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
        'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
        'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]


# Example usage
if __name__ == "__main__":
    """
    Example usage:
    
    # Single model evaluation
    results = compute_classwise_accuracy(
        model=my_model,
        data_loader=test_loader,
        device='cuda',
        num_classes=10,
        class_names=get_cifar10_class_names()
    )
    
    # Print results
    print_classwise_accuracy(results, class_names=get_cifar10_class_names())
    
    # Plot class-wise accuracy
    plot_classwise_accuracy(
        results, 
        class_names=get_cifar10_class_names(),
        title="ResNet-20 Class-wise Accuracy",
        save_path="./results/classwise_acc.png"
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        results,
        class_names=get_cifar10_class_names(),
        save_path="./results/confusion_matrix.png",
        normalize=True
    )
    
    # Compare multiple models
    results_baseline = compute_classwise_accuracy(baseline_model, test_loader)
    results_pruned = compute_classwise_accuracy(pruned_model, test_loader)
    results_regrown = compute_classwise_accuracy(regrown_model, test_loader)
    
    compare_classwise_accuracy(
        results_list=[results_baseline, results_pruned, results_regrown],
        model_names=['Baseline', 'Pruned 50%', 'Regrown'],
        class_names=get_cifar10_class_names(),
        save_path="./results/model_comparison.png"
    )
    """
    pass
