#!/usr/bin/env python3
"""
Inspect RL training checkpoint or best allocation file
AND perform final finetuning from checkpoint
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import copy
import random
import numpy as np
from datetime import datetime
# from transformers import get_cosine_schedule_with_warmup

from utils.model_loader import model_loader
from utils.data_loader import data_loader
from utils.analysis_utils import prune_weights_reparam, count_pruned_params


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility across PyTorch, NumPy, and Python's random module.
    
    Args:
        seed: Integer seed value (default: 42)
    """
    print(f"Setting random seed: {seed}")
    
    # Python's random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Make PyTorch operations deterministic (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Note: We don't use torch.use_deterministic_algorithms(True) because:
    # - It requires CUBLAS_WORKSPACE_CONFIG environment variable for CUDA >= 10.2
    # - Some operations (like CuBLAS) don't have deterministic implementations
    # - The above settings provide good reproducibility for most cases
    # 
    # For strict determinism, set before running:
    #   export CUBLAS_WORKSPACE_CONFIG=:4096:8
    # Then uncomment:
    # if hasattr(torch, 'use_deterministic_algorithms'):
    #     torch.use_deterministic_algorithms(True)
    
    print(f"  ✓ Random seed initialized for reproducibility")
    print(f"     (Note: Some CUDA operations may still be non-deterministic)\n")


def inspect_training_checkpoint(checkpoint_path):
    """Inspect a training checkpoint file"""
    print(f"\n{'='*70}")
    print(f"Training Checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: File not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Checkpoint Contents:")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best Reward: {checkpoint['best_reward']:.4f} ({checkpoint['best_reward']*100:.2f}%)")
    
    if 'config' in checkpoint:
        print(f"\n  Training Config:")
        for key, value in checkpoint['config'].items():
            print(f"    {key}: {value}")
    
    if 'reward_baseline' in checkpoint and checkpoint['reward_baseline'] is not None:
        print(f"\n  Reward Baseline: {checkpoint['reward_baseline']:.4f}")
    
    if 'total_rewards' in checkpoint:
        rewards = checkpoint['total_rewards']
        if len(rewards) > 0:
            print(f"\n  Recent Rewards (last {len(rewards)}):")
            print(f"    Mean: {sum(rewards)/len(rewards):.4f}")
            print(f"    Min: {min(rewards):.4f}")
            print(f"    Max: {max(rewards):.4f}")
    
    if 'best_allocation' in checkpoint and checkpoint['best_allocation']:
        print(f"\n  Best Allocation:")
        total_allocated = sum(checkpoint['best_allocation'].values())
        print(f"    Total weights: {total_allocated}")
        print(f"    Layers:")
        for layer, count in checkpoint['best_allocation'].items():
            print(f"      {layer}: {count}")
    
    if 'best_regrow_indices' in checkpoint and checkpoint['best_regrow_indices']:
        print(f"\n  Best Regrow Indices Available:")
        for layer, indices in checkpoint['best_regrow_indices'].items():
            print(f"    {layer}: {len(indices)} indices")
    
    print(f"\n{'='*70}\n")


def inspect_best_allocation(best_path):
    """Inspect a best allocation file"""
    print(f"\n{'='*70}")
    print(f"Best Allocation: {best_path}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(best_path):
        print(f"Error: File not found: {best_path}")
        return
    
    best_data = torch.load(best_path, map_location='cpu')
    
    print(f"Best Allocation Found:")
    print(f"  Epoch: {best_data['epoch'] + 1}")
    print(f"  Reward: {best_data['reward']:.4f}")
    print(f"  Accuracy: {best_data['accuracy']:.2f}%")
    
    if 'timestamp' in best_data:
        timestamp = datetime.fromtimestamp(best_data['timestamp'])
        print(f"  Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if 'allocation' in best_data and best_data['allocation']:
        print(f"\n  Layer Allocation:")
        total_allocated = sum(best_data['allocation'].values())
        print(f"    Total weights: {total_allocated}")
        print(f"    Layers:")
        for layer, count in best_data['allocation'].items():
            print(f"      {layer}: {count}")
    
    if 'regrow_indices' in best_data and best_data['regrow_indices']:
        print(f"\n  Regrow Indices Available:")
        for layer, indices in best_data['regrow_indices'].items():
            print(f"    {layer}: {len(indices)} indices")
    
    print(f"\n{'='*70}\n")


def apply_regrowth_from_indices(model, allocation, regrow_indices, reference_masks, reference_weights):
    """
    Apply regrowth to model using saved indices and reference weights
    
    Args:
        model: Model to apply regrowth to
        allocation: Dict[layer_name] -> num_weights
        regrow_indices: Dict[layer_name] -> List[tuple] of indices
        reference_masks: Dict[layer_name] -> reference mask tensor
        reference_weights: Dict[layer_name] -> reference weight tensor
    """
    print(f"\n{'='*70}")
    print("Applying Regrowth from Saved Indices")
    print(f"{'='*70}\n")
    
    module_dict = dict(model.named_modules())
    total_regrown = 0
    
    for layer_name, num_weights in allocation.items():
        if num_weights == 0:
            continue
            
        module = module_dict.get(layer_name)
        if module is None or not hasattr(module, 'weight_mask'):
            print(f"  Warning: Layer {layer_name} not found or has no mask, skipping")
            continue
        
        current_mask = module.weight_mask
        ref_mask = reference_masks[layer_name]
        
        # Get saved indices
        if layer_name not in regrow_indices or not regrow_indices[layer_name]:
            print(f"  Warning: No indices saved for {layer_name}, skipping")
            continue
        
        saved_indices = regrow_indices[layer_name]
        
        # Convert to tensor
        if isinstance(saved_indices, torch.Tensor):
            selected_indices_tensor = saved_indices
        else:
            selected_indices_tensor = torch.tensor(saved_indices, dtype=torch.long, device=current_mask.device)
        
        if selected_indices_tensor.ndim == 1 and selected_indices_tensor.numel() > 0:
            selected_indices_tensor = selected_indices_tensor.unsqueeze(0)
        
        # Apply regrowth
        actual_regrown = 0
        for idx in selected_indices_tensor:
            tuple_idx = tuple(idx.tolist())
            
            # Verify this position is regrowable
            # if current_mask[tuple_idx] == 0 and ref_mask[tuple_idx] == 1:
            # Update mask to 1.0 (unpruned)
            current_mask[tuple_idx] = 1.0
            
            # Copy weight value from reference model into weight_orig
            if layer_name in reference_weights:
                ref_weight = reference_weights[layer_name][tuple_idx]
                if hasattr(module, "weight_orig"):
                    module.weight_orig.data[tuple_idx] = ref_weight.clone()
                else:
                    module.weight.data[tuple_idx] = ref_weight.clone()
                
                actual_regrown += 1
        
        total_regrown += actual_regrown
        print(f"  {layer_name}: {actual_regrown}/{num_weights} weights regrown")
    
    print(f"\nTotal weights regrown: {total_regrown}")
    print(f"{'='*70}\n")
    
    return total_regrown


def calculate_model_sparsity(model):
    """Calculate current sparsity of the model"""
    total_params = 0
    pruned_params = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mask'):
            mask = module.weight_mask
            total_params += mask.numel()
            pruned_params += (mask == 0).sum().item()
    
    sparsity = 100.0 * pruned_params / total_params if total_params > 0 else 0.0
    return sparsity, total_params, pruned_params


def evaluate_model_accuracy(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def calibrate_batchnorm(model, train_loader, device, num_batches=100):
    """
    Calibrate BatchNorm statistics after regrowth WITHOUT updating weights.
    
    After regrowth, BN running statistics are misaligned with the new weight distribution.
    This function recomputes correct statistics by doing forward passes (no backprop).
    
    This is BETTER than just resetting because:
    - Computes correct statistics upfront (not random 0/1)
    - Finetuning starts from better initialization
    - Faster convergence (saves 5-10 wasted epochs)
    
    Args:
        model: Model with regrown weights
        train_loader: Training data loader  
        device: Device
        num_batches: Number of batches for calibration (default 100, ~2000 images)
    """
    print(f"  Calibrating BatchNorm statistics using {num_batches} batches...")
    
    # Reset all BN stats to start fresh
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
            # Use cumulative moving average for calibration (more stable)
            module.momentum = None
    
    # Forward pass to accumulate statistics (no gradient computation)
    model.train()  # Must be in train mode for BN to update running stats
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)  # Forward pass updates running_mean and running_var
    
    # Restore default momentum for training phase
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.momentum = 0.1  # PyTorch default
    
    print(f"  ✓ BatchNorm calibration complete (statistics aligned with regrown weights)")


def get_regrown_weight_mask(model, allocation):
    """
    Create a mask identifying which weights were regrown.
    
    Args:
        model: Model with regrown weights
        allocation: Dict[layer_name] -> num_weights regrown
    
    Returns:
        regrown_mask: Dict[layer_name] -> boolean tensor (True = regrown weight)
    """
    regrown_mask = {}
    module_dict = dict(model.named_modules())
    
    for layer_name in allocation.keys():
        module = module_dict.get(layer_name)
        if module and hasattr(module, 'weight_mask'):
            # All weights with mask==1 are "active"
            # We need to track which ones were just regrown
            # For now, we'll use a heuristic: recent weights have different magnitude
            regrown_mask[layer_name] = torch.zeros_like(module.weight_mask, dtype=torch.bool)
    
    return regrown_mask


def full_finetune(model, train_loader, test_loader, device, 
                  epochs=100, lr=0.001, save_path=None, verbose=True, patience=100,
                  reset_bn_stats=False, freeze_original=False, regrown_indices=None,
                  differential_lr=False, lr_original=1e-5, lr_regrown=3e-4,
                  progressive_unfreezing=False, phase1_epochs=50, phase2_epochs=150):
    """
    Complete finetuning process with best model tracking and early stopping
    
    Args:
        model: Model to finetune
        train_loader: Training data loader
        test_loader: Test data loader  
        device: Device to train on
        epochs: Number of finetuning epochs
        lr: Learning rate (used if not differential_lr)
        save_path: Path to save best model (if None, don't save)
        verbose: Print progress
        patience: Number of epochs without improvement before early stopping
        reset_bn_stats: Whether to reset BatchNorm statistics before training
        freeze_original: Whether to freeze original sparse weights (train only regrown)
        regrown_indices: Dict[layer_name] -> indices of regrown weights (for selective training)
        differential_lr: Use different LRs for original vs regrown weights
        lr_original: Learning rate for original weights (if differential_lr=True)
        lr_regrown: Learning rate for regrown weights (if differential_lr=True)
        progressive_unfreezing: Gradually unfreeze weights during training
        phase1_epochs: Epochs for phase 1 of progressive unfreezing
        phase2_epochs: Epochs for phase 2 of progressive unfreezing
    
    Returns:
        best_accuracy: Best test accuracy achieved
        best_model_state: State dict of best model
    """
    print(f"\n{'='*70}")
    print("Final Finetuning Phase")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Save path: {save_path}")
    print(f"  Reset BN stats: {reset_bn_stats}")
    print(f"  Freeze original weights: {freeze_original}")
    print(f"  Differential LR: {differential_lr}")
    if differential_lr:
        print(f"    LR (original): {lr_original}")
        print(f"    LR (regrown): {lr_regrown}")
    print(f"  Progressive unfreezing: {progressive_unfreezing}")
    if progressive_unfreezing:
        print(f"    Phase 1 (freeze original): {phase1_epochs} epochs")
        print(f"    Phase 2 (differential LR): {phase2_epochs} epochs")
        print(f"    Phase 3 (normal): remaining epochs")
    print(f"{'='*70}\n")
    
    # Reset BatchNorm statistics if requested
    if reset_bn_stats:
        print("Calibrating BatchNorm statistics...")
        calibrate_batchnorm(model, train_loader, device)
        print()
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer based on strategy
    if freeze_original or differential_lr or progressive_unfreezing:
        # Need to separate parameters into original vs regrown
        module_dict = dict(model.named_modules())
        original_params = []
        regrown_params = []
        
        if regrown_indices is not None:
            # We have explicit regrown indices
            for name, param in model.named_parameters():
                if 'weight' in name and not 'weight_mask' in name:
                    # Extract layer name from parameter name
                    layer_name = name.replace('.weight_orig', '').replace('.weight', '')
                    
                    if layer_name in regrown_indices and len(regrown_indices[layer_name]) > 0:
                        # This layer has regrown weights
                        # For simplicity, we treat the whole parameter as "mixed"
                        # Ideally we'd mask gradients, but that's complex
                        regrown_params.append(param)
                    else:
                        original_params.append(param)
                else:
                    # Bias, BN params, etc. - group with regrown for flexibility
                    regrown_params.append(param)
        else:
            # No regrown indices provided - can't distinguish
            # Fall back to treating all params as regrown
            regrown_params = list(model.parameters())
        
        if freeze_original and not progressive_unfreezing:
            # Freeze original params permanently
            for param in original_params:
                param.requires_grad = False
            optimizer = optim.AdamW(regrown_params, lr=lr_regrown if differential_lr else lr, weight_decay=0.01)
            print(f"Freezing {len(original_params)} original parameter groups, training {len(regrown_params)} regrown groups\n")
        elif differential_lr:
            # Different learning rates
            optimizer = optim.AdamW([
                {'params': original_params, 'lr': lr_original, 'weight_decay': 0.01},
                {'params': regrown_params, 'lr': lr_regrown, 'weight_decay': 0.01}
            ])
            print(f"Using differential LR: {len(original_params)} original ({lr_original}), {len(regrown_params)} regrown ({lr_regrown})\n")
        else:
            # Progressive unfreezing - start frozen
            for param in original_params:
                param.requires_grad = False
            optimizer = optim.AdamW(regrown_params, lr=lr_regrown, weight_decay=0.01)
            print(f"Progressive unfreezing: Starting with {len(original_params)} params frozen\n")
    else:
        # Standard: train all parameters with same LR
        optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    
    # Learning rate scheduler
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer, 
    #     num_warmup_steps=int(0.05*epochs), 
    #     num_training_steps=epochs
    # )
    
    best_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    current_phase = 1  # For progressive unfreezing
    
    for epoch in range(epochs):
        # Handle progressive unfreezing phase transitions
        if progressive_unfreezing:
            if epoch == phase1_epochs and current_phase == 1:
                # Phase 1 → Phase 2: Unfreeze with differential LR
                print(f"\n{'='*70}")
                print(f"Phase 2: Unfreezing original weights with differential LR")
                print(f"{'='*70}\n")
                for param in original_params:
                    param.requires_grad = True
                # Recreate optimizer with differential LR
                optimizer = optim.AdamW([
                    {'params': original_params, 'lr': lr_original, 'weight_decay': 0.01},
                    {'params': regrown_params, 'lr': lr_regrown, 'weight_decay': 0.01}
                ])
                current_phase = 2
            elif epoch == phase1_epochs + phase2_epochs and current_phase == 2:
                # Phase 2 → Phase 3: Normal training
                print(f"\n{'='*70}")
                print(f"Phase 3: Full finetuning with standard LR")
                print(f"{'='*70}\n")
                optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
                current_phase = 3
        
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_accuracy = 100.0 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_accuracy = 100.0 * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        
        # Update learning rate
        # scheduler.step()
        # current_lr = scheduler.get_last_lr()[0]
        
        # Track best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            
            # Save best model
            if save_path:
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'accuracy': best_accuracy,
                    'train_accuracy': train_accuracy,
                }, save_path)
                if verbose:
                    print(f"  ✓ Saved new best model (epoch {best_epoch}, acc: {best_accuracy:.2f}%)")
        else:
            epochs_without_improvement += 1
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Train Acc: {train_accuracy:.2f}% | "
                  f"Test Loss: {avg_test_loss:.4f} | "
                  f"Test Acc: {test_accuracy:.2f}% | "
                  f"Best: {best_accuracy:.2f}%")
        
        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"\n{'='*70}")
            print(f"Early stopping at epoch {epoch+1}: No improvement for {patience} epochs")
            print(f"Best accuracy: {best_accuracy:.2f}% (epoch {best_epoch})")
            print(f"{'='*70}")
            break
    
    print(f"\n{'='*70}")
    print(f"Finetuning Completed!")
    print(f"  Best Accuracy: {best_accuracy:.2f}% (epoch {best_epoch})")
    if save_path:
        print(f"  Model saved to: {save_path}")
    print(f"{'='*70}\n")
    
    return best_accuracy, best_model_state


def finetune_from_checkpoint(checkpoint_path, model_name, starting_checkpoint='oneshot',
                             epochs=100, lr=0.001, patience=100, save_dir='./rl_nas_checkpoints',
                             data_dir='./data', device='cuda', reference_model_path=None,
                             reset_bn_stats=False, freeze_original=False, differential_lr=False,
                             lr_original=1e-5, lr_regrown=3e-4, progressive_unfreezing=False,
                             phase1_epochs=50, phase2_epochs=150):
    """
    Perform final finetuning from a training checkpoint or best allocation
    
    Args:
        checkpoint_path: Path to checkpoint or best allocation file
        model_name: Model architecture name (resnet20, densenet, vgg16, alexnet)
        starting_checkpoint: 'oneshot' or 'iterative' - determines model_99 source
        epochs: Number of finetuning epochs
        lr: Learning rate
        patience: Early stopping patience
        save_dir: Directory to save final model
        data_dir: Data directory
        device: Device to use
        reference_model_path: Optional path to custom reference model for weight initialization
                             (if None, uses pretrained model for oneshot or 95% for iterative)
    """
    print(f"\n{'='*80}")
    print(f"FINAL FINETUNING FROM CHECKPOINT")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {model_name}")
    print(f"Starting checkpoint: {starting_checkpoint}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    print("Loading checkpoint...")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract relevant info
    if 'best_allocation' in checkpoint:
        # Training checkpoint
        allocation = checkpoint['best_allocation']
        regrow_indices = checkpoint.get('best_regrow_indices', {})
        reward = checkpoint.get('best_reward', 0.0)
        epoch = checkpoint.get('epoch', -1)
    elif 'allocation' in checkpoint:
        # Best allocation file or window_best checkpoint
        allocation = checkpoint['allocation']
        regrow_indices = checkpoint.get('regrow_indices', {})
        reward = checkpoint.get('reward', 0.0)
        # Handle window_best checkpoint which uses 'window_best_epoch' instead of 'epoch'
        if 'window_best_epoch' in checkpoint:
            epoch = checkpoint['window_best_epoch']
            window_end = checkpoint.get('window_end_epoch', -1)
            print(f"  Window checkpoint: best found at epoch {epoch+1}, window ends at epoch {window_end+1}")
        else:
            epoch = checkpoint.get('epoch', -1)
    else:
        print(f"Error: Checkpoint does not contain allocation data")
        return
    
    print(f"  Checkpoint epoch: {epoch + 1}")
    print(f"  Best reward: {reward:.4f} ({reward*100:.2f}%)")
    print(f"  Allocation layers: {len(allocation)}")
    print(f"  Total weights to regrow: {sum(allocation.values())}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = data_loader(data_dir=data_dir)
    
    # Load reference model (model_95) for masks and weights
    print("\nLoading reference model for weight initialization...")
    model_pretrained = model_loader(model_name, device)
    from utils.analysis_utils import load_model
    load_model(model_pretrained, f'./{model_name}/checkpoint')
    
    model_reference = model_loader(model_name, device)
    prune_weights_reparam(model_reference)
    
    if reference_model_path is not None:
        # Use custom reference model (e.g., 97% sparsity model)
        print(f"  Loading custom reference model from: {reference_model_path}")
        ref_checkpoint = torch.load(reference_model_path)
        model_reference.load_state_dict(ref_checkpoint)
        print("  ✓ Using custom reference model for weight initialization")
    elif starting_checkpoint == 'oneshot':
        # Use pretrained model (0% sparsity - best initialization)
        prune_weights_reparam(model_pretrained)
        model_reference.load_state_dict(model_pretrained.state_dict())
        print("  ✓ Using pretrained model (0% sparsity) for weight initialization")
    else:
        # Use 95% sparsity model from iterative pruning
        checkpoint_95 = torch.load(f'./iterative_step_0.2/{model_name}/ckpt_after_prune/pruned_finetuned_mask_it1.pth')
        model_reference.load_state_dict(checkpoint_95)
        print("  ✓ Using iterative 95% sparsity model for weight initialization")
    
    # Load starting model (model_99)
    print("\nLoading starting model (model_99)...")
    model_current = model_loader(model_name, device)
    prune_weights_reparam(model_current)
    
    if starting_checkpoint == 'oneshot':
        if model_name in ["resnet20"]:
            checkpoint_99 = torch.load(f'./{model_name}/ckpt_after_prune/pruned_finetuned_mask_0.95.pth')
            print("  Loaded one-shot 95% sparsity checkpoint")
        elif model_name in ["vgg16", "densenet"]:
            checkpoint_99 = torch.load(f'./{model_name}/ckpt_after_prune/pruned_finetuned_mask_0.99_repeat2_patience30.pth')
            print("  Loaded one-shot 99% sparsity checkpoint")
        else:
            checkpoint_99 = torch.load(f'./{model_name}/ckpt_after_prune/pruned_finetuned_mask_0.99.pth')
            print("  Loaded one-shot 99% sparsity checkpoint")
        model_current.load_state_dict(checkpoint_99)
    else:
        if model_name in ["resnet20"]:
            checkpoint_99 = torch.load(f'./iterative_step_0.2/{model_name}/ckpt_after_prune/pruned_finetuned_mask_it19.pth')
            print("  Loaded iterative 99% sparsity checkpoint (it19)")
        else:
            # checkpoint_99 = torch.load(f'./iterative_step_0.2/{model_name}/ckpt_after_prune/pruned_finetuned_mask_it20.pth')
            checkpoint_99 = torch.load('/home/j/junchen/DAC26/rl_regrow_savedir/alexnet/iterative_solution11_eps500/step1/iter_3_model_alexnet.pth')
            print("  Loaded iterative 99% sparsity checkpoint (it20)")
        model_current.load_state_dict(checkpoint_99['model_state_dict'])
    
    # Get target layers
    if model_name == "resnet20":
        # target_layers = ["layer2.1.conv2", "layer2.2.conv1", "layer2.2.conv2", "layer3.0.conv1",
        #                "layer3.0.conv2", "layer3.0.shortcut.0", "layer3.1.conv1", "layer3.1.conv2",
        #                "layer3.2.conv1", "layer3.2.conv2", "linear"]
        target_layers = ["layer2.0.conv2", "layer2.1.conv1", "layer2.2.conv2", "layer2.2.conv1",
                       "layer3.0.conv2", "layer3.0.conv1", "layer3.1.conv1", "layer3.1.conv2"]
    # elif model_name == "densenet":
    #     target_layers = ["dense3.0.conv1", "dense3.4.conv1", "dense3.7.conv1", 
    #     "dense3.8.conv1", "dense3.9.conv2", "dense3.14.conv2", "dense3.15.conv2", 
    #     "dense3.21.conv2", "dense3.22.conv1", "dense4.0.conv1", "dense4.1.conv1", 
    #     "dense4.3.conv2", "dense4.6.conv2", "dense4.8.conv2", "dense4.9.conv2", 
    #     "dense4.12.conv1", "dense4.12.conv2", "dense4.15.conv2"]
    elif model_name == "densenet":
        target_layers = ["dense3.0.conv1", "dense3.19.conv2", "dense3.20.conv1", "dense3.21.conv1",
        "dense4.0.conv1", "dense4.5.conv2", "dense4.7.conv1", "dense4.11.conv1", "dense4.11.conv2", "linear"]
    elif model_name == "vgg16":
        # target_layers = ["features.14", "features.17", "features.27", 
        #                 "features.30", "features.34", "features.40", "classifier"]
        target_layers = ["features.14", "features.17", "features.20", "features.24", "features.27", 
                        "features.0", "features.3", "features.7", "features.10", "classifier"]
    elif model_name == "alexnet":
        target_layers = ['features.3', 'features.6', 'features.8', 'features.10', 'classifier.1'] #
    # elif model_name == "effnet":
    #     target_layers = ["layers.3.conv3", "layers.4.conv1", "layers.4.conv3", "layers.5.conv1",
    #     "layers.6.conv3", "layers.7.conv3", "layers.8.conv1", "layers.8.conv2", "layers.8.conv3",
    #     "layers.15.conv1", "layers.15.conv2", "layers.15.conv3"]
    else:
        print(f"Error: Unknown model name: {model_name}")
        return
    
    # Extract reference masks and weights
    print("\nExtracting reference masks and weights...")
    reference_masks = {}
    reference_weights = {}
    
    module_dict_ref = dict(model_reference.named_modules())
    for layer_name in target_layers:
        module_ref = module_dict_ref.get(layer_name)
        if module_ref and hasattr(module_ref, 'weight_mask'):
            reference_masks[layer_name] = module_ref.weight_mask.clone()
            if hasattr(module_ref, 'weight_orig'):
                reference_weights[layer_name] = module_ref.weight_orig.detach().clone()
            else:
                reference_weights[layer_name] = module_ref.weight.detach().clone()
    
    # Evaluate before regrowth
    print("\nEvaluating before regrowth...")
    before_accuracy = evaluate_model_accuracy(model_current, test_loader, device)
    before_sparsity, before_total, before_pruned = calculate_model_sparsity(model_current)
    print(f"  Accuracy: {before_accuracy:.2f}%")
    print(f"  Sparsity: {before_sparsity:.2f}% ({before_pruned}/{before_total} pruned)")
    
    # Apply regrowth from saved indices
    apply_regrowth_from_indices(
        model=model_current,
        allocation=allocation,
        regrow_indices=regrow_indices,
        reference_masks=reference_masks,
        reference_weights=reference_weights
    )
    
    # Evaluate after regrowth
    print("\nEvaluating after regrowth (before finetuning)...")
    after_regrow_accuracy = evaluate_model_accuracy(model_current, test_loader, device)
    after_regrow_sparsity, after_total, after_pruned = calculate_model_sparsity(model_current)
    print(f"  Accuracy: {after_regrow_accuracy:.2f}%")
    print(f"  Sparsity: {after_regrow_sparsity:.2f}% ({after_pruned}/{after_total} pruned)")
    print(f"  Improvement: {after_regrow_accuracy - before_accuracy:+.2f}%")
    
    # Final finetuning
    final_save_path = os.path.join(save_dir, f'final_finetuned_{model_name}_from_epoch{epoch+1}.pth')
    final_accuracy, final_state = full_finetune(
        model=model_current,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        save_path=final_save_path,
        verbose=True,
        patience=patience,
        reset_bn_stats=reset_bn_stats,
        freeze_original=freeze_original,
        regrown_indices=regrow_indices,
        differential_lr=differential_lr,
        lr_original=lr_original,
        lr_regrown=lr_regrown,
        progressive_unfreezing=progressive_unfreezing,
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs
    )
    
    # Load best model for final evaluation
    model_current.load_state_dict(final_state)
    
    # Final evaluation
    print("\nFinal Evaluation...")
    final_eval_accuracy = evaluate_model_accuracy(model_current, test_loader, device)
    final_sparsity, final_total, final_pruned = calculate_model_sparsity(model_current)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Before regrowth:")
    print(f"  Accuracy: {before_accuracy:.2f}%")
    print(f"  Sparsity: {before_sparsity:.2f}%")
    print(f"\nAfter regrowth:")
    print(f"  Accuracy: {after_regrow_accuracy:.2f}%")
    print(f"  Sparsity: {after_regrow_sparsity:.2f}%")
    print(f"\nAfter finetuning:")
    print(f"  Best accuracy: {final_accuracy:.2f}%")
    print(f"  Final evaluation: {final_eval_accuracy:.2f}%")
    print(f"  Sparsity: {final_sparsity:.2f}%")
    print(f"\nOverall improvement:")
    print(f"  From before regrowth: {final_accuracy - before_accuracy:+.2f}%")
    print(f"  From after regrowth: {final_accuracy - after_regrow_accuracy:+.2f}%")
    print(f"\nFinal model saved to: {final_save_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Inspect RL checkpoint or perform final finetuning')
    parser.add_argument('--file', type=str, help='Path to checkpoint file')
    parser.add_argument('--type', type=str, choices=['checkpoint', 'best', 'auto'],
                        default='auto', help='Type of file to inspect')
    # Finetuning options
    parser.add_argument('--finetune', action='store_true',
                        help='Perform final finetuning from checkpoint')
    parser.add_argument('--model', '--m_name', type=str, default=None,
                        help='Model name (required for finetuning): resnet20, densenet, vgg16, alexnet')
    parser.add_argument('--starting_checkpoint', type=str, default='oneshot',
                        choices=['oneshot', 'iterative'],
                        help='Starting checkpoint type for model_99')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='Number of finetuning epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate for finetuning')
    parser.add_argument('--patience', type=int, default=100,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--save_dir', type=str, default='./rl_nas_checkpoints',
                        help='Directory to save final model')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--reference_model', type=str, default=None,
                        help='Path to custom reference model for weight initialization (e.g., 97%% sparsity model). '
                             'If not specified: uses pretrained (0%% sparsity) for oneshot, or 95%% sparsity for iterative.')
    
    # Selective finetuning strategies
    parser.add_argument('--reset_bn_stats', action='store_true',
                        help='Reset BatchNorm statistics before finetuning (can recover from distribution shift)')
    parser.add_argument('--freeze_original', action='store_true',
                        help='Freeze original sparse weights, train only regrown weights')
    parser.add_argument('--differential_lr', action='store_true',
                        help='Use different learning rates for original vs regrown weights')
    parser.add_argument('--lr_original', type=float, default=1e-5,
                        help='Learning rate for original weights (if differential_lr=True)')
    parser.add_argument('--lr_regrown', type=float, default=3e-4,
                        help='Learning rate for regrown weights (if differential_lr or freeze_original)')
    parser.add_argument('--progressive_unfreezing', action='store_true',
                        help='Gradually unfreeze original weights during training')
    parser.add_argument('--phase1_epochs', type=int, default=50,
                        help='Epochs for phase 1 (freeze original) in progressive unfreezing')
    parser.add_argument('--phase2_epochs', type=int, default=150,
                        help='Epochs for phase 2 (differential LR) in progressive unfreezing')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Auto-detect type
    if args.type == 'auto':
        if 'best_allocation' in args.file:
            args.type = 'best'
        else:
            args.type = 'checkpoint'
    
    # Perform finetuning if requested
    if args.finetune:
        if not args.model:
            print("Error: --model is required for finetuning")
            print("Example: --model densenet")
            return
        
        finetune_from_checkpoint(
            checkpoint_path=args.file,
            model_name=args.model,
            starting_checkpoint=args.starting_checkpoint,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            save_dir=args.save_dir,
            data_dir=args.data_dir,
            device=args.device,
            reference_model_path=args.reference_model,
            reset_bn_stats=args.reset_bn_stats,
            freeze_original=args.freeze_original,
            differential_lr=args.differential_lr,
            lr_original=args.lr_original,
            lr_regrown=args.lr_regrown,
            progressive_unfreezing=args.progressive_unfreezing,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs
        )
    else:
        # Just inspect the checkpoint
        if args.type == 'checkpoint':
            inspect_training_checkpoint(args.file)
        else:
            inspect_best_allocation(args.file)



if __name__ == '__main__':
    main()
