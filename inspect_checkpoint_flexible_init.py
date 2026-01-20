#!/usr/bin/env python3
"""
Inspect RL training checkpoint or best allocation file
AND perform final finetuning from checkpoint with FLEXIBLE INITIALIZATION

This is an extension of inspect_checkpoint.py that allows choosing different
weight initialization strategies for regrown weights:
- 'reference': Copy from reference model (default, same as inspect_checkpoint.py)
- 'zero': Initialize to 0.0
- 'kaiming': Kaiming/He initialization
- 'xavier': Xavier/Glorot initialization
- 'magnitude': Keep existing weight values

Usage:
    # Default (reference model copy)
    python inspect_checkpoint_flexible_init.py --file checkpoint.pth --model resnet20 --finetune
    
    # Zero initialization
    python inspect_checkpoint_flexible_init.py --file checkpoint.pth --model resnet20 --finetune --init_strategy zero
    
    # Kaiming initialization
    python inspect_checkpoint_flexible_init.py --file checkpoint.pth --model resnet20 --finetune --init_strategy kaiming
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import numpy as np
from datetime import datetime

from utils.model_loader import model_loader
from utils.data_loader import data_loader
from utils.analysis_utils import prune_weights_reparam, count_pruned_params


def inspect_training_checkpoint(checkpoint_path):
    """Inspect a training checkpoint file"""
    print(f"\n{'='*70}")
    print(f"Training Checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: File not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"Checkpoint Found:")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Reward: {checkpoint['reward']:.4f}")
    print(f"  Accuracy: {checkpoint['accuracy']:.2f}%")

    if 'timestamp' in checkpoint:
        timestamp = datetime.fromtimestamp(checkpoint['timestamp'])
        print(f"  Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    if 'allocation' in checkpoint and checkpoint['allocation']:
        print(f"\n  Layer Allocation:")
        total_allocated = sum(checkpoint['allocation'].values())
        print(f"    Total weights: {total_allocated}")
        print(f"    Layers:")
        for layer, count in checkpoint['allocation'].items():
            print(f"      {layer}: {count}")

    if 'regrow_indices' in checkpoint and checkpoint['regrow_indices']:
        print(f"\n  Regrow Indices Available:")
        for layer, indices in checkpoint['regrow_indices'].items():
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

    print(f"Best Allocation Contents:")
    print(f"  Epoch: {best_data['epoch'] + 1}")
    print(f"  Best Reward: {best_data['best_reward']:.4f} ({best_data['best_reward']*100:.2f}%)")

    if 'config' in best_data:
        print(f"\n  Training Config:")
        for key, value in best_data['config'].items():
            print(f"    {key}: {value}")

    if 'reward_baseline' in best_data and best_data['reward_baseline'] is not None:
        print(f"\n  Reward Baseline: {best_data['reward_baseline']:.4f}")

    if 'total_rewards' in best_data:
        rewards = best_data['total_rewards']
        if len(rewards) > 0:
            print(f"\n  Recent Rewards (last {len(rewards)}):")
            print(f"    Mean: {sum(rewards)/len(rewards):.4f}")
            print(f"    Min: {min(rewards):.4f}")
            print(f"    Max: {max(rewards):.4f}")
    if 'best_allocation' in best_data and best_data['best_allocation']:
        print(f"\n  Best Allocation:")
        total_allocated = sum(best_data['best_allocation'].values())
        print(f"    Total weights: {total_allocated}")
        print(f"    Layers:")
        for layer, count in best_data['best_allocation'].items():
            print(f"      {layer}: {count}")

    if 'best_regrow_indices' in best_data and best_data['best_regrow_indices']:
        print(f"\n  Best Regrow Indices Available:")
        for layer, indices in best_data['best_regrow_indices'].items():
            print(f"    {layer}: {len(indices)} indices")
    
    print(f"\n{'='*70}\n")


def apply_regrowth_from_indices(model, allocation, regrow_indices, reference_masks, 
                                reference_weights, init_strategy='reference', reinit_all=False):
    """
    Apply regrowth to model using saved indices with flexible initialization.
    
    Args:
        model: Model to apply regrowth to
        allocation: Dict[layer_name] -> num_weights
        regrow_indices: Dict[layer_name] -> List[tuple] of indices
        reference_masks: Dict[layer_name] -> reference mask tensor
        reference_weights: Dict[layer_name] -> reference weight tensor
        init_strategy: Initialization strategy for regrown weights
            - 'reference': Copy from reference model (default)
            - 'zero': Initialize to 0.0
            - 'kaiming': Kaiming/He initialization
            - 'xavier': Xavier/Glorot initialization
            - 'magnitude': Keep existing weight values (no change)
            - 'dual_lottery': Dual Lottery Ticket initialization (signed Kaiming on sparse mask)
        reinit_all: If True, reinitialize ALL weights (not just regrown), for training from scratch
    
    Returns:
        total_regrown: Number of weights regrown
    """
    print(f"\n{'='*70}")
    print("Applying Regrowth from Saved Indices")
    print(f"{'='*70}")
    print(f"Initialization Strategy: {init_strategy}")
    print(f"Reinitialize All Weights: {reinit_all}")
    print(f"{'='*70}\n")
    
    module_dict = dict(model.named_modules())
    total_regrown = 0
    total_reinitialized = 0
    
    # Step 1: Apply regrowth to UPDATE MASKS (don't initialize weights yet if reinit_all=True)
    print("Step 1: Applying regrowth to update mask patterns...\n")
    
    for layer_name, num_weights in allocation.items():
        if num_weights == 0:
            continue
            
        module = module_dict.get(layer_name)
        if module is None or not hasattr(module, 'weight_mask'):
            print(f"  Warning: Layer {layer_name} not found or has no mask, skipping")
            continue
        
        current_mask = module.weight_mask
        
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
        
        # Update mask to 1.0 (unpruned) for all regrown positions
        actual_regrown = 0
        for idx in selected_indices_tensor:
            tuple_idx = tuple(idx.tolist())
            current_mask[tuple_idx] = 1.0
            actual_regrown += 1
        
        total_regrown += actual_regrown
        print(f"  {layer_name}: {actual_regrown}/{num_weights} mask positions updated")
    
    print(f"\nTotal mask positions regrown: {total_regrown}")
    print(f"{'='*70}\n")
    
    # Step 2: Initialize weights based on strategy
    if reinit_all:
        # Reinitialize ALL weights (both existing and regrown) - Dual Lottery mode
        print("Step 2: Reinitializing ALL weights in the ENTIRE MODEL (training from scratch mode)...\n")
        
        # Iterate over ALL modules in the model, not just regrowth layers
        for layer_name, module in model.named_modules():
            # Skip modules without masks (e.g., ReLU, pooling, etc.)
            if not hasattr(module, 'weight_mask'):
                continue
            
            # Get weight parameter
            if hasattr(module, "weight_orig"):
                weight_param = module.weight_orig
            else:
                weight_param = module.weight
            
            # Get current mask (now includes regrowth from Step 1)
            current_mask = module.weight_mask
            
            # Reinitialize based on strategy
            if init_strategy == 'dual_lottery':
                # Dual Lottery: per-neuron fan-in aware initialization
                # Formula: w_ij ~ N(0, 2 / fan-in_i) where fan-in_i is sparse connectivity
                with torch.no_grad():
                    # Iterate over all output neurons
                    if len(current_mask.shape) == 4:  # Conv: (out_ch, in_ch, kh, kw)
                        for out_idx in range(current_mask.shape[0]):
                            # Count incoming connections to this output channel
                            neuron_mask = current_mask[out_idx, :, :, :]
                            sparse_fan_in = (neuron_mask == 1.0).sum().item()
                            if sparse_fan_in == 0:
                                continue  # Skip neurons with no connections
                            
                            # Initialize all incoming weights to this neuron
                            std = np.sqrt(2.0 / sparse_fan_in)
                            weight_param.data[out_idx, :, :, :] = torch.randn_like(
                                weight_param.data[out_idx, :, :, :]
                            ) * std
                            # Zero out pruned positions
                            weight_param.data[out_idx, :, :, :].mul_(neuron_mask)
                    
                    elif len(current_mask.shape) == 2:  # Linear: (out_features, in_features)
                        for out_idx in range(current_mask.shape[0]):
                            # Count incoming connections to this output neuron
                            neuron_mask = current_mask[out_idx, :]
                            sparse_fan_in = (neuron_mask == 1.0).sum().item()
                            if sparse_fan_in == 0:
                                continue
                            
                            # Initialize all incoming weights to this neuron
                            std = np.sqrt(2.0 / sparse_fan_in)
                            weight_param.data[out_idx, :] = torch.randn_like(
                                weight_param.data[out_idx, :]
                            ) * std
                            # Zero out pruned positions
                            weight_param.data[out_idx, :].mul_(neuron_mask)
            
            elif init_strategy == 'kaiming':
                # Standard Kaiming (dense fan-in, not mask-aware)
                nn.init.kaiming_normal_(weight_param, mode='fan_in', nonlinearity='relu')
                with torch.no_grad():
                    weight_param.data.mul_(current_mask)
            
            elif init_strategy == 'xavier':
                nn.init.xavier_normal_(weight_param)
                with torch.no_grad():
                    weight_param.data.mul_(current_mask)
            
            elif init_strategy == 'zero':
                nn.init.zeros_(weight_param)
            
            elif init_strategy == 'reference':
                # Keep reference weights
                if layer_name in reference_weights:
                    weight_param.data.copy_(reference_weights[layer_name])
                with torch.no_grad():
                    weight_param.data.mul_(current_mask)
            
            # magnitude: keep existing (no change)
            
            num_weights = (current_mask == 1.0).sum().item()
            total_reinitialized += num_weights
            print(f"  {layer_name}: Reinitialized {num_weights} active weights")
        
        print(f"\nTotal weights reinitialized: {total_reinitialized}")
        print(f"{'='*70}\n")
    
    else:
        # Only initialize regrown weights (not reinit_all mode)
        print("Step 2: Initializing only regrown weights...\n")
        
        for layer_name, num_weights in allocation.items():
            if num_weights == 0:
                continue
                
            module = module_dict.get(layer_name)
            if module is None or not hasattr(module, 'weight_mask'):
                continue
            
            current_mask = module.weight_mask
            
            # Get saved indices
            if layer_name not in regrow_indices or not regrow_indices[layer_name]:
                continue
            
            saved_indices = regrow_indices[layer_name]
            
            # Convert to tensor
            if isinstance(saved_indices, torch.Tensor):
                selected_indices_tensor = saved_indices
            else:
                selected_indices_tensor = torch.tensor(saved_indices, dtype=torch.long, device=current_mask.device)
            
            if selected_indices_tensor.ndim == 1 and selected_indices_tensor.numel() > 0:
                selected_indices_tensor = selected_indices_tensor.unsqueeze(0)
            
            # Get weight parameter
            if hasattr(module, "weight_orig"):
                weight_param = module.weight_orig
            else:
                weight_param = module.weight
            
            # Initialize only the regrown weights
            actual_initialized = 0
            for idx in selected_indices_tensor:
                tuple_idx = tuple(idx.tolist())
                
                # Initialize weight based on strategy
                if init_strategy == 'reference':
                    # Copy from reference model
                    if layer_name in reference_weights:
                        ref_weight = reference_weights[layer_name][tuple_idx]
                        weight_param.data[tuple_idx] = ref_weight.clone()
                    else:
                        weight_param.data[tuple_idx] = 0.0
                
                elif init_strategy == 'zero':
                    weight_param.data[tuple_idx] = 0.0
                
                elif init_strategy == 'kaiming':
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight_param)
                    bound = np.sqrt(6.0 / fan_in)
                    weight_param.data[tuple_idx] = torch.empty(1).uniform_(-bound, bound).item()
                
                elif init_strategy == 'xavier':
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight_param)
                    bound = np.sqrt(6.0 / (fan_in + fan_out))
                    weight_param.data[tuple_idx] = torch.empty(1).uniform_(-bound, bound).item()
                
                elif init_strategy == 'magnitude':
                    # Keep existing weight value (no change)
                    pass
                
                elif init_strategy == 'dual_lottery':
                    # Per-weight dual lottery initialization
                    out_idx = tuple_idx[0]
                    
                    if len(current_mask.shape) == 4:  # Conv layer
                        neuron_mask = current_mask[out_idx, :, :, :]
                    elif len(current_mask.shape) == 2:  # Linear layer
                        neuron_mask = current_mask[out_idx, :]
                    else:
                        raise ValueError(f"Unsupported weight shape: {current_mask.shape}")
                    
                    sparse_fan_in = (neuron_mask == 1.0).sum().item()
                    if sparse_fan_in == 0:
                        sparse_fan_in = 1
                    
                    std = np.sqrt(2.0 / sparse_fan_in)
                    weight_param.data[tuple_idx] = torch.randn(1).item() * std
                
                else:
                    raise ValueError(f"Unknown init_strategy: {init_strategy}")
                
                actual_initialized += 1
            
            print(f"  {layer_name}: Initialized {actual_initialized} regrown weights")
        
        print(f"\nTotal regrown weights initialized: {total_regrown}")
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
    
    print(f"  ✓ BatchNorm calibration complete")


def full_finetune(model, train_loader, test_loader, device, 
                  epochs=100, lr=0.001, save_path=None, verbose=True, patience=100,
                  reset_bn_stats=False):
    """
    Complete finetuning process with best model tracking and early stopping.
    
    Args:
        model: Model to finetune
        train_loader: Training data loader
        test_loader: Test data loader  
        device: Device to train on
        epochs: Number of finetuning epochs
        lr: Learning rate
        save_path: Path to save best model (if None, don't save)
        verbose: Print progress
        patience: Number of epochs without improvement before early stopping
        reset_bn_stats: Whether to reset BatchNorm statistics before training
    
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
    print(f"{'='*70}\n")
    
    # Reset BatchNorm statistics if requested
    if reset_bn_stats:
        print("Calibrating BatchNorm statistics...")
        calibrate_batchnorm(model, train_loader, device)
        print()
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    best_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
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
        
        # Track best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = model.state_dict()
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # Save checkpoint
            if save_path:
                torch.save(best_model_state, save_path)
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  ✓ New best model saved (epoch {epoch+1})")
        else:
            epochs_without_improvement += 1
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}%")
            print(f"  Test  - Loss: {avg_test_loss:.4f}, Acc: {test_accuracy:.2f}%")
            print(f"  Best: {best_accuracy:.2f}% (epoch {best_epoch+1}), No improve: {epochs_without_improvement}")
        
        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
            break
    
    print(f"\n{'='*70}")
    print(f"Finetuning Completed!")
    print(f"  Best Accuracy: {best_accuracy:.2f}% (epoch {best_epoch+1})")
    if save_path:
        print(f"  Model saved to: {save_path}")
    print(f"{'='*70}\n")
    
    return best_accuracy, best_model_state


def finetune_from_checkpoint(checkpoint_path, model_name, starting_checkpoint='oneshot',
                             epochs=100, lr=0.001, patience=100, save_dir='./rl_nas_checkpoints',
                             data_dir='./data', device='cuda', reference_model_path=None,
                             reset_bn_stats=False, init_strategy='reference', reinit_all=False):
    """
    Perform final finetuning from a training checkpoint or best allocation.
    
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
        reset_bn_stats: Whether to reset BatchNorm statistics before finetuning
        init_strategy: Weight initialization strategy for regrown weights
            - 'reference': Copy from reference model (default)
            - 'zero': Initialize to 0.0
            - 'kaiming': Kaiming/He initialization
            - 'xavier': Xavier/Glorot initialization
            - 'magnitude': Keep existing weight values
            - 'dual_lottery': Dual Lottery Ticket initialization (AAAI 2022)
        reinit_all: If True, reinitialize ALL weights and train from scratch (for Dual Lottery experiments)
    """
    print(f"\n{'='*80}")
    print(f"FINAL FINETUNING FROM CHECKPOINT")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {model_name}")
    print(f"Starting checkpoint: {starting_checkpoint}")
    print(f"Initialization strategy: {init_strategy}")
    print(f"Reinitialize all weights: {reinit_all}")
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
    
    # Load reference model for masks and weights (if using reference initialization)
    reference_masks = {}
    reference_weights = {}
    
    if init_strategy == 'reference':
        print("\nLoading reference model for weight initialization...")
        model_pretrained = model_loader(model_name, device)
        from utils.analysis_utils import load_model
        load_model(model_pretrained, f'./{model_name}/checkpoint')
        
        model_reference = model_loader(model_name, device)
        prune_weights_reparam(model_reference)
        
        if reference_model_path is not None:
            # Use custom reference model
            print(f"  Loading custom reference model from: {reference_model_path}")
            ref_checkpoint = torch.load(reference_model_path)
            model_reference.load_state_dict(ref_checkpoint)
            print("  ✓ Using custom reference model for weight initialization")
        elif starting_checkpoint == 'oneshot':
            # Use pretrained model (0% sparsity)
            prune_weights_reparam(model_pretrained)
            model_reference.load_state_dict(model_pretrained.state_dict())
            print("  ✓ Using pretrained model (0% sparsity) for weight initialization")
        else:
            # Use 95% sparsity model from iterative pruning
            checkpoint_95 = torch.load(f'./iterative_step_0.2/{model_name}/ckpt_after_prune/pruned_finetuned_mask_it1.pth')
            model_reference.load_state_dict(checkpoint_95)
            print("  ✓ Using iterative 95% sparsity model for weight initialization")
    else:
        print(f"\nUsing '{init_strategy}' initialization (no reference model needed)")
        model_reference = None
    
    # Load starting model (model_99)
    print("\nLoading starting model (model_99)...")
    model_current = model_loader(model_name, device)
    prune_weights_reparam(model_current)
    
    if starting_checkpoint == 'oneshot':
        if model_name in ["resnet20"]:
            checkpoint_99 = torch.load(f'./{model_name}/ckpt_after_prune/pruned_finetuned_mask_0.95.pth')
        elif model_name in ["vgg16", "densenet"]:
            checkpoint_99 = torch.load(f'./{model_name}/ckpt_after_prune/pruned_finetuned_mask_0.995.pth')
        else:
            checkpoint_99 = torch.load(f'./{model_name}/ckpt_after_prune/pruned_finetuned_mask_0.99.pth')
        model_current.load_state_dict(checkpoint_99)
    else:
        if model_name in ["resnet20"]:
            checkpoint_99 = torch.load(f'./iterative_step_0.2/{model_name}/ckpt_after_prune/pruned_finetuned_mask_it2.pth')
        else:
            checkpoint_99 = torch.load(f'./iterative_step_0.2/{model_name}/ckpt_after_prune/pruned_finetuned_mask_it3.pth')
        model_current.load_state_dict(checkpoint_99['model_state_dict'])
    
    # Get target layers
    if model_name == "resnet20":
        target_layers = ["layer2.0.conv2", "layer2.1.conv1", "layer2.2.conv2", "layer2.2.conv1",
                       "layer3.0.conv2", "layer3.0.conv1", "layer3.1.conv1", "layer3.1.conv2"]
    elif model_name == "densenet":
        target_layers = ["dense3.0.conv1", "dense3.19.conv2", "dense3.20.conv1", "dense3.21.conv1",
        "dense4.0.conv1", "dense4.5.conv2", "dense4.7.conv1", "dense4.11.conv1", "dense4.11.conv2", "linear"]
    elif model_name == "vgg16":
        target_layers = ["features.14", "features.17", "features.20", "features.24", "features.27", 
                        "features.0", "features.3", "features.7", "features.10", "classifier"]
    elif model_name == "alexnet":
        target_layers = ['features.3', 'features.6', 'features.8', 'features.10', 'classifier.1']
    else:
        print(f"Error: Unknown model name: {model_name}")
        return
    
    # Extract reference masks and weights (if using reference initialization)
    if init_strategy == 'reference' and model_reference is not None:
        print("\nExtracting reference masks and weights...")
        module_dict_ref = dict(model_reference.named_modules())
        for layer_name in target_layers:
            module_ref = module_dict_ref.get(layer_name)
            if module_ref and hasattr(module_ref, 'weight_mask'):
                reference_masks[layer_name] = module_ref.weight_mask.clone()
                if hasattr(module_ref, 'weight_orig'):
                    reference_weights[layer_name] = module_ref.weight_orig.data.clone()
                else:
                    reference_weights[layer_name] = module_ref.weight.data.clone()
    
    # Evaluate before regrowth
    print("\nEvaluating before regrowth...")
    before_accuracy = evaluate_model_accuracy(model_current, test_loader, device)
    before_sparsity, before_total, before_pruned = calculate_model_sparsity(model_current)
    print(f"  Accuracy: {before_accuracy:.2f}%")
    print(f"  Sparsity: {before_sparsity:.2f}% ({before_pruned}/{before_total} pruned)")
    
    # Apply regrowth from saved indices with chosen initialization
    apply_regrowth_from_indices(
        model=model_current,
        allocation=allocation,
        regrow_indices=regrow_indices,
        reference_masks=reference_masks,
        reference_weights=reference_weights,
        init_strategy=init_strategy,
        reinit_all=reinit_all
    )
    
    # Evaluate after regrowth
    print("\nEvaluating after regrowth (before finetuning)...")
    after_regrow_accuracy = evaluate_model_accuracy(model_current, test_loader, device)
    after_regrow_sparsity, after_total, after_pruned = calculate_model_sparsity(model_current)
    print(f"  Accuracy: {after_regrow_accuracy:.2f}%")
    print(f"  Sparsity: {after_regrow_sparsity:.2f}% ({after_pruned}/{after_total} pruned)")
    print(f"  Improvement: {after_regrow_accuracy - before_accuracy:+.2f}%")
    
    # Final finetuning
    final_save_path = os.path.join(save_dir, f'final_finetuned_{model_name}_init_{init_strategy}_epoch{epoch+1}.pth')
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
        reset_bn_stats=reset_bn_stats
    )
    
    # Load best model for final state (already evaluated during finetuning)
    model_current.load_state_dict(final_state)
    final_sparsity, final_total, final_pruned = calculate_model_sparsity(model_current)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Initialization strategy: {init_strategy}")
    print(f"\nBefore regrowth:")
    print(f"  Accuracy: {before_accuracy:.2f}%")
    print(f"  Sparsity: {before_sparsity:.2f}%")
    print(f"\nAfter regrowth:")
    print(f"  Accuracy: {after_regrow_accuracy:.2f}%")
    print(f"  Sparsity: {after_regrow_sparsity:.2f}%")
    print(f"\nAfter finetuning:")
    print(f"  Best accuracy: {final_accuracy:.2f}%")
    print(f"  Sparsity: {final_sparsity:.2f}%")
    print(f"\nOverall improvement:")
    print(f"  From before regrowth: {final_accuracy - before_accuracy:+.2f}%")
    print(f"  From after regrowth: {final_accuracy - after_regrow_accuracy:+.2f}%")
    print(f"\nFinal model saved to: {final_save_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Inspect RL checkpoint or perform final finetuning with flexible initialization')
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
                        help='Path to custom reference model for weight initialization (only used with --init_strategy reference)')
    
    # Initialization strategy (NEW)
    parser.add_argument('--init_strategy', type=str, default='reference',
                        choices=['reference', 'zero', 'kaiming', 'xavier', 'magnitude', 'dual_lottery'],
                        help='Weight initialization strategy for regrown weights:\n'
                             '  reference: Copy from reference model (default)\n'
                             '  zero: Initialize to 0.0\n'
                             '  kaiming: Kaiming/He initialization\n'
                             '  xavier: Xavier/Glorot initialization\n'
                             '  magnitude: Keep existing weight values\n'
                             '  dual_lottery: Dual Lottery Ticket init (AAAI 2022)')
    
    # Full reinitialization for training from scratch (Dual Lottery experiments)
    parser.add_argument('--reinit_all', action='store_true',
                        help='Reinitialize ALL weights (not just regrown) and train from scratch. '
                             'Useful for Dual Lottery Ticket experiments where the sparse mask pattern '
                             'is reused but weights are reinitialized.')
    
    # Other finetuning options
    parser.add_argument('--reset_bn_stats', action='store_true',
                        help='Reset BatchNorm statistics before finetuning')
    
    args = parser.parse_args()
    
    # Auto-detect type based on file content
    if args.type == 'auto':
        if 'best_allocation' in args.file:
            args.type = 'best'  # best_allocation_xxx.pth files
        else:
            args.type = 'checkpoint'  # training checkpoint files
    
    # Perform finetuning if requested
    if args.finetune:
        if not args.model:
            print("Error: --model is required for finetuning")
            print("Available models: resnet20, densenet, vgg16, alexnet")
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
            init_strategy=args.init_strategy,
            reinit_all=args.reinit_all
        )
    else:
        # Just inspect the checkpoint
        if args.type == 'checkpoint':
            inspect_training_checkpoint(args.file)
        else:
            inspect_best_allocation(args.file)


if __name__ == '__main__':
    main()
