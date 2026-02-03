"""
Single-Layer Regrowth Analysis

This script tests the hypothesis that SSIM scores correlate with regrowth effectiveness.

Strategy:
1. Calculate GLOBAL regrowth budget = (start_sparsity - target_sparsity) × total_parameters
   Example: (98% - 97%) × 270,000 = 2,700 weights
   
2. For each target layer:
   a. Apply the ENTIRE global budget to that single layer
   b. Use reference model (97%) weights where available
   c. Use random initialization (Kaiming) for any remaining budget
   d. Finetune the model
   e. Measure accuracy improvement
   f. Compare with SSIM score to check correlation

This helps identify which layers benefit most from regrowth and validates
the SSIM-based prioritization strategy used in the RL agent.
"""

import os
import sys
import argparse
import copy
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from utils.model_loader import model_loader
from utils.data_loader import data_loader
from utils.analysis_utils import (
    BlockwiseFeatureExtractor, compute_block_ssim,
    load_model, prune_weights_reparam, count_pruned_params, load_model_name
)


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_layer_ssim_scores(model_current, model_pretrained, target_layers, train_loader):
    """
    Compute SSIM scores for each target layer.
    Lower SSIM = more dissimilar features = higher degradation
    
    Returns:
        dict: {layer_name: ssim_score}
    """
    print("\nComputing SSIM scores for all layers...")
    
    block_dict = {'target_block': target_layers}
    
    extractor_pretrained = BlockwiseFeatureExtractor(model_pretrained, block_dict)
    extractor_current = BlockwiseFeatureExtractor(model_current, block_dict)
    
    with torch.no_grad():
        features_pretrained = extractor_pretrained.extract_block_features(train_loader, num_batches=128)
        features_current = extractor_current.extract_block_features(train_loader, num_batches=128)
    
    ssim_scores = compute_block_ssim(features_pretrained, features_current)
    
    # Extract scores for each layer
    layer_ssim = {}
    target_block_scores = ssim_scores.get('target_block', {})
    
    for layer_name in target_layers:
        if layer_name in target_block_scores:
            layer_ssim[layer_name] = target_block_scores[layer_name]
        else:
            print(f"Warning: SSIM not computed for {layer_name}, using default 0.5")
            layer_ssim[layer_name] = 0.5
    
    return layer_ssim


def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy on test set"""
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


def calculate_sparsity(model):
    """Calculate model sparsity"""
    total_params = 0
    pruned_params = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mask'):
            mask = module.weight_mask
            total_params += mask.numel()
            pruned_params += (mask == 0).sum().item()
    
    sparsity = 100.0 * pruned_params / total_params if total_params > 0 else 0.0
    return sparsity, total_params, pruned_params


def apply_regrowth_to_layer(model, layer_name, num_weights, reference_weights, reference_mask):
    """
    Apply regrowth to a single layer.
    First tries to use reference model weights, then random initialization if needed.
    
    Args:
        model: Model to modify
        layer_name: Name of layer to regrow
        num_weights: Number of weights to regrow (can exceed reference capacity)
        reference_weights: Reference weights (from model_97 or pretrained)
        reference_mask: Reference mask (from model_97 or pretrained)
    
    Returns:
        tuple: (from_reference, random_init) - number of weights regrown from each source
    """
    module_dict = dict(model.named_modules())
    module = module_dict.get(layer_name)
    
    if module is None or not hasattr(module, 'weight_mask'):
        print(f"Warning: Layer {layer_name} not found or has no mask")
        return 0, 0
    
    current_mask = module.weight_mask
    
    # Phase 1: Regrow from reference model (where available)
    # Regrowable positions: currently pruned (0) but were present in reference (1)
    regrowable_from_ref = (current_mask == 0) & (reference_mask == 1)
    regrowable_ref_indices = torch.nonzero(regrowable_from_ref, as_tuple=False)
    
    num_from_ref = min(num_weights, len(regrowable_ref_indices))
    
    # Random selection from reference positions
    perm = torch.randperm(len(regrowable_ref_indices))
    selected_ref_indices = regrowable_ref_indices[perm[:num_from_ref]]
    
    # Apply regrowth from reference
    for idx in selected_ref_indices:
        tuple_idx = tuple(idx.tolist())
        # Update mask
        current_mask[tuple_idx] = 1.0
        # Copy weight from reference
        if hasattr(module, 'weight_orig'):
            module.weight_orig.data[tuple_idx] = reference_weights[tuple_idx]
    
    # Phase 2: Random regrowth if budget not exhausted
    remaining_budget = num_weights - num_from_ref
    num_random = 0
    
    if remaining_budget > 0:
        # Find positions that are currently pruned but NOT in reference (random regrowth)
        regrowable_random = (current_mask == 0) & (reference_mask == 0)
        regrowable_random_indices = torch.nonzero(regrowable_random, as_tuple=False)
        
        num_random = min(remaining_budget, len(regrowable_random_indices))
        
        if num_random > 0:
            # Random selection from random positions
            perm = torch.randperm(len(regrowable_random_indices))
            selected_random_indices = regrowable_random_indices[perm[:num_random]]
            
            # Apply random regrowth with Kaiming initialization
            for idx in selected_random_indices:
                tuple_idx = tuple(idx.tolist())
                # Update mask
                current_mask[tuple_idx] = 1.0
                # Random initialization
                if hasattr(module, 'weight_orig'):
                    # Use fan-in for He initialization
                    fan_in = module.weight_orig.data.shape[1] if len(module.weight_orig.data.shape) > 1 else 1
                    std = np.sqrt(2.0 / fan_in)
                    module.weight_orig.data[tuple_idx] = torch.randn(1).item() * std
    
    total_regrown = num_from_ref + num_random
    print(f"  Regrew {total_regrown}/{num_weights} weights in {layer_name}")
    print(f"    From reference: {num_from_ref}, Random init: {num_random}")
    
    return num_from_ref, num_random


def finetune_model(model, train_loader, test_loader, device, epochs=300, lr=0.0003, patience=30):
    """
    Finetune model after regrowth with early stopping.
    
    Args:
        model: Model to finetune
        train_loader: Training data
        test_loader: Test data
        device: Device
        epochs: Number of epochs
        lr: Learning rate
        patience: Number of epochs without improvement before early stopping
    
    Returns:
        best_accuracy: Best test accuracy achieved
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    best_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    
    print(f"  Finetuning for up to {epochs} epochs (patience: {patience})...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_accuracy = 100.0 * train_correct / train_total
        
        # Evaluation
        test_accuracy = evaluate_model(model, test_loader, device)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | Train: {train_accuracy:.2f}% | "
                  f"Test: {test_accuracy:.2f}% | Best: {best_accuracy:.2f}% (epoch {best_epoch}) | "
                  f"No improvement: {epochs_without_improvement}")
        
        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"    Early stopping at epoch {epoch+1}: No improvement for {patience} epochs")
            print(f"    Best accuracy: {best_accuracy:.2f}% at epoch {best_epoch}")
            break
    
    if epochs_without_improvement < patience:
        print(f"  Finetuning completed. Best accuracy: {best_accuracy:.2f}% at epoch {best_epoch}")
    
    return best_accuracy


def create_model_copy(source_model, model_name, device):
    """Create a proper copy of a pruned model"""
    new_model = model_loader(model_name, device)
    prune_weights_reparam(new_model)
    new_model.load_state_dict(source_model.state_dict())
    return new_model


def analyze_single_layer_regrowth(args):
    """
    Main analysis function: test regrowth on each layer individually
    """

    wandb.init(
        project="single-layer-regrowth",
        name=f"{args.m_name}_s{args.start_sparsity}_t{args.target_sparsity}_{args.starting_checkpoint}",
        config=vars(args)
    )

    print("="*80)
    print("Single-Layer Regrowth Analysis")
    print("="*80)
    print(f"Model: {args.m_name}")
    print(f"Start sparsity: {args.start_sparsity:.2%}, Target sparsity: {args.target_sparsity:.2%}")
    print(f"Strategy: Apply ENTIRE global budget to each layer individually")
    print(f"Finetune epochs: {args.finetune_epochs}")
    print(f"Random seed: {args.seed}")
    print("="*80)
    
    # Set seed
    set_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = data_loader(data_dir=args.data_dir)
    
    # Load pretrained model
    print("\nLoading pretrained model...")
    model_pretrained = model_loader(args.m_name, device)
#    load_model(model_pretrained, f'./{args.m_name}/checkpoint')
    load_model_name(model_pretrained, f'./{args.m_name}/checkpoint', args.m_name)
    # Add masks to pretrained for SSIM computation
    prune_weights_reparam(model_pretrained)
    
    # Load reference model (model_95 or model_97)
    print(f"\nLoading reference model (target sparsity: {args.target_sparsity})...")
    model_reference = model_loader(args.m_name, device)
    prune_weights_reparam(model_reference)
    
    if args.target_sparsity == 0.0:
        # Use pretrained as reference
        model_reference.load_state_dict(model_pretrained.state_dict())
        print("  Using pretrained model as reference (0% sparsity)")
    else:
        # Load from checkpoint
        if args.starting_checkpoint == 'oneshot':
            checkpoint_path = f'./{args.m_name}/ckpt_after_prune_oneshot/pruned_oneshot_mask_{args.target_sparsity}.pth'
        else:
            # Map sparsity to iteration number (rough approximation)
            iteration = int(args.target_sparsity * 10)
            checkpoint_path = f'./iterative_0.4_10/{args.m_name}/pruned_finetuned_mask_it{iteration}.pth'
        
        print(f"  Loading from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model_reference.load_state_dict(checkpoint)
    
    # Load starting model (high sparsity, e.g., 99%)
    print(f"\nLoading starting model (sparsity: {args.start_sparsity})...")
    model_start = model_loader(args.m_name, device)
    prune_weights_reparam(model_start)
    
    if args.starting_checkpoint == 'oneshot':
        if args.m_name == "resnet20":
            checkpoint_path = f'./{args.m_name}/ckpt_after_prune_oneshot/pruned_oneshot_mask_{args.start_sparsity}.pth'
        elif args.m_name == "vgg16":
            checkpoint_path = f'./{args.m_name}/ckpt_after_prune_oneshot/pruned_oneshot_mask_{args.start_sparsity}.pth'
        else:
            checkpoint_path = f'./{args.m_name}/ckpt_after_prune_oneshot/pruned_oneshot_mask_{args.start_sparsity}.pth'
    else:
        checkpoint_path = f'./{args.m_name}/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_{args.start_sparsity}.pth'
    
    print(f"  Loading from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model_start.load_state_dict(checkpoint)
    
    # Define target layers
    # if args.m_name == "resnet20":
    #     target_layers = ["layer2.1.conv2", "layer2.2.conv1", "layer2.2.conv2", "layer3.0.conv1",
    #                     "layer3.0.conv2", "layer3.0.shortcut.0", "layer3.1.conv1", "layer3.1.conv2",
    #                     "layer3.2.conv1", "layer3.2.conv2", "linear"]
    # elif args.m_name == "vgg16":
    #     target_layers = ["features.14", "features.17", "features.27", 
    #                     "features.30", "features.34", "features.40", "classifier"]
    # elif args.m_name == "alexnet":
    #     target_layers = ['features.3', 'features.6', 'features.8', 'features.10', 'classifier.1']
    # else:
    
    # For undefined models, use all layers with weight_mask
    print(f"Model {args.m_name} not predefined, auto-detecting layers with masks...")
    target_layers = []
    for name, module in model_start.named_modules():
        if hasattr(module, 'weight_mask') and len(name) > 0:
            target_layers.append(name)
    
    if len(target_layers) == 0:
        print(f"Error: No layers with weight_mask found in model {args.m_name}")
        return
    
    print(f"Found {len(target_layers)} layers with masks")
    
    print(f"\nTarget layers ({len(target_layers)}): {target_layers}")
    
    # Compute SSIM scores
    layer_ssim = compute_layer_ssim_scores(model_start, model_pretrained, target_layers, test_loader)
    
    print("\nSSIM scores (lower = more degraded):")
    for layer_name in target_layers:
        ssim_val = layer_ssim.get(layer_name, 0.5)
        print(f"  {layer_name}: {ssim_val:.4f}")
    
    # Evaluate baseline (no regrowth)
    print("\n" + "-"*80)
    print("Baseline Evaluation (no regrowth)")
    print("-"*80)
    baseline_accuracy = evaluate_model(model_start, test_loader, device)
    baseline_sparsity, total_params, pruned_params = calculate_sparsity(model_start)
    print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
    print(f"Baseline sparsity: {baseline_sparsity:.2f}%")
    print(f"Total parameters: {total_params}, Pruned: {pruned_params}")
    
    # Calculate global regrowth budget
    # Budget = (start_sparsity - target_sparsity) × total_parameters
    sparsity_reduction = args.start_sparsity - args.target_sparsity
    global_regrowth_budget = int(sparsity_reduction * total_params)
    
    print(f"\nGlobal regrowth budget:")
    print(f"  Sparsity reduction: {args.start_sparsity:.2%} → {args.target_sparsity:.2%} = {sparsity_reduction:.2%}")
    print(f"  Total budget: {global_regrowth_budget} weights")
    print(f"  This entire budget will be applied to EACH layer individually for testing")
    
    # Extract reference weights and masks
    reference_weights = {}
    reference_masks = {}
    for layer_name in target_layers:
        module_ref = dict(model_reference.named_modules())[layer_name]
        if hasattr(module_ref, 'weight_orig'):
            reference_weights[layer_name] = module_ref.weight_orig.data.clone()
        elif hasattr(module_ref, 'weight'):
            reference_weights[layer_name] = module_ref.weight.data.clone()
        
        if hasattr(module_ref, 'weight_mask'):
            reference_masks[layer_name] = module_ref.weight_mask.clone()
        else:
            reference_masks[layer_name] = torch.ones_like(reference_weights[layer_name])
    
    # Test each layer individually
    results = []
    
    print("\n" + "="*80)
    print("Testing Individual Layer Regrowth")
    print("="*80)
    
    for i, layer_name in enumerate(target_layers):
        print(f"\n[{i+1}/{len(target_layers)}] Testing layer: {layer_name}")
        print("-"*80)
        
        # Create fresh copy for this test
        model_test = create_model_copy(model_start, args.m_name, device)
        
        # Get layer info
        module = dict(model_test.named_modules())[layer_name]
        if not hasattr(module, 'weight_mask'):
            print(f"  Skipping {layer_name}: No weight mask")
            continue
        
        current_mask = module.weight_mask
        ref_mask = reference_masks[layer_name]
        
        # Calculate capacity for this layer
        regrowable_from_ref = (current_mask == 0) & (ref_mask == 1)
        capacity_from_ref = regrowable_from_ref.sum().item()
        
        regrowable_random = (current_mask == 0) & (ref_mask == 0)
        capacity_random = regrowable_random.sum().item()
        
        total_capacity = capacity_from_ref + capacity_random
        
        # Calculate total weights and current active in this layer
        total_weights = current_mask.numel()
        current_active = (current_mask == 1).sum().item()
        current_sparsity = 1.0 - (current_active / total_weights)
        
        print(f"  SSIM score: {layer_ssim[layer_name]:.4f}")
        print(f"  Total weights: {total_weights}")
        print(f"  Current active: {current_active} (sparsity: {current_sparsity:.2%})")
        print(f"  Regrowth budget (global): {global_regrowth_budget} weights")
        print(f"  Layer capacity - from reference: {capacity_from_ref}, random: {capacity_random}, total: {total_capacity}")
        
        if total_capacity == 0:
            print(f"  Skipping {layer_name}: No capacity to regrow")
            continue
        
        # Use global budget (will be capped by layer capacity)
        layer_regrow_budget = min(global_regrowth_budget, total_capacity)
        
        print(f"  Applying {layer_regrow_budget} weights to this layer")
        
        # Apply regrowth to this layer only
        num_from_ref, num_random = apply_regrowth_to_layer(
            model_test, 
            layer_name, 
            layer_regrow_budget,
            reference_weights[layer_name],
            reference_masks[layer_name]
        )
        
        actual_regrown = num_from_ref + num_random
        
        if actual_regrown == 0:
            print(f"  Skipping {layer_name}: Failed to regrow any weights")
            continue
        
        # Evaluate immediately after regrowth
        after_regrow_accuracy = evaluate_model(model_test, test_loader, device)
        after_regrow_sparsity, _, _ = calculate_sparsity(model_test)
        
        print(f"  After regrowth (before finetune):")
        print(f"    Accuracy: {after_regrow_accuracy:.2f}% (Δ{after_regrow_accuracy - baseline_accuracy:+.2f}%)")
        print(f"    Sparsity: {after_regrow_sparsity:.2f}%")
        
        # Finetune
        best_accuracy = finetune_model(
            model_test, 
            train_loader, 
            test_loader, 
            device,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr
        )
        
        # Calculate improvements
        immediate_improvement = after_regrow_accuracy - baseline_accuracy
        final_improvement = best_accuracy - baseline_accuracy
        
        print(f"  Summary for {layer_name}:")
        print(f"    Baseline: {baseline_accuracy:.2f}%")
        print(f"    After regrowth: {after_regrow_accuracy:.2f}% (Δ{immediate_improvement:+.2f}%)")
        print(f"    After finetune: {best_accuracy:.2f}% (Δ{final_improvement:+.2f}%)")
        print(f"    SSIM score: {layer_ssim[layer_name]:.4f}")
        
        # Store results
        results.append({
            'layer_name': layer_name,
            'ssim_score': layer_ssim[layer_name],
            'total_capacity': total_capacity,
            'capacity_from_ref': capacity_from_ref,
            'capacity_random': capacity_random,
            'weights_regrown': actual_regrown,
            'from_reference': num_from_ref,
            'random_init': num_random,
            'baseline_accuracy': baseline_accuracy,
            'after_regrow_accuracy': after_regrow_accuracy,
            'after_finetune_accuracy': best_accuracy,
            'immediate_improvement': immediate_improvement,
            'final_improvement': final_improvement,
            'sparsity_after_regrow': after_regrow_sparsity,
        })
    
    # Analysis and visualization
    print("\n" + "="*80)
    print("Analysis Results")
    print("="*80)
    
    if len(results) == 0:
        print("Error: No results collected!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by SSIM (lower SSIM = higher degradation = should benefit more from regrowth)
    df_sorted_ssim = df.sort_values('ssim_score')
    
    # Sort by improvement
    df_sorted_improvement = df.sort_values('final_improvement', ascending=False)
    
    print("\nLayers sorted by SSIM score (lower = more degraded):")
    print(df_sorted_ssim[['layer_name', 'ssim_score', 'final_improvement']].to_string(index=False))
    
    print("\nLayers sorted by improvement (higher = more beneficial):")
    print(df_sorted_improvement[['layer_name', 'ssim_score', 'final_improvement']].to_string(index=False))
    
    # Correlation analysis
    print("\n" + "-"*80)
    print("Correlation Analysis: SSIM vs Improvement")
    print("-"*80)
    
    ssim_scores = df['ssim_score'].values
    final_improvements = df['final_improvement'].values
    immediate_improvements = df['immediate_improvement'].values
    
    # Pearson correlation (linear)
    pearson_final, p_value_final = pearsonr(ssim_scores, final_improvements)
    pearson_immediate, p_value_immediate = pearsonr(ssim_scores, immediate_improvements)
    
    # Spearman correlation (rank-based, more robust)
    spearman_final, sp_value_final = spearmanr(ssim_scores, final_improvements)
    spearman_immediate, sp_value_immediate = spearmanr(ssim_scores, immediate_improvements)
    
    print(f"\nPearson correlation (SSIM vs Final Improvement):")
    print(f"  r = {pearson_final:.4f}, p-value = {p_value_final:.4f}")
    print(f"  {'Significant' if p_value_final < 0.05 else 'Not significant'} at α=0.05")
    
    print(f"\nPearson correlation (SSIM vs Immediate Improvement):")
    print(f"  r = {pearson_immediate:.4f}, p-value = {p_value_immediate:.4f}")
    print(f"  {'Significant' if p_value_immediate < 0.05 else 'Not significant'} at α=0.05")
    
    print(f"\nSpearman correlation (SSIM vs Final Improvement):")
    print(f"  ρ = {spearman_final:.4f}, p-value = {sp_value_final:.4f}")
    print(f"  {'Significant' if sp_value_final < 0.05 else 'Not significant'} at α=0.05")
    
    print(f"\nSpearman correlation (SSIM vs Immediate Improvement):")
    print(f"  ρ = {spearman_immediate:.4f}, p-value = {sp_value_immediate:.4f}")
    print(f"  {'Significant' if sp_value_immediate < 0.05 else 'Not significant'} at α=0.05")
    
    # Interpretation
    print("\n" + "-"*80)
    print("Interpretation:")
    print("-"*80)
    
    if pearson_final < -0.3 and p_value_final < 0.05:
        print("✓ STRONG NEGATIVE correlation: Lower SSIM → Better improvement")
        print("  This validates the hypothesis that degraded layers (low SSIM) benefit most from regrowth.")
    elif pearson_final < -0.1:
        print("○ WEAK NEGATIVE correlation: Some evidence that lower SSIM helps")
        print("  The relationship exists but other factors also matter.")
    elif pearson_final > 0.1:
        print("✗ POSITIVE correlation: Counter-intuitive result")
        print("  This suggests SSIM may not be the best priority metric for this model/dataset.")
    else:
        print("○ NO clear correlation")
        print("  SSIM may not be a reliable predictor of regrowth benefit for this model.")
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    
    csv_path = os.path.join(args.save_dir, f'single_layer_regrowth_{args.m_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Visualization
    print("\nGenerating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 10)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: SSIM vs Final Improvement
    ax1 = axes[0, 0]
    ax1.scatter(ssim_scores, final_improvements, s=100, alpha=0.6, c='blue')
    ax1.set_xlabel('SSIM Score', fontsize=12)
    ax1.set_ylabel('Final Improvement (%)', fontsize=12)
    ax1.set_title(f'SSIM vs Final Improvement\n(Pearson r={pearson_final:.3f}, p={p_value_final:.3f})', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(ssim_scores, final_improvements, 1)
    p = np.poly1d(z)
    ax1.plot(ssim_scores, p(ssim_scores), "r--", alpha=0.8, linewidth=2, label='Trend line')
    ax1.legend()
    
    # Annotate points
    for idx, row in df.iterrows():
        ax1.annotate(row['layer_name'].split('.')[-1], 
                    (row['ssim_score'], row['final_improvement']),
                    fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
    # Plot 2: SSIM vs Immediate Improvement
    ax2 = axes[0, 1]
    ax2.scatter(ssim_scores, immediate_improvements, s=100, alpha=0.6, c='green')
    ax2.set_xlabel('SSIM Score', fontsize=12)
    ax2.set_ylabel('Immediate Improvement (%)', fontsize=12)
    ax2.set_title(f'SSIM vs Immediate Improvement\n(Pearson r={pearson_immediate:.3f}, p={p_value_immediate:.3f})', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(ssim_scores, immediate_improvements, 1)
    p = np.poly1d(z)
    ax2.plot(ssim_scores, p(ssim_scores), "r--", alpha=0.8, linewidth=2, label='Trend line')
    ax2.legend()
    
    # Plot 3: Bar chart of improvements sorted by SSIM
    ax3 = axes[1, 0]
    df_sorted = df.sort_values('ssim_score')
    x_pos = np.arange(len(df_sorted))
    ax3.bar(x_pos, df_sorted['final_improvement'], alpha=0.7, color='orange')
    ax3.set_xlabel('Layers (sorted by SSIM, low to high)', fontsize=12)
    ax3.set_ylabel('Final Improvement (%)', fontsize=12)
    ax3.set_title('Improvement by Layer (sorted by SSIM)', fontsize=13)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([name.split('.')[-1] for name in df_sorted['layer_name']], 
                        rotation=45, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Scatter with size = capacity
    ax4 = axes[1, 1]
    scatter = ax4.scatter(ssim_scores, final_improvements, 
                         s=df['total_capacity']/10, 
                         alpha=0.6, c=final_improvements, cmap='RdYlGn')
    ax4.set_xlabel('SSIM Score', fontsize=12)
    ax4.set_ylabel('Final Improvement (%)', fontsize=12)
    ax4.set_title('SSIM vs Improvement (size = capacity)', fontsize=13)
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Improvement (%)')
    
    plt.tight_layout()
    
    plot_path = os.path.join(args.save_dir, f'single_layer_regrowth_{args.m_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_path}")
    
    plt.close()
    
    # Summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print(f"\nModel: {args.m_name}")
    print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
    print(f"Number of layers tested: {len(results)}")
    print(f"\nImprovement statistics:")
    print(f"  Mean: {df['final_improvement'].mean():.2f}%")
    print(f"  Std: {df['final_improvement'].std():.2f}%")
    print(f"  Min: {df['final_improvement'].min():.2f}%")
    print(f"  Max: {df['final_improvement'].max():.2f}%")
    print(f"\nBest layer: {df_sorted_improvement.iloc[0]['layer_name']}")
    print(f"  SSIM: {df_sorted_improvement.iloc[0]['ssim_score']:.4f}")
    print(f"  Improvement: {df_sorted_improvement.iloc[0]['final_improvement']:.2f}%")
    print(f"  Final accuracy: {df_sorted_improvement.iloc[0]['after_finetune_accuracy']:.2f}%")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='Single-Layer Regrowth Analysis')

    # Model and data
    parser.add_argument('--m_name', type=str, default='effnet',
                       help='Model name')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    
    # Regrowth parameters
    parser.add_argument('--target_sparsity', type=float, default=0.0,
                       help='Target sparsity for reference model AND regrowth target (0.0 = pretrained, 0.95, 0.97, etc.)')
    parser.add_argument('--start_sparsity', type=float, default=0.995,
                       help='Starting sparsity (highly pruned model)')
    parser.add_argument('--starting_checkpoint', type=str, default='oneshot',
                       choices=['oneshot', 'iterative'],
                       help='Checkpoint type: oneshot or iterative')
    
    # Finetuning
    parser.add_argument('--finetune_epochs', type=int, default=300,
                       help='Number of finetuning epochs per layer')
    parser.add_argument('--finetune_lr', type=float, default=0.0003,
                       help='Finetuning learning rate')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='./single_layer_analysis',
                       help='Directory to save results')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_single_layer_regrowth(args)


if __name__ == '__main__':
    main()
