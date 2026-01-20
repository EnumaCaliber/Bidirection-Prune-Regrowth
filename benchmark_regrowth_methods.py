"""
Benchmark: Reference-based vs Saliency-based Regrowth

Compares execution time of two regrowth approaches:
1. Reference-based (rl_regrowth_nas.py): Uses SSIM + pre-computed reference masks
2. Saliency-based (rl_saliency_regrowth.py): Uses gradient magnitude scores

Measures:
- SSIM computation time (reference method)
- Saliency computation time (gradient method)
- Weight selection time
- Mask update time
- Mini-finetuning time
- Total episode time
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import argparse
from contextlib import contextmanager
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns

from utils.model_loader import model_loader
from utils.data_loader import data_loader
from utils.analysis_utils import (
    BlockwiseFeatureExtractor, compute_block_ssim,
    load_model, prune_weights_reparam, count_pruned_params
)


class TimingContext:
    """Context manager for precise timing measurement"""
    
    def __init__(self, name, device='cuda', warmup=False):
        self.name = name
        self.device = device
        self.warmup = warmup
        self.start_time = None
        self.end_time = None
        self.duration = None
        
    def __enter__(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        
        if not self.warmup:
            print(f"  [{self.name}] Time: {self.duration:.4f}s")
        
        return False


class RegrowthBenchmark:
    """Benchmark suite for comparing regrowth methods"""
    
    def __init__(self, model_name, device='cuda', seed=42):
        self.model_name = model_name
        self.device = device
        self.seed = seed
        
        # Set seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        # Load data
        print("Loading data...")
        self.train_loader, self.val_loader, self.test_loader = data_loader(data_dir='./data')
        
        # Load models
        print("Loading models...")
        self.model_pretrained = model_loader(model_name, device)
        load_model(self.model_pretrained, f'./{model_name}/checkpoint')
        
        self.model_99 = model_loader(model_name, device)
        prune_weights_reparam(self.model_99)
        
        # Load pruned checkpoint
        if model_name == 'resnet20':
            checkpoint_99 = torch.load(f'./{model_name}/ckpt_after_prune/pruned_finetuned_mask_0.99.pth')
            self.target_layers = ["layer2.0.conv2", "layer2.1.conv1", "layer2.2.conv2", 
                                  "layer2.2.conv1", "layer3.0.conv2", "layer3.0.conv1", 
                                  "layer3.1.conv1", "layer3.1.conv2"]
        elif model_name == 'vgg16':
            checkpoint_99 = torch.load(f'./{model_name}/ckpt_after_prune/pruned_finetuned_mask_0.99.pth')
            self.target_layers = ["features.14", "features.10", "features.17", "features.7", 
                                 "features.24", "features.20", "features.0", "features.27", 
                                 "features.3", "classifier"]
        elif model_name == 'alexnet':
            checkpoint_99 = torch.load(f'./{model_name}/ckpt_after_prune/pruned_finetuned_mask_0.99.pth')
            self.target_layers = ['features.3', 'features.6', 'features.8', 
                                 'features.10', 'classifier.1']
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_99.load_state_dict(checkpoint_99)
        
        # Calculate regrowth budget
        total_weights, _, _ = count_pruned_params(self.model_99)
        self.target_regrow = int(total_weights * 0.01)  # 1% regrowth
        
        # Get layer capacities
        self.layer_capacities = []
        module_dict = dict(self.model_99.named_modules())
        for layer_name in self.target_layers:
            module = module_dict[layer_name]
            if hasattr(module, 'weight_mask'):
                capacity = int((module.weight_mask == 0).sum().item())
                self.layer_capacities.append(capacity)
            else:
                self.layer_capacities.append(0)
        
        print(f"\nBenchmark configuration:")
        print(f"  Model: {model_name}")
        print(f"  Target layers: {len(self.target_layers)}")
        print(f"  Target regrowth: {self.target_regrow:,} weights")
        print(f"  Total capacity: {sum(self.layer_capacities):,} weights")
        print(f"  Device: {device}")
        
        # Results storage
        self.results = {
            'reference_based': defaultdict(list),
            'saliency_based': defaultdict(list)
        }
    
    def benchmark_ssim_computation(self, num_batches=20):
        """
        Benchmark SSIM computation (used in reference-based method)
        
        Process:
        1. Extract features from pretrained model
        2. Extract features from pruned model
        3. Compute SSIM scores
        """
        print("\n" + "="*70)
        print("Benchmarking SSIM Computation (Reference-based Method)")
        print("="*70)
        
        block_dict = {'target_block': self.target_layers}
        
        # Feature extraction from pretrained model
        with TimingContext("SSIM: Extract features (pretrained)", self.device) as timer:
            extractor_pretrained = BlockwiseFeatureExtractor(self.model_pretrained, block_dict)
            features_pretrained = extractor_pretrained.extract_block_features(
                self.test_loader, num_batches=num_batches
            )
        time_extract_pretrained = timer.duration
        
        # Feature extraction from pruned model
        with TimingContext("SSIM: Extract features (pruned)", self.device) as timer:
            extractor_pruned = BlockwiseFeatureExtractor(self.model_99, block_dict)
            features_pruned = extractor_pruned.extract_block_features(
                self.test_loader, num_batches=num_batches
            )
        time_extract_pruned = timer.duration
        
        # SSIM computation
        with TimingContext("SSIM: Compute scores", self.device) as timer:
            ssim_scores = compute_block_ssim(features_pretrained, features_pruned)
        time_compute_ssim = timer.duration
        
        total_time = time_extract_pretrained + time_extract_pruned + time_compute_ssim
        
        print(f"\n  Total SSIM computation time: {total_time:.4f}s")
        print(f"    - Extract pretrained features: {time_extract_pretrained:.4f}s ({100*time_extract_pretrained/total_time:.1f}%)")
        print(f"    - Extract pruned features: {time_extract_pruned:.4f}s ({100*time_extract_pruned/total_time:.1f}%)")
        print(f"    - Compute SSIM: {time_compute_ssim:.4f}s ({100*time_compute_ssim/total_time:.1f}%)")
        
        return {
            'total': total_time,
            'extract_pretrained': time_extract_pretrained,
            'extract_pruned': time_extract_pruned,
            'compute_ssim': time_compute_ssim,
            'ssim_scores': ssim_scores
        }
    
    def benchmark_saliency_computation(self, num_batches=50):
        """
        Benchmark saliency computation (used in gradient-based method)
        
        Process:
        1. Forward pass on multiple batches
        2. Backward pass to get gradients
        3. Accumulate squared gradients
        4. Average and return saliency scores
        """
        print("\n" + "="*70)
        print("Benchmarking Saliency Computation (Gradient-based Method)")
        print("="*70)
        
        self.model_pretrained.eval()
        criterion = nn.CrossEntropyLoss()
        
        # Initialize storage
        module_dict = dict(self.model_pretrained.named_modules())
        accumulated_grads = {}
        
        for layer_name in self.target_layers:
            module = module_dict.get(layer_name)
            if module is not None and hasattr(module, 'weight'):
                accumulated_grads[layer_name] = torch.zeros_like(module.weight.data)
        
        # Time gradient accumulation
        time_forward = 0.0
        time_backward = 0.0
        time_accumulate = 0.0
        
        batch_count = 0
        for inputs, labels in self.train_loader:
            if batch_count >= num_batches:
                break
            
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            with TimingContext("Saliency: Forward pass", self.device, warmup=True) as timer:
                self.model_pretrained.zero_grad()
                outputs = self.model_pretrained(inputs)
                loss = criterion(outputs, labels)
            time_forward += timer.duration
            
            # Backward pass
            with TimingContext("Saliency: Backward pass", self.device, warmup=True) as timer:
                grads = torch.autograd.grad(loss, self.model_pretrained.parameters(), 
                                           create_graph=False)
            time_backward += timer.duration
            
            # Accumulate gradients
            with TimingContext("Saliency: Accumulate grads", self.device, warmup=True) as timer:
                for param, grad in zip(self.model_pretrained.parameters(), grads):
                    for layer_name in self.target_layers:
                        module = module_dict[layer_name]
                        if hasattr(module, 'weight') and param is module.weight:
                            accumulated_grads[layer_name] += grad.detach() ** 2
            time_accumulate += timer.duration
            
            batch_count += 1
        
        # Average gradients
        with TimingContext("Saliency: Average scores", self.device) as timer:
            saliency_dict = {}
            for layer_name in self.target_layers:
                if layer_name in accumulated_grads:
                    saliency_dict[layer_name] = accumulated_grads[layer_name] / batch_count
        time_average = timer.duration
        
        total_time = time_forward + time_backward + time_accumulate + time_average
        
        print(f"\n  Total saliency computation time: {total_time:.4f}s")
        print(f"    - Forward passes ({batch_count} batches): {time_forward:.4f}s ({100*time_forward/total_time:.1f}%)")
        print(f"    - Backward passes: {time_backward:.4f}s ({100*time_backward/total_time:.1f}%)")
        print(f"    - Gradient accumulation: {time_accumulate:.4f}s ({100*time_accumulate/total_time:.1f}%)")
        print(f"    - Averaging: {time_average:.4f}s ({100*time_average/total_time:.1f}%)")
        
        return {
            'total': total_time,
            'forward': time_forward,
            'backward': time_backward,
            'accumulate': time_accumulate,
            'average': time_average,
            'saliency_scores': saliency_dict
        }
    
    def benchmark_reference_based_selection(self, allocation, ssim_scores):
        """
        Benchmark reference-based weight selection
        
        Process:
        1. Load reference masks and weights
        2. For each layer, sort reference positions by similarity to pretrained
        3. Select top-K positions
        """
        print("\n" + "="*70)
        print("Benchmarking Reference-based Weight Selection")
        print("="*70)
        
        # Load reference model (0.95 sparsity)
        model_95 = model_loader(self.model_name, self.device)
        prune_weights_reparam(model_95)
        
        if self.model_name == 'resnet20':
            checkpoint_95 = torch.load(f'./{self.model_name}/ckpt_after_prune/pruned_finetuned_mask_0.95.pth')
        elif self.model_name == 'vgg16':
            checkpoint_95 = torch.load(f'./{self.model_name}/ckpt_after_prune/pruned_finetuned_mask_0.95.pth')
        elif self.model_name == 'alexnet':
            checkpoint_95 = torch.load(f'./{self.model_name}/ckpt_after_prune/pruned_finetuned_mask_0.95.pth')
        
        model_95.load_state_dict(checkpoint_95)
        
        # Time reference preparation
        with TimingContext("Reference: Load masks and weights", self.device) as timer:
            reference_masks = {}
            reference_weights = {}
            module_dict_95 = dict(model_95.named_modules())
            module_dict_pretrained = dict(self.model_pretrained.named_modules())
            
            for layer_name in self.target_layers:
                module_95 = module_dict_95[layer_name]
                module_pretrained = module_dict_pretrained[layer_name]
                
                if hasattr(module_95, 'weight_mask'):
                    reference_masks[layer_name] = module_95.weight_mask.clone()
                    reference_weights[layer_name] = module_pretrained.weight.data.clone()
        time_load = timer.duration
        
        # Time weight selection
        time_selection = 0.0
        total_selected = 0
        
        for layer_name, num_weights in allocation.items():
            if num_weights == 0:
                continue
            
            with TimingContext(f"Reference: Select {num_weights} weights from {layer_name}", 
                             self.device, warmup=True) as timer:
                # Get reference positions (where mask == 1 in 0.95 model)
                ref_mask = reference_masks[layer_name]
                ref_weights = reference_weights[layer_name]
                
                ref_positions = (ref_mask == 1)
                
                # Get corresponding weights from pretrained model
                ref_weight_values = ref_weights[ref_positions]
                
                # Rank by absolute magnitude (similarity to pretrained)
                abs_values = torch.abs(ref_weight_values)
                _, sorted_indices = torch.sort(abs_values, descending=True)
                
                # Select top-K
                k = min(num_weights, len(sorted_indices))
                selected = sorted_indices[:k]
                
            time_selection += timer.duration
            total_selected += k
        
        total_time = time_load + time_selection
        
        print(f"\n  Total reference-based selection time: {total_time:.4f}s")
        print(f"    - Load reference masks/weights: {time_load:.4f}s ({100*time_load/total_time:.1f}%)")
        print(f"    - Weight selection: {time_selection:.4f}s ({100*time_selection/total_time:.1f}%)")
        print(f"    - Total weights selected: {total_selected}")
        
        return {
            'total': total_time,
            'load': time_load,
            'selection': time_selection,
            'num_selected': total_selected
        }
    
    def benchmark_saliency_based_selection(self, allocation, saliency_scores):
        """
        Benchmark saliency-based weight selection
        
        Process:
        1. For each layer, mask out active weights in saliency scores
        2. Sort pruned positions by saliency
        3. Select top-K
        """
        print("\n" + "="*70)
        print("Benchmarking Saliency-based Weight Selection")
        print("="*70)
        
        time_selection = 0.0
        total_selected = 0
        
        module_dict = dict(self.model_99.named_modules())
        
        for layer_name, num_weights in allocation.items():
            if num_weights == 0:
                continue
            
            with TimingContext(f"Saliency: Select {num_weights} weights from {layer_name}", 
                             self.device, warmup=True) as timer:
                module = module_dict[layer_name]
                current_mask = module.weight_mask
                saliency = saliency_scores[layer_name]
                
                # Find pruned positions
                pruned_positions = (current_mask == 0)
                
                # Mask saliency (exclude active weights)
                saliency_masked = saliency.clone()
                saliency_masked[~pruned_positions] = -float('inf')
                
                # Flatten and sort
                flat_saliency = saliency_masked.flatten()
                k = min(num_weights, (flat_saliency > -float('inf')).sum().item())
                
                # Get top-K indices
                _, top_k_flat_indices = torch.topk(flat_saliency, k=k)
                
            time_selection += timer.duration
            total_selected += k
        
        print(f"\n  Total saliency-based selection time: {time_selection:.4f}s")
        print(f"    - Weight selection: {time_selection:.4f}s (100.0%)")
        print(f"    - Total weights selected: {total_selected}")
        
        return {
            'total': time_selection,
            'selection': time_selection,
            'num_selected': total_selected
        }
    
    def benchmark_mask_update(self, allocation):
        """
        Benchmark mask update operation
        
        Process:
        1. Create dummy indices for regrowth
        2. Update masks for each layer
        """
        print("\n" + "="*70)
        print("Benchmarking Mask Update")
        print("="*70)
        
        # Create model copy
        model_copy = model_loader(self.model_name, self.device)
        prune_weights_reparam(model_copy)
        model_copy.load_state_dict(self.model_99.state_dict())
        
        module_dict = dict(model_copy.named_modules())
        
        time_update = 0.0
        total_updated = 0
        
        for layer_name, num_weights in allocation.items():
            if num_weights == 0:
                continue
            
            module = module_dict[layer_name]
            
            with TimingContext(f"Mask update: {layer_name} ({num_weights} weights)", 
                             self.device, warmup=True) as timer:
                # Find pruned positions
                current_mask = module.weight_mask
                pruned_positions = (current_mask == 0)
                pruned_indices = torch.nonzero(pruned_positions, as_tuple=False)
                
                # Select random subset
                k = min(num_weights, len(pruned_indices))
                perm = torch.randperm(len(pruned_indices))[:k]
                selected_indices = pruned_indices[perm]
                
                # Update mask
                for idx_tuple in selected_indices:
                    idx_tuple = tuple(idx_tuple.tolist())
                    module.weight_mask[idx_tuple] = 1.0
                    
            time_update += timer.duration
            total_updated += k
        
        print(f"\n  Total mask update time: {time_update:.4f}s")
        print(f"    - Total weights updated: {total_updated}")
        
        return {
            'total': time_update,
            'num_updated': total_updated
        }
    
    def benchmark_mini_finetune(self, epochs=50):
        """
        Benchmark mini-finetuning
        
        Process:
        1. Create model copy
        2. Run finetuning for specified epochs
        3. Measure time per epoch and total time
        """
        print("\n" + "="*70)
        print(f"Benchmarking Mini-Finetuning ({epochs} epochs)")
        print("="*70)
        
        # Create model copy
        model_copy = model_loader(self.model_name, self.device)
        prune_weights_reparam(model_copy)
        model_copy.load_state_dict(self.model_99.state_dict())
        
        model_copy.train()
        optimizer = optim.AdamW(model_copy.parameters(), lr=0.0003, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        epoch_times = []
        
        with TimingContext("Mini-finetuning: Total", self.device) as timer_total:
            for epoch in range(epochs):
                with TimingContext(f"Mini-finetuning: Epoch {epoch+1}", self.device, warmup=True) as timer_epoch:
                    for inputs, targets in self.train_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        optimizer.zero_grad()
                        outputs = model_copy(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                
                epoch_times.append(timer_epoch.duration)
                
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}: {timer_epoch.duration:.4f}s")
        
        total_time = timer_total.duration
        avg_epoch_time = np.mean(epoch_times)
        
        print(f"\n  Total mini-finetuning time: {total_time:.4f}s")
        print(f"    - Average time per epoch: {avg_epoch_time:.4f}s")
        print(f"    - Std dev: {np.std(epoch_times):.4f}s")
        
        return {
            'total': total_time,
            'avg_epoch': avg_epoch_time,
            'std_epoch': np.std(epoch_times),
            'epoch_times': epoch_times
        }
    
    def create_dummy_allocation(self):
        """Create a dummy allocation for benchmarking"""
        allocation = {}
        remaining = self.target_regrow
        
        for i, layer_name in enumerate(self.target_layers):
            if remaining <= 0:
                allocation[layer_name] = 0
            else:
                # Distribute budget proportionally to capacity
                layer_capacity = self.layer_capacities[i]
                total_capacity = sum(self.layer_capacities)
                layer_fraction = layer_capacity / total_capacity if total_capacity > 0 else 0
                layer_budget = int(self.target_regrow * layer_fraction)
                layer_budget = min(layer_budget, remaining, layer_capacity)
                allocation[layer_name] = layer_budget
                remaining -= layer_budget
        
        return allocation
    
    def run_full_benchmark(self, num_runs=3):
        """Run complete benchmark comparing both methods"""
        print("\n" + "="*80)
        print("FULL BENCHMARK: Reference-based vs Saliency-based Regrowth")
        print("="*80)
        print(f"Number of runs: {num_runs}")
        print("="*80)
        
        allocation = self.create_dummy_allocation()
        print(f"\nDummy allocation created:")
        for layer_name, num_weights in allocation.items():
            print(f"  {layer_name}: {num_weights} weights")
        
        # Run multiple times and average
        for run in range(num_runs):
            print(f"\n{'='*80}")
            print(f"RUN {run + 1}/{num_runs}")
            print(f"{'='*80}")
            
            # ============================================================
            # Reference-based method
            # ============================================================
            print("\n" + "="*70)
            print("REFERENCE-BASED METHOD")
            print("="*70)
            
            # 1. SSIM computation
            ssim_result = self.benchmark_ssim_computation(num_batches=20)
            self.results['reference_based']['ssim_total'].append(ssim_result['total'])
            
            # 2. Weight selection
            selection_result = self.benchmark_reference_based_selection(
                allocation, ssim_result['ssim_scores']
            )
            self.results['reference_based']['selection_total'].append(selection_result['total'])
            
            # 3. Mask update
            mask_result = self.benchmark_mask_update(allocation)
            self.results['reference_based']['mask_update'].append(mask_result['total'])
            
            # 4. Mini-finetuning
            finetune_result = self.benchmark_mini_finetune(epochs=50)
            self.results['reference_based']['finetune_total'].append(finetune_result['total'])
            
            # Total for reference-based
            ref_total = (ssim_result['total'] + selection_result['total'] + 
                        mask_result['total'] + finetune_result['total'])
            self.results['reference_based']['total'].append(ref_total)
            
            print(f"\n  Reference-based TOTAL: {ref_total:.4f}s")
            
            # ============================================================
            # Saliency-based method
            # ============================================================
            print("\n" + "="*70)
            print("SALIENCY-BASED METHOD")
            print("="*70)
            
            # 1. Saliency computation
            saliency_result = self.benchmark_saliency_computation(num_batches=50)
            self.results['saliency_based']['saliency_total'].append(saliency_result['total'])
            
            # 2. Weight selection
            sal_selection_result = self.benchmark_saliency_based_selection(
                allocation, saliency_result['saliency_scores']
            )
            self.results['saliency_based']['selection_total'].append(sal_selection_result['total'])
            
            # 3. Mask update
            sal_mask_result = self.benchmark_mask_update(allocation)
            self.results['saliency_based']['mask_update'].append(sal_mask_result['total'])
            
            # 4. Mini-finetuning
            sal_finetune_result = self.benchmark_mini_finetune(epochs=50)
            self.results['saliency_based']['finetune_total'].append(sal_finetune_result['total'])
            
            # Total for saliency-based
            sal_total = (saliency_result['total'] + sal_selection_result['total'] + 
                        sal_mask_result['total'] + sal_finetune_result['total'])
            self.results['saliency_based']['total'].append(sal_total)
            
            print(f"\n  Saliency-based TOTAL: {sal_total:.4f}s")
            
            # Comparison
            print(f"\n  {'='*70}")
            print(f"  RUN {run + 1} COMPARISON:")
            print(f"  {'='*70}")
            print(f"  Reference-based: {ref_total:.4f}s")
            print(f"  Saliency-based:  {sal_total:.4f}s")
            speedup = ref_total / sal_total if sal_total > 0 else 0
            if speedup > 1:
                print(f"  Saliency is {speedup:.2f}x FASTER")
            else:
                print(f"  Reference is {1/speedup:.2f}x FASTER")
        
        # Print summary
        self.print_summary()
        
        # Generate visualizations
        self.plot_results()
        
        # Save results
        self.save_results()
    
    def print_summary(self):
        """Print benchmark summary statistics"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY (Average over all runs)")
        print("="*80)
        
        # Reference-based
        print("\nReference-based Method:")
        print("-" * 80)
        ref_results = self.results['reference_based']
        
        ssim_mean = np.mean(ref_results['ssim_total'])
        selection_mean = np.mean(ref_results['selection_total'])
        mask_mean = np.mean(ref_results['mask_update'])
        finetune_mean = np.mean(ref_results['finetune_total'])
        ref_total_mean = np.mean(ref_results['total'])
        
        print(f"  SSIM computation:    {ssim_mean:8.4f}s  ({100*ssim_mean/ref_total_mean:5.1f}%)")
        print(f"  Weight selection:    {selection_mean:8.4f}s  ({100*selection_mean/ref_total_mean:5.1f}%)")
        print(f"  Mask update:         {mask_mean:8.4f}s  ({100*mask_mean/ref_total_mean:5.1f}%)")
        print(f"  Mini-finetuning:     {finetune_mean:8.4f}s  ({100*finetune_mean/ref_total_mean:5.1f}%)")
        print(f"  {'-'*80}")
        print(f"  TOTAL:               {ref_total_mean:8.4f}s  (100.0%)")
        
        # Saliency-based
        print("\nSaliency-based Method:")
        print("-" * 80)
        sal_results = self.results['saliency_based']
        
        saliency_mean = np.mean(sal_results['saliency_total'])
        sal_selection_mean = np.mean(sal_results['selection_total'])
        sal_mask_mean = np.mean(sal_results['mask_update'])
        sal_finetune_mean = np.mean(sal_results['finetune_total'])
        sal_total_mean = np.mean(sal_results['total'])
        
        print(f"  Saliency computation: {saliency_mean:8.4f}s  ({100*saliency_mean/sal_total_mean:5.1f}%)")
        print(f"  Weight selection:     {sal_selection_mean:8.4f}s  ({100*sal_selection_mean/sal_total_mean:5.1f}%)")
        print(f"  Mask update:          {sal_mask_mean:8.4f}s  ({100*sal_mask_mean/sal_total_mean:5.1f}%)")
        print(f"  Mini-finetuning:      {sal_finetune_mean:8.4f}s  ({100*sal_finetune_mean/sal_total_mean:5.1f}%)")
        print(f"  {'-'*80}")
        print(f"  TOTAL:                {sal_total_mean:8.4f}s  (100.0%)")
        
        # Comparison
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        print(f"Reference-based total: {ref_total_mean:.4f}s")
        print(f"Saliency-based total:  {sal_total_mean:.4f}s")
        
        speedup = ref_total_mean / sal_total_mean if sal_total_mean > 0 else 0
        if speedup > 1:
            print(f"\n✓ Saliency-based is {speedup:.2f}x FASTER")
            print(f"  Time saved per episode: {ref_total_mean - sal_total_mean:.4f}s")
        else:
            print(f"\n✗ Reference-based is {1/speedup:.2f}x FASTER")
            print(f"  Time overhead per episode: {sal_total_mean - ref_total_mean:.4f}s")
        
        # Key insights
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        
        # Compare preprocessing
        print(f"\n1. Preprocessing comparison:")
        print(f"   - SSIM (reference):     {ssim_mean:.4f}s")
        print(f"   - Saliency (gradient):  {saliency_mean:.4f}s")
        if saliency_mean > ssim_mean:
            print(f"   → Saliency is {saliency_mean/ssim_mean:.2f}x SLOWER (requires gradient computation)")
        else:
            print(f"   → SSIM is {ssim_mean/saliency_mean:.2f}x SLOWER")
        
        # Compare selection
        print(f"\n2. Weight selection comparison:")
        print(f"   - Reference-based: {selection_mean:.4f}s (requires loading reference masks)")
        print(f"   - Saliency-based:  {sal_selection_mean:.4f}s (direct ranking)")
        if sal_selection_mean < selection_mean:
            print(f"   → Saliency is {selection_mean/sal_selection_mean:.2f}x FASTER")
        else:
            print(f"   → Reference is {sal_selection_mean/selection_mean:.2f}x FASTER")
        
        # Finetuning is same for both
        print(f"\n3. Mini-finetuning (same for both):")
        print(f"   - Average: {finetune_mean:.4f}s")
        print(f"   - Dominates total time: {100*finetune_mean/ref_total_mean:.1f}%")
        
        print("\n" + "="*80)
    
    def plot_results(self):
        """Generate comparison visualizations"""
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Total time comparison
        ax = axes[0, 0]
        methods = ['Reference-based', 'Saliency-based']
        totals = [
            np.mean(self.results['reference_based']['total']),
            np.mean(self.results['saliency_based']['total'])
        ]
        colors = ['steelblue', 'coral']
        bars = ax.bar(methods, totals, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, total in zip(bars, totals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{total:.2f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Total Time (seconds)', fontsize=12)
        ax.set_title('Total Episode Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Component breakdown (stacked bar)
        ax = axes[0, 1]
        
        ref_components = [
            np.mean(self.results['reference_based']['ssim_total']),
            np.mean(self.results['reference_based']['selection_total']),
            np.mean(self.results['reference_based']['mask_update']),
            np.mean(self.results['reference_based']['finetune_total'])
        ]
        
        sal_components = [
            np.mean(self.results['saliency_based']['saliency_total']),
            np.mean(self.results['saliency_based']['selection_total']),
            np.mean(self.results['saliency_based']['mask_update']),
            np.mean(self.results['saliency_based']['finetune_total'])
        ]
        
        component_labels = ['Preprocessing', 'Selection', 'Mask Update', 'Finetuning']
        x_pos = np.arange(len(methods))
        width = 0.6
        
        bottom_ref = 0
        bottom_sal = 0
        colors_components = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072']
        
        for i, (ref_val, sal_val, label, color) in enumerate(zip(ref_components, sal_components, 
                                                                   component_labels, colors_components)):
            ax.bar(0, ref_val, width, bottom=bottom_ref, label=label if i == 0 else "", 
                  color=color, alpha=0.8)
            ax.bar(1, sal_val, width, bottom=bottom_sal, color=color, alpha=0.8)
            
            # Add text labels for large components
            if ref_val > 1.0:
                ax.text(0, bottom_ref + ref_val/2, f'{ref_val:.1f}s', 
                       ha='center', va='center', fontsize=10, fontweight='bold')
            if sal_val > 1.0:
                ax.text(1, bottom_sal + sal_val/2, f'{sal_val:.1f}s', 
                       ha='center', va='center', fontsize=10, fontweight='bold')
            
            bottom_ref += ref_val
            bottom_sal += sal_val
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Time Breakdown by Component', fontsize=14, fontweight='bold')
        ax.legend(component_labels, loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Preprocessing comparison
        ax = axes[1, 0]
        preprocessing = ['SSIM\n(Reference)', 'Saliency\n(Gradient)']
        preprocessing_times = [
            np.mean(self.results['reference_based']['ssim_total']),
            np.mean(self.results['saliency_based']['saliency_total'])
        ]
        bars = ax.bar(preprocessing, preprocessing_times, color=['steelblue', 'coral'], alpha=0.7)
        
        for bar, time_val in zip(bars, preprocessing_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_val:.2f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Preprocessing Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Speedup visualization
        ax = axes[1, 1]
        ref_total = np.mean(self.results['reference_based']['total'])
        sal_total = np.mean(self.results['saliency_based']['total'])
        speedup = ref_total / sal_total if sal_total > 0 else 0
        
        if speedup > 1:
            # Saliency is faster
            ax.barh(['Speedup'], [speedup], color='green', alpha=0.7)
            ax.text(speedup/2, 0, f'{speedup:.2f}x FASTER', ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white')
            ax.set_xlabel('Speedup Factor (Saliency vs Reference)', fontsize=12)
        else:
            # Reference is faster
            slowdown = 1 / speedup
            ax.barh(['Slowdown'], [slowdown], color='red', alpha=0.7)
            ax.text(slowdown/2, 0, f'{slowdown:.2f}x SLOWER', ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white')
            ax.set_xlabel('Slowdown Factor (Saliency vs Reference)', fontsize=12)
        
        ax.set_title('Performance Ratio', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = f'benchmark_results_{self.model_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved visualization: {save_path}")
        
        plt.show()
    
    def save_results(self):
        """Save benchmark results to JSON"""
        # Convert numpy arrays and other types to JSON-serializable format
        results_json = {
            'model_name': self.model_name,
            'target_regrowth': self.target_regrow,
            'num_runs': len(self.results['reference_based']['total']),
            'reference_based': {
                'mean': {
                    'ssim': float(np.mean(self.results['reference_based']['ssim_total'])),
                    'selection': float(np.mean(self.results['reference_based']['selection_total'])),
                    'mask_update': float(np.mean(self.results['reference_based']['mask_update'])),
                    'finetune': float(np.mean(self.results['reference_based']['finetune_total'])),
                    'total': float(np.mean(self.results['reference_based']['total']))
                },
                'std': {
                    'ssim': float(np.std(self.results['reference_based']['ssim_total'])),
                    'selection': float(np.std(self.results['reference_based']['selection_total'])),
                    'mask_update': float(np.std(self.results['reference_based']['mask_update'])),
                    'finetune': float(np.std(self.results['reference_based']['finetune_total'])),
                    'total': float(np.std(self.results['reference_based']['total']))
                }
            },
            'saliency_based': {
                'mean': {
                    'saliency': float(np.mean(self.results['saliency_based']['saliency_total'])),
                    'selection': float(np.mean(self.results['saliency_based']['selection_total'])),
                    'mask_update': float(np.mean(self.results['saliency_based']['mask_update'])),
                    'finetune': float(np.mean(self.results['saliency_based']['finetune_total'])),
                    'total': float(np.mean(self.results['saliency_based']['total']))
                },
                'std': {
                    'saliency': float(np.std(self.results['saliency_based']['saliency_total'])),
                    'selection': float(np.std(self.results['saliency_based']['selection_total'])),
                    'mask_update': float(np.std(self.results['saliency_based']['mask_update'])),
                    'finetune': float(np.std(self.results['saliency_based']['finetune_total'])),
                    'total': float(np.std(self.results['saliency_based']['total']))
                }
            }
        }
        
        # Calculate speedup
        ref_total = results_json['reference_based']['mean']['total']
        sal_total = results_json['saliency_based']['mean']['total']
        results_json['speedup'] = ref_total / sal_total if sal_total > 0 else 0
        
        save_path = f'benchmark_results_{self.model_name}.json'
        with open(save_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"  ✓ Saved results: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark regrowth methods')
    parser.add_argument('--m_name', type=str, default='resnet20',
                       choices=['resnet20', 'vgg16', 'alexnet'])
    parser.add_argument('--num_runs', type=int, default=3,
                       help='Number of benchmark runs')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = RegrowthBenchmark(
        model_name=args.m_name,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=args.seed
    )
    
    # Run full benchmark
    benchmark.run_full_benchmark(num_runs=args.num_runs)


if __name__ == '__main__':
    main()
