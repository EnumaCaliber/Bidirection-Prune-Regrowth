"""
Quick Single-Episode Benchmark

Simulates ONE sampling episode for both methods to compare execution time.
This is useful for quick testing and understanding the time breakdown.
"""

import torch
import argparse
import sys

# Add parent directory to path
sys.path.append('/home/j/junchen/DAC26_Final')

from benchmark_regrowth_methods import RegrowthBenchmark


def quick_benchmark(model_name='vgg16'):
    """Run a single episode benchmark for quick comparison"""
    
    print("\n" + "="*80)
    print("QUICK BENCHMARK: Single Episode Comparison")
    print("="*80)
    print(f"Model: {model_name}")
    print("="*80 + "\n")
    
    # Create benchmark
    benchmark = RegrowthBenchmark(
        model_name=model_name,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42
    )
    
    # Create allocation
    allocation = benchmark.create_dummy_allocation()
    
    print("\nAllocation created:")
    total_alloc = 0
    for layer_name, num_weights in allocation.items():
        if num_weights > 0:
            print(f"  {layer_name}: {num_weights:,} weights")
            total_alloc += num_weights
    print(f"  Total: {total_alloc:,} / {benchmark.target_regrow:,} weights")
    
    # ============================================================
    # Reference-based method
    # ============================================================
    print("\n" + "="*70)
    print("REFERENCE-BASED METHOD (Single Episode)")
    print("="*70)
    
    print("\n[1/4] SSIM Computation...")
    ssim_result = benchmark.benchmark_ssim_computation(num_batches=20)
    
    print("\n[2/4] Reference-based Weight Selection...")
    selection_result = benchmark.benchmark_reference_based_selection(
        allocation, ssim_result['ssim_scores']
    )
    
    print("\n[3/4] Mask Update...")
    mask_result = benchmark.benchmark_mask_update(allocation)
    
    print("\n[4/4] Mini-Finetuning (50 epochs)...")
    finetune_result = benchmark.benchmark_mini_finetune(epochs=50)
    
    ref_total = (ssim_result['total'] + selection_result['total'] + 
                mask_result['total'] + finetune_result['total'])
    
    print(f"\n{'='*70}")
    print(f"Reference-based TOTAL: {ref_total:.4f}s")
    print(f"{'='*70}")
    print(f"  SSIM computation:  {ssim_result['total']:8.4f}s  ({100*ssim_result['total']/ref_total:5.1f}%)")
    print(f"  Weight selection:  {selection_result['total']:8.4f}s  ({100*selection_result['total']/ref_total:5.1f}%)")
    print(f"  Mask update:       {mask_result['total']:8.4f}s  ({100*mask_result['total']/ref_total:5.1f}%)")
    print(f"  Mini-finetuning:   {finetune_result['total']:8.4f}s  ({100*finetune_result['total']/ref_total:5.1f}%)")
    
    # ============================================================
    # Saliency-based method
    # ============================================================
    print("\n" + "="*70)
    print("SALIENCY-BASED METHOD (Single Episode)")
    print("="*70)
    
    print("\n[1/4] Saliency Computation...")
    saliency_result = benchmark.benchmark_saliency_computation(num_batches=50)
    
    print("\n[2/4] Saliency-based Weight Selection...")
    sal_selection_result = benchmark.benchmark_saliency_based_selection(
        allocation, saliency_result['saliency_scores']
    )
    
    print("\n[3/4] Mask Update...")
    sal_mask_result = benchmark.benchmark_mask_update(allocation)
    
    print("\n[4/4] Mini-Finetuning (50 epochs)...")
    sal_finetune_result = benchmark.benchmark_mini_finetune(epochs=50)
    
    sal_total = (saliency_result['total'] + sal_selection_result['total'] + 
                sal_mask_result['total'] + sal_finetune_result['total'])
    
    print(f"\n{'='*70}")
    print(f"Saliency-based TOTAL: {sal_total:.4f}s")
    print(f"{'='*70}")
    print(f"  Saliency computation: {saliency_result['total']:8.4f}s  ({100*saliency_result['total']/sal_total:5.1f}%)")
    print(f"  Weight selection:     {sal_selection_result['total']:8.4f}s  ({100*sal_selection_result['total']/sal_total:5.1f}%)")
    print(f"  Mask update:          {sal_mask_result['total']:8.4f}s  ({100*sal_mask_result['total']/sal_total:5.1f}%)")
    print(f"  Mini-finetuning:      {sal_finetune_result['total']:8.4f}s  ({100*sal_finetune_result['total']/sal_total:5.1f}%)")
    
    # ============================================================
    # Comparison Summary
    # ============================================================
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\nTotal Time:")
    print(f"  Reference-based: {ref_total:8.4f}s")
    print(f"  Saliency-based:  {sal_total:8.4f}s")
    print(f"  Difference:      {abs(ref_total - sal_total):8.4f}s")
    
    speedup = ref_total / sal_total if sal_total > 0 else 0
    if speedup > 1:
        print(f"\n  ✓ Saliency-based is {speedup:.2f}x FASTER")
    else:
        print(f"\n  ✗ Reference-based is {1/speedup:.2f}x FASTER")
    
    print(f"\nComponent-wise Comparison:")
    print(f"  {'Component':<25} {'Reference':>12} {'Saliency':>12} {'Difference':>12}")
    print(f"  {'-'*65}")
    
    print(f"  {'Preprocessing':<25} {ssim_result['total']:>11.4f}s {saliency_result['total']:>11.4f}s", end="")
    diff = saliency_result['total'] - ssim_result['total']
    if diff > 0:
        print(f" {'+' + f'{diff:.4f}s':>11} (slower)")
    else:
        print(f" {f'{diff:.4f}s':>12} (faster)")
    
    print(f"  {'Weight Selection':<25} {selection_result['total']:>11.4f}s {sal_selection_result['total']:>11.4f}s", end="")
    diff = sal_selection_result['total'] - selection_result['total']
    if diff > 0:
        print(f" {'+' + f'{diff:.4f}s':>11} (slower)")
    else:
        print(f" {f'{diff:.4f}s':>12} (faster)")
    
    print(f"  {'Mask Update':<25} {mask_result['total']:>11.4f}s {sal_mask_result['total']:>11.4f}s", end="")
    diff = sal_mask_result['total'] - mask_result['total']
    if diff > 0:
        print(f" {'+' + f'{diff:.4f}s':>11}")
    else:
        print(f" {f'{diff:.4f}s':>12}")
    
    print(f"  {'Mini-Finetuning':<25} {finetune_result['total']:>11.4f}s {sal_finetune_result['total']:>11.4f}s", end="")
    diff = sal_finetune_result['total'] - finetune_result['total']
    if diff > 0:
        print(f" {'+' + f'{diff:.4f}s':>11}")
    else:
        print(f" {f'{diff:.4f}s':>12}")
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    
    # Preprocessing comparison
    ssim_time = ssim_result['total']
    saliency_time = saliency_result['total']
    
    print(f"\n1. Preprocessing (SSIM vs Saliency):")
    if saliency_time > ssim_time:
        overhead = saliency_time - ssim_time
        print(f"   - Saliency is {saliency_time/ssim_time:.2f}x slower than SSIM")
        print(f"   - Overhead: {overhead:.4f}s per episode")
        print(f"   - Reason: Gradient computation requires forward + backward passes")
        print(f"   - SSIM only needs forward passes for feature extraction")
    else:
        print(f"   - SSIM is {ssim_time/saliency_time:.2f}x slower than saliency")
    
    # Selection comparison
    print(f"\n2. Weight Selection:")
    if sal_selection_result['total'] < selection_result['total']:
        print(f"   - Saliency-based selection is {selection_result['total']/sal_selection_result['total']:.2f}x faster")
        print(f"   - Reason: Direct ranking without loading reference masks")
    else:
        print(f"   - Reference-based selection is {sal_selection_result['total']/selection_result['total']:.2f}x faster")
    
    # Finetuning dominance
    finetune_pct_ref = 100 * finetune_result['total'] / ref_total
    finetune_pct_sal = 100 * sal_finetune_result['total'] / sal_total
    
    print(f"\n3. Mini-Finetuning Dominance:")
    print(f"   - Takes {finetune_pct_ref:.1f}% of reference-based total time")
    print(f"   - Takes {finetune_pct_sal:.1f}% of saliency-based total time")
    print(f"   - This is the bottleneck for both methods")
    print(f"   - Optimization opportunities: Reduce epochs, use faster optimizer, etc.")
    
    # Overall recommendation
    print(f"\n4. Overall Recommendation:")
    if speedup > 1:
        print(f"   ✓ Use SALIENCY-BASED method ({speedup:.2f}x faster per episode)")
        print(f"   - For {500} RL episodes, save ~{(ref_total - sal_total) * 500 / 3600:.1f} hours")
    else:
        print(f"   ✓ Use REFERENCE-BASED method ({1/speedup:.2f}x faster per episode)")
        print(f"   - For {500} RL episodes, save ~{(sal_total - ref_total) * 500 / 3600:.1f} hours")
    
    print("\n" + "="*80)
    
    return {
        'reference_total': ref_total,
        'saliency_total': sal_total,
        'speedup': speedup
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick benchmark for single episode')
    parser.add_argument('--m_name', type=str, default='resnet20',
                       choices=['resnet20', 'vgg16', 'alexnet'])
    
    args = parser.parse_args()
    
    results = quick_benchmark(args.m_name)
