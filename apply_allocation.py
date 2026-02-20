"""
apply_best_allocation.py
加载已保存的 best allocation 并应用到模型上
"""

import torch
import argparse
import os
from utils.model_loader import model_loader
from utils.analysis_utils import prune_weights_reparam, load_model_name
from rl_saliency_regrowth import SaliencyBasedRegrowth, SaliencyComputer  # 从主文件导入


def apply_saved_allocation(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 加载 best allocation 文件
    print(f"\nLoading best allocation from: {args.allocation_path}")
    best_data = torch.load(args.allocation_path, map_location=device,weights_only=False)

    allocation = best_data['allocation']
    best_reward = best_data['reward']
    epoch = best_data['epoch']

    print(f"  Saved at epoch: {epoch}")
    print(f"  Best reward: {best_reward:.4f} ({best_reward * 100:.2f}%)")
    print(f"  Allocation:")
    for layer_name, count in allocation.items():
        print(f"    {layer_name}: {count} weights")

    # 加载 pruned model (model_99)
    print(f"\nLoading pruned model...")
    model_99 = model_loader(args.m_name, device)
    prune_weights_reparam(model_99)
    checkpoint_99 = torch.load(args.pruned_model_path, map_location=device,weights_only=False)
    model_99.load_state_dict(checkpoint_99)

    # 加载 pretrained model（用于计算 saliency）
    print(f"Loading pretrained model for saliency computation...")
    model_pretrained = model_loader(args.m_name, device)

    load_model_name(model_pretrained, f'./{args.m_name}/checkpoint', args.m_name)


    # 加载数据（用于 saliency 计算）
    from utils.data_loader import data_loader
    train_loader, _, test_loader = data_loader(data_dir=args.data_dir)

    # 计算 saliency
    target_layers = list(allocation.keys())
    saliency_computer = SaliencyComputer(
        model=model_pretrained,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device
    )
    saliency_dict = saliency_computer.compute_saliency_scores(
        data_loader=train_loader,
        target_layers=target_layers,
        max_batches=args.saliency_max_batches,
        num_classes= 10
    )

    # 评估 regrowth 前的 accuracy
    def evaluate(model):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return 100.0 * correct / total

    before_acc = evaluate(model_99)
    print(f"\nAccuracy before regrowth: {before_acc:.2f}%")

    # 应用 allocation
    print("\nApplying best allocation...")
    for layer_name, num_weights in allocation.items():
        if num_weights > 0:
            saliency_tensor = saliency_dict.get(layer_name)
            if saliency_tensor is not None:
                actual, _ = SaliencyBasedRegrowth.apply_regrowth(
                    model=model_99,
                    layer_name=layer_name,
                    saliency_tensor=saliency_tensor,
                    num_weights=num_weights,
                    init_strategy=args.init_strategy,
                    device=device
                )
                print(f"  {layer_name}: regrown {actual} weights")
            else:
                print(f"  {layer_name}: saliency not found, skipping")

    after_acc = evaluate(model_99)
    print(f"\nAccuracy after regrowth: {after_acc:.2f}%")
    print(f"Improvement: {after_acc - before_acc:+.2f}%")

    # 保存结果模型
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(model_99.state_dict(), args.save_path)
        print(f"\nModel saved to: {args.save_path}")

    return model_99


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_name', type=str, default='vgg16')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--allocation_path', type=str,
                        default='./rl_saliency_checkpoints/vgg16/oneshot/0.96/best_saliency_allocation.pth')
    parser.add_argument('--pruned_model_path', type=str,
                        default='./vgg16/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.995.pth')
    parser.add_argument('--pretrained_model_path', type=str,
                        default='./vgg16/checkpoint')
    parser.add_argument('--init_strategy', type=str, default='zero',
                        choices=['zero', 'kaiming', 'xavier', 'magnitude'])
    parser.add_argument('--saliency_max_batches', type=int, default=50)
    parser.add_argument('--save_path', type=str,
                        default='./rl_saliency_checkpoints/vgg16/oneshot/0.96/regrown_model.pth')
    args = parser.parse_args()

    apply_saved_allocation(args)