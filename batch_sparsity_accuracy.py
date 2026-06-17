#!/usr/bin/env python
"""
批量评估模型文件夹下所有 checkpoint 的稀疏度和准确率，并写入 Excel。
"""
import argparse
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from utils.model_loader import model_loader
from utils.data_loader import data_loader


def parse_args():
    parser = argparse.ArgumentParser(description='批量计算模型稀疏度和准确率，并输出 Excel')
    parser.add_argument('--model_dir', type=str, default='./vgg16/ckpt_after_prune_0.3_epoch_finetune_40',
                        help='模型 checkpoint 所在文件夹')
    parser.add_argument('--m_name', type=str, default='vgg16',
                        help='模型类型名称，例如 effnet/vgg16/resnet20 等')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录，默认 ./data')
    parser.add_argument('--output', type=str, default='batch_eval.csv',
                        help='输出文件名，默认 batch_eval.csv')
    parser.add_argument('--device', type=str, default=None,
                        help='运行设备，例如 cpu 或 cuda，默认自动选择')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='测试集批量大小，默认 100')
    parser.add_argument('--recursive', action='store_true',
                        help='是否递归扫描子目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，默认 42')
    parser.add_argument('--skip-errors', action='store_true',
                        help='遇到单个模型加载/评估失败时继续处理后续文件')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_checkpoint_paths(model_dir, recursive=False):
    p = Path(model_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f'模型目录不存在: {model_dir}')
    pattern = '**/*.pth' if recursive else '*.pth'
    paths = list(p.glob(pattern))
    paths += list(p.glob('*.pt'))
    if recursive:
        paths += list(p.rglob('*.pt'))
    return sorted(set(paths))


def load_checkpoint_state_dict(ckpt_path, device):
    data = torch.load(ckpt_path, map_location=device)
    if isinstance(data, dict):
        if 'state_dict' in data:
            return data['state_dict']
        if 'model' in data:
            return data['model']
    return data


def get_sparsity(sd):
    masks = [v for k, v in sd.items() if k.endswith('_mask')]
    total = sum(v.numel() for v in masks)
    zeros = sum((v == 0).sum().item() for v in masks)
    return zeros / total * 100 if total > 0 else 0.0


def merge_weight_and_mask(sd):
    merged = {}
    done = set()
    for key, value in sd.items():
        if key in done:
            continue
        if key.endswith('_orig'):
            base = key[:-5]
            mask_key = base + '_mask'
            mask_value = sd.get(mask_key)
            if mask_value is not None:
                merged[base] = value * mask_value
                done.update({key, mask_key})
            else:
                merged[key] = value
        elif key.endswith('_mask'):
            continue
        else:
            merged[key] = value
    return merged


def load_model(ckpt_path, model_name, device):
    sd = load_checkpoint_state_dict(ckpt_path, device)
    if not isinstance(sd, dict):
        raise ValueError(f'不支持的 checkpoint 格式: {ckpt_path}')
    merged = merge_weight_and_mask(sd)
    model = model_loader(model_name, device)
    try:
        model.load_state_dict(merged)
    except RuntimeError as exc:
        model.load_state_dict(merged, strict=False)
        print(f'警告: {ckpt_path.name} 使用 strict=False 加载，可能存在参数不匹配: {exc}')
    return model, sd


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_sum += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    acc = 100.0 * correct / total if total > 0 else 0.0
    loss = loss_sum / len(loader) if len(loader) > 0 else 0.0
    return acc, loss


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    _, _, test_loader = data_loader(args.data_dir)
    ckpt_paths = find_checkpoint_paths(args.model_dir, recursive=args.recursive)
    if not ckpt_paths:
        raise SystemExit(f'未找到 checkpoint 文件: {args.model_dir}')

    rows = []
    for ckpt_path in ckpt_paths:
        try:
            model, sd = load_model(ckpt_path, args.m_name, device)
            sparsity = get_sparsity(sd)
            acc, loss = evaluate(model, test_loader, device)
            rows.append({
                'file_name': ckpt_path.name,
                'file_path': str(ckpt_path),
                'sparsity_%': round(sparsity, 2),
                'accuracy_%': acc,
                'loss': loss,
            })
            print(f'已评估: {ckpt_path.name} | sparsity={sparsity:.2f}% | acc={acc:.2f}%')
        except Exception as exc:
            msg = f'错误: {ckpt_path.name} -> {exc}'
            if args.skip_errors:
                print(msg)
                continue
            raise

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f'已生成 CSV: {args.output}')


if __name__ == '__main__':
    main()
