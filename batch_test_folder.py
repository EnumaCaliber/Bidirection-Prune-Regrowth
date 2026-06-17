"""
批量测试文件夹下所有 .pth 模型的稀疏度与准确率，输出为 Excel
用法:
  python batch_test_folder.py --folder ./effnet/ckpt_after_prune_0.3_epoch_finetune_40 --m_name effnet
  python batch_test_folder.py --folder ./googlenet --m_name googlenet --recursive --n_runs 5
"""
import torch
import torch.nn as nn
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from utils.model_loader import model_loader
from utils.data_loader import data_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',    type=str, default='./vgg16/ckpt_after_prune_1', help='要扫描的文件夹路径')
    parser.add_argument('--m_name',   type=str, default='vgg16',  help='模型名称')
    parser.add_argument('--data_dir', type=str, default='./data',  help='数据集路径')
    parser.add_argument('--output',   type=str, default='',        help='输出路径（默认: <folder>/results）')
    parser.add_argument('--recursive', action='store_true',        help='是否递归搜索子文件夹')
    parser.add_argument('--pattern',  type=str, default='*.pth',   help='文件匹配模式')
    parser.add_argument('--seeds', type=int, nargs='+', default=1, help='随机种子列表')
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_sparsity(sd):
    total = sum(v.numel() for k, v in sd.items() if k.endswith('_mask'))
    zeros = sum((v == 0).sum().item() for k, v in sd.items() if k.endswith('_mask'))
    return zeros / total * 100 if total > 0 else 0.0


def load_model(ckpt_path, model_name, device):
    sd = torch.load(ckpt_path, map_location=device)

    merged, done = {}, set()
    for k in sd:
        if k in done:
            continue
        if k.endswith('_orig'):
            base = k[:-5]
            merged[base] = sd[k] * sd[base + '_mask']
            done.update([k, base + '_mask'])
        elif not k.endswith('_mask'):
            merged[k] = sd[k]

    model = model_loader(model_name, device)
    model.load_state_dict(merged)
    return model, sd


criterion = nn.CrossEntropyLoss()


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}  |  seeds={args.seeds}")

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder}")

    glob_fn = folder.rglob if args.recursive else folder.glob
    ckpt_files = sorted(glob_fn(args.pattern))
    if not ckpt_files:
        print(f"未找到匹配 '{args.pattern}' 的文件，退出。")
        return

    print(f"找到 {len(ckpt_files)} 个文件")

    records = []
    for i, ckpt_path in enumerate(ckpt_files, 1):
        print(f"\n[{i}/{len(ckpt_files)}] {ckpt_path.name}")
        try:
            # 只加载一次模型，sparsity 与 seed 无关
            model, sd = load_model(ckpt_path, args.m_name, device)
            sparsity = get_sparsity(sd)

            accs = []
            for seed in args.seeds:
                set_seed(seed)
                _, _, test_loader = data_loader(data_dir=args.data_dir)
                acc = evaluate(model, test_loader, device)
                accs.append(acc)
                print(f"  seed={seed}  acc={acc:.2f}%")

            acc_mean = float(np.mean(accs))
            acc_std  = float(np.std(accs, ddof=1))
            print(f"  => Sparsity: {sparsity:.2f}%  Acc: {acc_mean:.2f} ± {acc_std:.2f}%")

            records.append({
                'Sparsity(%)': round(sparsity, 2),
                'Accuracy(%)': round(acc_mean, 4),
                'Acc_std':     round(acc_std,  4),
            })
        except Exception as e:
            print(f"  失败: {e}")
            records.append({
                'Sparsity(%)': None,
                'Accuracy(%)': None,
                'Acc_std':     None,
            })

    df = pd.DataFrame(records)
    df.sort_values('Sparsity(%)', inplace=True, ignore_index=True)

    base_path = args.output or str(folder / 'results')
    base_path = base_path.removesuffix('.xlsx').removesuffix('.csv')

    csv_path = base_path + '.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nCSV 已保存至: {csv_path}")

    try:
        xlsx_path = base_path + '.xlsx'
        df.to_excel(xlsx_path, index=False)
        print(f"Excel 已保存至: {xlsx_path}")
    except Exception as e:
        print(f"Excel 保存失败（{e}），请用 CSV 文件）")

    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
