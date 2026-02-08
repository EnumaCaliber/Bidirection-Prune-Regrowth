import torch, math, os
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from utils.model_loader import model_loader

MODELS = {
    "vgg16": {
        "pretrained": "vgg16/checkpoint/pretrain_vgg16_ckpt.pth",
        "pruned-oneshot": "vgg16/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.995.pth",
        "pruned-iterative": "vgg16/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.9953.pth",
    },
    "resnet20": {
        "pretrained": "resnet20/checkpoint/pretrain_resnet20_ckpt.pth",
        "pruned-oneshot": "resnet20/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.995.pth",
        "pruned-iterative": "resnet20/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.9862.pth",
    },
    "densenet": {
        "pretrained": "densenet/checkpoint/pretrain_densenet_ckpt.pth",
        "pruned-oneshot": "densenet/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.995.pth",
        "pruned-iterative": "densenet/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.9953.pth",
    },
    "effnet": {
        "pretrained": "effnet/checkpoint/pretrain_effnet_ckpt.pth",
        "pruned-oneshot": "effnet/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.995.pth",
        "pruned-iterative": "effnet/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.9953.pth",
    },
}
IMG_IDX = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
OUT_BASE = "feature_comparison"
PRUNE_THRESHOLD = 1e-3  # 判断通道是否被剪枝的阈值


def load_pruned(ckpt_path, device, model_name):
    sd = torch.load(ckpt_path, map_location=device)
    merged, done = {}, set()
    for k in sd:
        if k in done:
            continue
        if k.endswith("_orig"):
            base = k[:-5]
            merged[base] = sd[k] * sd[base + "_mask"]
            done.update([k, base + "_mask"])
        elif not k.endswith("_mask"):
            merged[k] = sd[k]
    model = model_loader(model_name, device)
    model.load_state_dict(merged)
    return model.eval()


def load_pretrained(ckpt_path, device, model_name):
    model = model_loader(model_name, device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    return model.eval()


def make_hook(name, storage, order):
    def fn(module, inp, out):
        storage[name] = out.detach()
        if name not in order:
            order.append(name)

    return fn


def register_hooks(model, storage, order):
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(make_hook(name, storage, order)))
    return hooks


def extract_features(model, img, device):
    """提取模型的特征"""
    storage, order = {}, []
    hooks = register_hooks(model, storage, order)
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
    for h in hooks:
        h.remove()
    return storage, order, pred


def normalize_to_range(feat, min_val=-1, max_val=1):
    """将特征归一化到指定范围"""
    feat_min = feat.min()
    feat_max = feat.max()

    # 避免除零
    if feat_max - feat_min < 1e-8:
        return torch.zeros_like(feat)

    # 归一化到[0, 1]
    feat_norm = (feat - feat_min) / (feat_max - feat_min)
    # 缩放到[min_val, max_val]
    feat_scaled = feat_norm * (max_val - min_val) + min_val

    return feat_scaled


def is_channel_pruned(feat_channel, threshold=PRUNE_THRESHOLD):
    """判断通道是否被剪枝（激活值接近0）"""
    return feat_channel.abs().max().item() < threshold


def show_comparison_feat(feat_dict, name, save_dir, max_ch=16):
    """
    对比显示三个模型的特征图以及差异
    feat_dict: {'pretrained': tensor, 'pruned-iterative': tensor, 'pruned-oneshot': tensor}
    """
    # 检查是否所有特征都存在
    if not all(k in feat_dict for k in ['pretrained', 'pruned-iterative', 'pruned-oneshot']):
        return

    feat_pre = feat_dict['pretrained'][0]
    feat_iter = feat_dict['pruned-iterative'][0]
    feat_one = feat_dict['pruned-oneshot'][0]

    # 只处理3D特征图 (C, H, W)
    if feat_pre.dim() != 3:
        return

    C, H, W = feat_pre.shape
    n = min(C, max_ch)

    # 检测剪枝情况
    pruned_iter = [is_channel_pruned(feat_iter[i]) for i in range(C)]
    pruned_one = [is_channel_pruned(feat_one[i]) for i in range(C)]
    pruned_iter_count = sum(pruned_iter)
    pruned_one_count = sum(pruned_one)

    # 创建5列的图：pretrained, iterative, oneshot, pre-iter差异, pre-oneshot差异
    fig, axes = plt.subplots(n, 5, figsize=(18, n * 2.5))
    main_title = f"{name}  |  ch={C}, {H}x{W}\n"
    main_title += f"Pruned channels - Iterative: {pruned_iter_count}/{C} ({pruned_iter_count / C * 100:.1f}%), "
    main_title += f"Oneshot: {pruned_one_count}/{C} ({pruned_one_count / C * 100:.1f}%)"
    fig.suptitle(main_title, fontsize=14, fontweight='bold')

    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        # 归一化每个通道到[-1, 1]
        feat_pre_norm = normalize_to_range(feat_pre[i])
        feat_iter_norm = normalize_to_range(feat_iter[i])
        feat_one_norm = normalize_to_range(feat_one[i])

        # Pretrained
        axes[i, 0].imshow(feat_pre_norm.cpu().numpy(), cmap='viridis', vmin=-1, vmax=1)
        title_pre = f"Pretrained ch{i}"
        axes[i, 0].set_title(title_pre, fontsize=10)
        axes[i, 0].axis('off')

        # Iterative - 标注是否被剪枝
        axes[i, 1].imshow(feat_iter_norm.cpu().numpy(), cmap='viridis', vmin=-1, vmax=1)
        title_iter = f"Iterative ch{i}"
        if pruned_iter[i]:
            title_iter += " ❌ PRUNED"
            axes[i, 1].set_title(title_iter, fontsize=10, color='red', fontweight='bold')
        else:
            axes[i, 1].set_title(title_iter, fontsize=10)
        axes[i, 1].axis('off')

        # Oneshot - 标注是否被剪枝
        axes[i, 2].imshow(feat_one_norm.cpu().numpy(), cmap='viridis', vmin=-1, vmax=1)
        title_one = f"Oneshot ch{i}"
        if pruned_one[i]:
            title_one += " ❌ PRUNED"
            axes[i, 2].set_title(title_one, fontsize=10, color='red', fontweight='bold')
        else:
            axes[i, 2].set_title(title_one, fontsize=10)
        axes[i, 2].axis('off')

        # Pretrained - Iterative 差异（归一化后再相减）
        diff_iter = (feat_pre_norm - feat_iter_norm).cpu().numpy()
        axes[i, 3].imshow(diff_iter, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i, 3].set_title(f"Pre - Iter ch{i}", fontsize=10)
        axes[i, 3].axis('off')

        # Pretrained - Oneshot 差异（归一化后再相减）
        diff_one = (feat_pre_norm - feat_one_norm).cpu().numpy()
        axes[i, 4].imshow(diff_one, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i, 4].set_title(f"Pre - One ch{i}", fontsize=10)
        axes[i, 4].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(save_dir, f"{name.replace('.', '_')}_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def show_pruning_overview(feat_dict, name, save_dir):
    """显示整体剪枝情况的概览"""
    if not all(k in feat_dict for k in ['pretrained', 'pruned-iterative', 'pruned-oneshot']):
        return

    feat_pre = feat_dict['pretrained'][0]
    feat_iter = feat_dict['pruned-iterative'][0]
    feat_one = feat_dict['pruned-oneshot'][0]

    if feat_pre.dim() != 3:
        return

    C, H, W = feat_pre.shape

    # 计算每个通道的L1 norm（未归一化的原始激活）
    l1_pre = torch.tensor([feat_pre[i].abs().mean().item() for i in range(C)])
    l1_iter = torch.tensor([feat_iter[i].abs().mean().item() for i in range(C)])
    l1_one = torch.tensor([feat_one[i].abs().mean().item() for i in range(C)])

    # 检测剪枝
    pruned_iter = [is_channel_pruned(feat_iter[i]) for i in range(C)]
    pruned_one = [is_channel_pruned(feat_one[i]) for i in range(C)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{name} - Channel Activation Analysis", fontsize=14, fontweight='bold')

    # 1. 每个通道的平均激活强度
    x = np.arange(C)
    axes[0, 0].plot(x, l1_pre.numpy(), 'o-', label='Pretrained', linewidth=2, markersize=4)
    axes[0, 0].plot(x, l1_iter.numpy(), 's-', label='Iterative', linewidth=2, markersize=4, alpha=0.7)
    axes[0, 0].plot(x, l1_one.numpy(), '^-', label='Oneshot', linewidth=2, markersize=4, alpha=0.7)
    axes[0, 0].axhline(y=PRUNE_THRESHOLD, color='r', linestyle='--', label=f'Prune threshold ({PRUNE_THRESHOLD})')
    axes[0, 0].set_xlabel('Channel Index')
    axes[0, 0].set_ylabel('Mean Absolute Activation')
    axes[0, 0].set_title('Channel-wise Activation Strength')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # 2. 剪枝通道的分布
    axes[0, 1].bar(x, [1 if not p else 0 for p in pruned_iter], alpha=0.5, label='Iterative Active', color='orange')
    axes[0, 1].bar(x, [-1 if p else 0 for p in pruned_iter], alpha=0.5, label='Iterative Pruned', color='red')
    axes[0, 1].set_xlabel('Channel Index')
    axes[0, 1].set_ylabel('Active (1) / Pruned (-1)')
    axes[0, 1].set_title(f'Iterative: {sum(pruned_iter)}/{C} channels pruned ({sum(pruned_iter) / C * 100:.1f}%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Oneshot剪枝分布
    axes[1, 0].bar(x, [1 if not p else 0 for p in pruned_one], alpha=0.5, label='Oneshot Active', color='green')
    axes[1, 0].bar(x, [-1 if p else 0 for p in pruned_one], alpha=0.5, label='Oneshot Pruned', color='red')
    axes[1, 0].set_xlabel('Channel Index')
    axes[1, 0].set_ylabel('Active (1) / Pruned (-1)')
    axes[1, 0].set_title(f'Oneshot: {sum(pruned_one)}/{C} channels pruned ({sum(pruned_one) / C * 100:.1f}%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 激活保留率
    retention_iter = (l1_iter / (l1_pre + 1e-8)).numpy()
    retention_one = (l1_one / (l1_pre + 1e-8)).numpy()
    axes[1, 1].plot(x, retention_iter, 's-', label='Iterative', linewidth=2, markersize=4)
    axes[1, 1].plot(x, retention_one, '^-', label='Oneshot', linewidth=2, markersize=4)
    axes[1, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='100% retention')
    axes[1, 1].axhline(y=0.0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Channel Index')
    axes[1, 1].set_ylabel('Activation Retention Ratio')
    axes[1, 1].set_title('Activation Retention (Pruned / Pretrained)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([-0.1, 1.5])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, f"{name.replace('.', '_')}_pruning_overview.png"), dpi=150, bbox_inches='tight')
    plt.close()


def show_comparison_linear(feat_dict, name, save_dir):
    """对比显示线性层输出（1D）"""
    if not all(k in feat_dict for k in ['pretrained', 'pruned-iterative', 'pruned-oneshot']):
        return

    feat_pre = feat_dict['pretrained'][0]
    feat_iter = feat_dict['pruned-iterative'][0]
    feat_one = feat_dict['pruned-oneshot'][0]

    if feat_pre.dim() != 1:
        return

    # 归一化到[-1, 1]
    feat_pre_norm = normalize_to_range(feat_pre)
    feat_iter_norm = normalize_to_range(feat_iter)
    feat_one_norm = normalize_to_range(feat_one)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"{name}  |  dim={len(feat_pre)}", fontsize=14, fontweight='bold')

    # 第一行：三个模型的输出
    axes[0, 0].bar(range(len(feat_pre_norm)), feat_pre_norm.cpu().numpy(), color='steelblue')
    axes[0, 0].set_title('Pretrained (normalized)')
    axes[0, 0].set_xticks(range(len(NAMES)))
    axes[0, 0].set_xticklabels(NAMES, rotation=45)
    axes[0, 0].set_ylim([-1.1, 1.1])
    axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    axes[0, 1].bar(range(len(feat_iter_norm)), feat_iter_norm.cpu().numpy(), color='orange')
    axes[0, 1].set_title('Pruned-Iterative (normalized)')
    axes[0, 1].set_xticks(range(len(NAMES)))
    axes[0, 1].set_xticklabels(NAMES, rotation=45)
    axes[0, 1].set_ylim([-1.1, 1.1])
    axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    axes[0, 2].bar(range(len(feat_one_norm)), feat_one_norm.cpu().numpy(), color='green')
    axes[0, 2].set_title('Pruned-Oneshot (normalized)')
    axes[0, 2].set_xticks(range(len(NAMES)))
    axes[0, 2].set_xticklabels(NAMES, rotation=45)
    axes[0, 2].set_ylim([-1.1, 1.1])
    axes[0, 2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # 第二行：差异
    diff_iter = (feat_pre_norm - feat_iter_norm).cpu().numpy()
    axes[1, 0].bar(range(len(diff_iter)), diff_iter, color='purple')
    axes[1, 0].set_title('Pretrained - Iterative')
    axes[1, 0].set_xticks(range(len(NAMES)))
    axes[1, 0].set_xticklabels(NAMES, rotation=45)
    axes[1, 0].set_ylim([-1.1, 1.1])
    axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    diff_one = (feat_pre_norm - feat_one_norm).cpu().numpy()
    axes[1, 1].bar(range(len(diff_one)), diff_one, color='red')
    axes[1, 1].set_title('Pretrained - Oneshot')
    axes[1, 1].set_xticks(range(len(NAMES)))
    axes[1, 1].set_xticklabels(NAMES, rotation=45)
    axes[1, 1].set_ylim([-1.1, 1.1])
    axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # 统计信息
    axes[1, 2].axis('off')
    stats_text = f"Iterative差异:\n  Mean: {diff_iter.mean():.4f}\n  Std: {diff_iter.std():.4f}\n  Max: {diff_iter.max():.4f}\n  Min: {diff_iter.min():.4f}\n\n"
    stats_text += f"Oneshot差异:\n  Mean: {diff_one.mean():.4f}\n  Std: {diff_one.std():.4f}\n  Max: {diff_one.max():.4f}\n  Min: {diff_one.min():.4f}"
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', family='monospace')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, f"{name.replace('.', '_')}_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def run_comparison(model_name, ckpts, img, device):
    """对比运行三个版本的模型"""
    save_dir = os.path.join(OUT_BASE, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # 加载三个模型
    models = {}
    models['pretrained'] = load_pretrained(ckpts['pretrained'], device, model_name)
    models['pruned-iterative'] = load_pruned(ckpts['pruned-iterative'], device, model_name)
    models['pruned-oneshot'] = load_pruned(ckpts['pruned-oneshot'], device, model_name)

    # 提取特征
    features = {}
    predictions = {}
    orders = {}

    for tag, model in models.items():
        storage, order, pred = extract_features(model, img, device)
        features[tag] = storage
        orders[tag] = order
        predictions[tag] = pred
        print(f"[{model_name}/{tag}] 预测: {NAMES[pred]}")

    # 使用pretrained的layer顺序作为基准
    base_order = orders['pretrained']

    # 对每个layer进行对比可视化
    for layer_name in base_order:
        # 收集该层的三个版本特征
        feat_dict = {}
        for tag in ['pretrained', 'pruned-iterative', 'pruned-oneshot']:
            if layer_name in features[tag]:
                feat_dict[tag] = features[tag][layer_name]

        # 根据特征维度选择可视化方式
        if len(feat_dict) == 3:
            sample_feat = feat_dict['pretrained'][0]
            if sample_feat.dim() == 3:
                show_comparison_feat(feat_dict, layer_name, save_dir)
                show_pruning_overview(feat_dict, layer_name, save_dir)  # 添加剪枝概览
            elif sample_feat.dim() == 1:
                show_comparison_linear(feat_dict, layer_name, save_dir)

    print(f"[{model_name}] 对比图已保存到 {save_dir}/")


if __name__ == "__main__":
    # 加载数据
    dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ]))
    img, label = dataset[IMG_IDX]
    print(f"[INFO] 第 {IMG_IDX} 张，真实标签: {NAMES[label]}")

    # 保存原始图像
    os.makedirs(OUT_BASE, exist_ok=True)
    raw = np.clip(img.permute(1, 2, 0).numpy() * [.247, .2435, .2616] + [.4914, .4822, .4465], 0, 1)
    plt.figure(figsize=(3, 3))
    plt.imshow(raw)
    plt.title(f"label: {NAMES[label]}")
    plt.axis('off')
    plt.savefig(os.path.join(OUT_BASE, "original.png"), dpi=150)
    plt.close()

    # 对每个模型进行对比分析
    for model_name, ckpts in MODELS.items():
        print(f"\n{'=' * 50}")
        print(f"处理模型: {model_name}")
        print(f"{'=' * 50}")
        run_comparison(model_name, ckpts, img, DEVICE)