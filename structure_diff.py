

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils.model_loader import model_loader

# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "vgg16": {
        "pretrained":  "vgg16/checkpoint/pretrain_vgg16_ckpt.pth",
        "pruned-low": "vgg16/ckpt_structured_iterative/step19_sp0.754.pth",
        "pruned-high":  "vgg16/ckpt_structured_iterative/step23_sp0.913.pth",
    },
}

TAGS        = ["pretrained", "pruned-high", "pruned-low"]
COL_LABELS  = ["Baseline (Dense)", "Structured High Sparsity", "Structured Low Sparsity"]
COL_COLORS  = ["steelblue", "tomato", "darkorange"]

# 颜色定义
POSITIVE_COLOR = np.array([0.0,   0.0,   0.0],   dtype=np.float32)  # 黑 正值
NEGATIVE_COLOR = np.array([1.0,   1.0,   1.0],   dtype=np.float32)  # 白 负值
ZERO_COLOR     = np.array([0.5,   0.5,   0.5],   dtype=np.float32)  # 灰 零值
PRUNED_COLOR   = np.array([0.18,  0.45,  0.72],  dtype=np.float32)  # 蓝 已删除通道
KEPT_SAME_COLOR= np.array([0.61,  0.72,  0.61],  dtype=np.float32)  # 绿 保留且同号
KEPT_FLIP_COLOR= np.array([0.97,  0.92,  0.78],  dtype=np.float32)  # 米黄 保留且异号

MAX_CH    = 64    # 每层最多展示通道数
IMG_IDX   = 0
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
NAMES     = ['airplane', 'automobile', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck']
OUT_BASE  = "structured_comparison"


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────

def load_dense(ckpt_path: str, model_name: str, device: str) -> nn.Module:
    """加载 dense baseline（state_dict 格式）"""
    model = model_loader(model_name, device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['net'])
    return model.eval()


def load_structured(ckpt_path: str, device: str) -> nn.Module:
    """
    加载结构化剪枝模型。
    支持两种保存方式：
      1. torch.save(model, path)                 → 直接是 nn.Module
      2. torch.save({'net': state_dict}, path)   → 需要配合模型定义（暂不支持）
    """
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval()
    raise ValueError(
        f"{ckpt_path} 不是 nn.Module 对象，请确认用 torch.save(model, path) 保存。"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 通道索引映射（结构化剪枝核心）
# ─────────────────────────────────────────────────────────────────────────────

def get_channel_indices(dense_model: nn.Module,
                        pruned_model: nn.Module,
                        layer_name: str) -> tuple[int, list[int] | None]:
    """
    尝试推断剪枝模型在 layer_name 层保留了 dense 的哪些通道。

    策略：通过比较权重余弦相似度做最优匹配（贪心）。
    如果通道数相同，直接认为一一对应。

    返回：
        dense_out_ch  : dense 模型的输出通道数
        kept_indices  : 保留通道在 dense 中的索引列表（None 表示无法推断）
    """
    def get_conv(model):
        for n, m in model.named_modules():
            if n == layer_name and isinstance(m, nn.Conv2d):
                return m
        return None

    d_conv = get_conv(dense_model)
    p_conv = get_conv(pruned_model)

    if d_conv is None or p_conv is None:
        return 0, None

    d_out = d_conv.out_channels
    p_out = p_conv.out_channels

    if d_out == p_out:
        # 未剪枝该层，直接映射
        return d_out, list(range(d_out))

    # 贪心余弦相似度匹配
    d_w = d_conv.weight.detach().reshape(d_out, -1).float()  # (D, K)
    p_w = p_conv.weight.detach().reshape(p_out, -1).float()  # (P, K)

    # 截断到相同 in_channels * kH * kW（结构化剪枝可能级联改变 in_ch）
    k = min(d_w.shape[1], p_w.shape[1])
    d_w = d_w[:, :k]
    p_w = p_w[:, :k]

    d_norm = d_w / (d_w.norm(dim=1, keepdim=True) + 1e-8)
    p_norm = p_w / (p_w.norm(dim=1, keepdim=True) + 1e-8)

    sim = (p_norm @ d_norm.T).cpu().numpy()   # (P, D)

    used = set()
    kept = []
    for pi in range(p_out):
        order = np.argsort(-sim[pi])
        for di in order:
            if di not in used:
                used.add(di)
                kept.append(int(di))
                break

    return d_out, kept


# ─────────────────────────────────────────────────────────────────────────────
# Hook 工具
# ─────────────────────────────────────────────────────────────────────────────

def register_hooks(model, storage, order):
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            def fn(mod, inp, out, _n=name):
                storage[_n] = out.detach()
                if _n not in order:
                    order.append(_n)
            hooks.append(m.register_forward_hook(fn))
    return hooks


def extract_features(model, img, device):
    storage, order = {}, []
    hooks = register_hooks(model, storage, order)
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
    for h in hooks:
        h.remove()
    return storage, order, pred


# ─────────────────────────────────────────────────────────────────────────────
# 绘图辅助
# ─────────────────────────────────────────────────────────────────────────────

def to_sign_rgb(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.cpu().float().numpy()
    out = np.full((*arr.shape, 3), 0.5, dtype=np.float32)
    out[arr > 0] = POSITIVE_COLOR
    out[arr < 0] = NEGATIVE_COLOR
    return out


def _make_grid(tag, total, n,
               dense_feat, pruned_feat,
               kept_indices, d_out):
    """
    构建 n×n 通道图网格。
    - tag == 'pretrained'：全部通道符号图
    - 其他 tag：
        * 被删通道  → PRUNED_COLOR 蓝色填充
        * 保留通道  → 与 dense 比对，同号绿 / 异号米黄
    """
    h, w = dense_feat[0].shape[-2:]
    pruned_set = set(kept_indices) if kept_indices else set(range(d_out))
    # 建立 dense idx → pruned channel idx 的映射
    kept_map = {di: pi for pi, di in enumerate(kept_indices)} \
               if kept_indices else {i: i for i in range(d_out)}

    rows = []
    for r in range(n):
        row_imgs = []
        for c in range(n):
            ch = r * n + c
            if ch >= total:
                row_imgs.append(np.full((h, w, 3), 0.85, dtype=np.float32))
                continue

            if tag == "pretrained":
                row_imgs.append(to_sign_rgb(dense_feat[ch]))
            else:
                if ch not in pruned_set:
                    # 已删除通道 → 蓝色
                    tile = np.full((h, w, 3), PRUNED_COLOR, dtype=np.float32)
                else:
                    pi   = kept_map[ch]
                    if pi >= pruned_feat.shape[0]:
                        tile = np.full((h, w, 3), PRUNED_COLOR, dtype=np.float32)
                    else:
                        d_ch = dense_feat[ch]
                        p_ch = pruned_feat[pi]
                        flip = ((d_ch * p_ch) < 0).cpu().numpy()
                        tile = np.where(flip[..., None],
                                        KEPT_FLIP_COLOR,
                                        KEPT_SAME_COLOR).astype(np.float32)
                row_imgs.append(tile)
        rows.append(np.concatenate(row_imgs, axis=1))
    return np.concatenate(rows, axis=0)


def _add_legend(ax, tag):
    if tag == "pretrained":
        handles = [
            mpatches.Patch(facecolor=tuple(POSITIVE_COLOR), label='positive',
                           edgecolor='lightgray'),
            mpatches.Patch(facecolor=tuple(NEGATIVE_COLOR), label='negative',
                           edgecolor='lightgray'),
            mpatches.Patch(facecolor=tuple(ZERO_COLOR),     label='zero'),
        ]
    else:
        handles = [
            mpatches.Patch(facecolor=tuple(KEPT_SAME_COLOR), label='kept – same sign as dense'),
            mpatches.Patch(facecolor=tuple(KEPT_FLIP_COLOR), label='kept – sign flipped'),
            mpatches.Patch(facecolor=tuple(PRUNED_COLOR),    label='pruned (channel removed)'),
        ]
    ax.legend(handles=handles, loc='lower right', fontsize=6,
              framealpha=0.8, handlelength=1.2)


# ─────────────────────────────────────────────────────────────────────────────
# Conv 层可视化
# ─────────────────────────────────────────────────────────────────────────────

def save_conv_per_tag(feat_dict, layer_name, dirs,
                      dense_model, pruned_models):
    dense_feat = feat_dict["pretrained"][0]
    if dense_feat.dim() != 3:
        return

    d_out = dense_feat.shape[0]
    total = min(d_out, MAX_CH)
    n     = int(total ** 0.5)
    fname = layer_name.replace(".", "_") + ".pdf"
    h, w  = dense_feat.shape[-2:]

    for tag, label, color in zip(TAGS, COL_LABELS, COL_COLORS):
        if tag not in feat_dict:
            continue

        pruned_feat = feat_dict[tag][0]
        p_out       = pruned_feat.shape[0]

        # 获取通道映射
        if tag == "pretrained":
            kept_indices = list(range(d_out))
        else:
            _, kept_indices = get_channel_indices(
                dense_model, pruned_models[tag], layer_name
            )
            if kept_indices is None:
                kept_indices = list(range(p_out))

        pruned_ratio = 1.0 - p_out / d_out if d_out > 0 else 0.0

        grid = _make_grid(tag, total, n,
                          dense_feat, pruned_feat,
                          kept_indices, d_out)

        fig_w = max(w * n / 50, 4) + 0.3
        fig_h = max(h * n / 50, 4)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(grid, interpolation='nearest', aspect='equal')
        ax.axis('off')

        title = f"{label} — {layer_name}"
        if tag != "pretrained":
            title += (f"\nChannels: {p_out}/{d_out} kept  "
                      f"({pruned_ratio * 100:.1f}% pruned)")
        ax.set_title(title, fontsize=8, color=color)
        _add_legend(ax, tag)

        plt.tight_layout()
        plt.savefig(os.path.join(dirs[tag], fname),
                    bbox_inches="tight", dpi=150)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Linear 层可视化
# ─────────────────────────────────────────────────────────────────────────────

def save_linear_per_tag(feat_dict, layer_name, dirs):
    dense_feat = feat_dict["pretrained"][0]
    if dense_feat.dim() != 1:
        return

    fname  = layer_name.replace(".", "_") + ".pdf"
    d_len  = dense_feat.shape[0]
    g_min  = min(feat_dict[t][0].min().item() for t in TAGS if t in feat_dict)
    g_max  = max(feat_dict[t][0].max().item() for t in TAGS if t in feat_dict)

    for tag, label, color in zip(TAGS, COL_LABELS, COL_COLORS):
        if tag not in feat_dict:
            continue
        feat = feat_dict[tag][0]
        p_len = feat.shape[0]
        norm = ((feat - g_min) / (g_max - g_min + 1e-8)).cpu().numpy()

        if tag == "pretrained":
            def sign_c(v):
                if v > 0: return tuple(POSITIVE_COLOR)
                if v < 0: return tuple(NEGATIVE_COLOR)
                return tuple(ZERO_COLOR)
            bar_colors  = [sign_c(v) for v in feat.cpu().numpy()]
            pruned_ratio = 0.0
        else:
            # Linear 层结构化剪枝：直接按位置对齐（取 min 长度）
            l = min(d_len, p_len)
            flip        = ((dense_feat[:l] * feat[:l]) < 0).cpu().numpy()
            bar_colors  = []
            for i in range(p_len):
                if i >= d_len:
                    bar_colors.append(tuple(KEPT_SAME_COLOR))
                elif flip[i]:
                    bar_colors.append(tuple(KEPT_FLIP_COLOR))
                else:
                    bar_colors.append(tuple(KEPT_SAME_COLOR))
            pruned_ratio = 1.0 - p_len / d_len if d_len > 0 else 0.0

        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.bar(range(len(norm)), norm, color=bar_colors, width=1.0,
               edgecolor='none')
        ax.set_ylim([0, 1.05])
        ax.axhline(y=0, color='k', linewidth=0.4)
        ax.tick_params(labelbottom=False)

        title = f"{label} — {layer_name}"
        if tag != "pretrained":
            title += f"\nNeurons: {p_len}/{d_len}  ({pruned_ratio*100:.1f}% pruned)"
        ax.set_title(title, fontsize=8, color=color)
        _add_legend(ax, tag)

        plt.tight_layout()
        plt.savefig(os.path.join(dirs[tag], fname),
                    bbox_inches="tight", dpi=150)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 总览热图：各层通道保留率
# ─────────────────────────────────────────────────────────────────────────────

def save_retention_heatmap(dense_model, pruned_models, save_dir):
    """横轴: 模型，纵轴: 层名，颜色: 通道保留率"""
    layer_names = [n for n, m in dense_model.named_modules()
                   if isinstance(m, nn.Conv2d)]

    tag_list    = [t for t in TAGS if t != "pretrained"]
    data        = np.zeros((len(layer_names), len(tag_list)))

    for j, tag in enumerate(tag_list):
        pm = pruned_models[tag]
        for i, ln in enumerate(layer_names):
            d_conv = next((m for n, m in dense_model.named_modules()
                           if n == ln and isinstance(m, nn.Conv2d)), None)
            p_conv = next((m for n, m in pm.named_modules()
                           if n == ln and isinstance(m, nn.Conv2d)), None)
            if d_conv and p_conv and d_conv.out_channels > 0:
                data[i, j] = p_conv.out_channels / d_conv.out_channels
            else:
                data[i, j] = 1.0

    fig_h = max(4, len(layer_names) * 0.35)
    fig, ax = plt.subplots(figsize=(3 + len(tag_list) * 1.5, fig_h))

    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(tag_list)))
    ax.set_xticklabels([COL_LABELS[TAGS.index(t)] for t in tag_list],
                       fontsize=8, rotation=15, ha='right')
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=7)

    for i in range(len(layer_names)):
        for j in range(len(tag_list)):
            v = data[i, j]
            txt_color = 'black' if v > 0.5 else 'white'
            ax.text(j, i, f"{v*100:.0f}%", ha='center', va='center',
                    fontsize=6.5, color=txt_color)

    plt.colorbar(im, ax=ax, label='Channel Retention Rate')
    ax.set_title("Structured Pruning — Channel Retention per Layer",
                 fontsize=10, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "channel_retention.pdf"),
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved channel_retention.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def make_dirs(model_name):
    dirs = {}
    for tag in TAGS:
        d = os.path.join(OUT_BASE, model_name, tag)
        os.makedirs(d, exist_ok=True)
        dirs[tag] = d
    return dirs


def run_comparison(model_name, ckpts, img, device):
    print(f"\n{'='*55}\nModel: {model_name}\n{'='*55}")

    dirs = make_dirs(model_name)

    # 加载模型
    dense_model = load_dense(ckpts["pretrained"], model_name, device)
    pruned_models = {}
    for tag in TAGS:
        if tag == "pretrained":
            continue
        try:
            pruned_models[tag] = load_structured(ckpts[tag], device)
            print(f"  Loaded {tag}")
        except Exception as e:
            print(f"  ✗ {tag} 加载失败: {e}")

    # 保存总览热图
    if pruned_models:
        save_retention_heatmap(dense_model,
                               {t: m for t, m in pruned_models.items()},
                               os.path.join(OUT_BASE, model_name))

    # 提取 feature maps
    all_models  = {"pretrained": dense_model, **pruned_models}
    features, orders = {}, {}
    for tag, model in all_models.items():
        storage, order, pred = extract_features(model, img, device)
        features[tag] = storage
        orders[tag]   = order
        print(f"  [{tag}] pred: {NAMES[pred]}")

    # 逐层可视化
    for layer_name in orders["pretrained"]:
        feat_dict = {t: features[t][layer_name]
                     for t in TAGS if layer_name in features.get(t, {})}
        if "pretrained" not in feat_dict:
            continue

        sample = feat_dict["pretrained"][0]
        if sample.dim() == 3:
            save_conv_per_tag(feat_dict, layer_name, dirs,
                              dense_model, pruned_models)
        elif sample.dim() == 1:
            save_linear_per_tag(feat_dict, layer_name, dirs)

    print(f"  Saved to {OUT_BASE}/{model_name}/")


if __name__ == "__main__":
    dataset = datasets.CIFAR10(
        "./data", train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
    )
    img, label = dataset[IMG_IDX]
    print(f"[INFO] Image #{IMG_IDX}  label: {NAMES[label]}")

    os.makedirs(OUT_BASE, exist_ok=True)

    # 保存原始图片
    raw = np.clip(
        img.permute(1, 2, 0).numpy() * [.247, .2435, .2616]
        + [.4914, .4822, .4465], 0, 1
    )
    plt.figure(figsize=(3, 3))
    plt.imshow(raw)
    plt.axis('off')
    plt.title(f"Input: {NAMES[label]}", fontsize=10)
    plt.savefig(os.path.join(OUT_BASE, "original.pdf"), bbox_inches="tight")
    plt.close()

    for model_name, ckpts in MODELS.items():
        run_comparison(model_name, ckpts, img, DEVICE)