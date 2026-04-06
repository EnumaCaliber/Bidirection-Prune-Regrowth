import torch, os
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from utils.model_loader import model_loader

MODELS = {
    "vgg16": {
        "pretrained":  "vgg16/checkpoint/pretrain_vgg16_ckpt.pth",
        "pruned-high": "vgg16/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.9953.pth",
        "pruned-low":  "vgg16/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.9424.pth",
    },
}

TAGS       = ["pretrained", "pruned-high", "pruned-low"]
COL_LABELS = ["Baseline", "High Sparsity", "Low Sparsity"]
COL_COLORS = ["steelblue", "tomato", "orange"]

HIGHLIGHT_COLOR = np.array([1.0, 0.2, 0.2])

IMG_IDX  = 0
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
NAMES    = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']
OUT_BASE = "feature_comparison"


# ── model loading ─────────────────────────────────────────────────────────────
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


# ── hooks ─────────────────────────────────────────────────────────────────────
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
    storage, order = {}, []
    hooks = register_hooks(model, storage, order)
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
    for h in hooks:
        h.remove()
    return storage, order, pred


# ── helpers ───────────────────────────────────────────────────────────────────
def all_tags_present(feat_dict):
    return all(t in feat_dict for t in TAGS)


def safe_name(layer_name):
    return layer_name.replace(".", "_")


def make_dirs(base, model_name):
    dirs = {}
    for tag in TAGS:
        d = os.path.join(base, model_name, tag)
        os.makedirs(d, exist_ok=True)
        dirs[tag] = d
    return dirs


def make_highlight_mask(base: torch.Tensor, pruned: torch.Tensor) -> torch.Tensor:
    """True 表示与 baseline 符号相反（异号）。"""
    return (base * pruned) < 0


def to_gray_rgb(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.cpu().float().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return np.stack([arr, arr, arr], axis=-1)


def apply_highlight(gray_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = gray_rgb.copy()
    out[mask] = HIGHLIGHT_COLOR
    return out


def add_legend(ax, tag):
    if tag == "pretrained":
        return
    handles = [
        mpatches.Patch(facecolor='gray',          label='same sign as baseline'),
        mpatches.Patch(facecolor=HIGHLIGHT_COLOR, label='opposite sign'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=6,
              framealpha=0.75, handlelength=1.2)


# ── conv ──────────────────────────────────────────────────────────────────────
def save_conv_per_tag(feat_dict, layer_name, dirs, max_ch=64):
    if not all_tags_present(feat_dict):
        return
    feats = {t: feat_dict[t][0] for t in TAGS}
    if feats["pretrained"].dim() != 3:
        return

    total = min(feats["pretrained"].shape[0], max_ch)
    n     = int(total ** 0.5)
    fname = safe_name(layer_name) + ".pdf"
    base  = feats["pretrained"]
    h, w  = base[0].shape[-2:]

    for tag, label, color in zip(TAGS, COL_LABELS, COL_COLORS):
        pruned = feats[tag]
        rows   = []

        for r in range(n):
            row_imgs = []
            for c in range(n):
                ch   = r * n + c
                gray = to_gray_rgb(base[ch])
                if tag == "pretrained":
                    row_imgs.append(gray)
                else:
                    mask = make_highlight_mask(base[ch], pruned[ch])
                    row_imgs.append(apply_highlight(gray, mask.cpu().numpy()))
            rows.append(np.concatenate(row_imgs, axis=1))

        grid = np.concatenate(rows, axis=0)

        if tag != "pretrained":
            flat_mask   = make_highlight_mask(base[:total].reshape(-1),
                                              pruned[:total].reshape(-1))
            changed_pct = flat_mask.float().mean().item() * 100
        else:
            changed_pct = None

        fig_w = max(w * n / 50, 4) + 0.3
        fig_h = max(h * n / 50, 4)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(grid, interpolation='nearest', aspect='equal')
        ax.axis('off')

        title = f"{label} — {layer_name}"
        if changed_pct is not None:
            title += f"\n{changed_pct:.1f}% pixels flipped sign"
        ax.set_title(title, fontsize=8, color=color)
        add_legend(ax, tag)

        plt.tight_layout()
        plt.savefig(os.path.join(dirs[tag], fname), bbox_inches="tight", dpi=150)
        plt.close()


# ── linear ────────────────────────────────────────────────────────────────────
def save_linear_per_tag(feat_dict, layer_name, dirs):
    if not all_tags_present(feat_dict):
        return
    feats = {t: feat_dict[t][0] for t in TAGS}
    if feats["pretrained"].dim() != 1:
        return

    fname = safe_name(layer_name) + ".pdf"
    base  = feats["pretrained"]

    g_min = min(feats[t].min().item() for t in TAGS)
    g_max = max(feats[t].max().item() for t in TAGS)

    for tag, label, color in zip(TAGS, COL_LABELS, COL_COLORS):
        pruned = feats[tag]
        norm   = ((pruned - g_min) / (g_max - g_min + 1e-8)).cpu().numpy()

        if tag == "pretrained":
            bar_colors  = ['gray'] * len(norm)
            changed_pct = None
        else:
            mask        = make_highlight_mask(base, pruned).cpu().numpy()
            bar_colors  = [tuple(HIGHLIGHT_COLOR) if m else (0.5, 0.5, 0.5)
                           for m in mask]
            changed_pct = mask.mean() * 100

        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.bar(range(len(norm)), norm, color=bar_colors, width=1.0)
        ax.set_ylim([0, 1.05])
        ax.axhline(y=0, color='k', linewidth=0.4)
        ax.tick_params(labelbottom=False)

        title = f"{label} — {layer_name}"
        if changed_pct is not None:
            title += f"\n{changed_pct:.1f}% neurons flipped sign"
        ax.set_title(title, fontsize=8, color=color)
        add_legend(ax, tag)

        plt.tight_layout()
        plt.savefig(os.path.join(dirs[tag], fname), bbox_inches="tight", dpi=150)
        plt.close()


# ── main ──────────────────────────────────────────────────────────────────────
def run_comparison(model_name, ckpts, img, device):
    dirs = make_dirs(OUT_BASE, model_name)

    models = {
        "pretrained":  load_pretrained(ckpts["pretrained"],  device, model_name),
        "pruned-high": load_pruned(ckpts["pruned-high"], device, model_name),
        "pruned-low":  load_pruned(ckpts["pruned-low"],  device, model_name),
    }

    features, orders = {}, {}
    for tag, model in models.items():
        storage, order, pred = extract_features(model, img, device)
        features[tag] = storage
        orders[tag]   = order
        print(f"[{model_name}/{tag}] pred: {NAMES[pred]}")

    for layer_name in orders["pretrained"]:
        feat_dict = {t: features[t][layer_name]
                     for t in TAGS if layer_name in features[t]}
        if not all_tags_present(feat_dict):
            continue

        sample = feat_dict["pretrained"][0]
        if sample.dim() == 3:
            save_conv_per_tag(feat_dict, layer_name, dirs)
        elif sample.dim() == 1:
            save_linear_per_tag(feat_dict, layer_name, dirs)

    print(f"[{model_name}] saved to {OUT_BASE}/{model_name}/")


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
    print(f"[INFO] image {IMG_IDX}, label: {NAMES[label]}")

    os.makedirs(OUT_BASE, exist_ok=True)
    raw = np.clip(
        img.permute(1, 2, 0).numpy() * [.247, .2435, .2616]
        + [.4914, .4822, .4465], 0, 1
    )
    plt.figure(figsize=(3, 3))
    plt.imshow(raw)
    plt.axis('off')
    plt.savefig(os.path.join(OUT_BASE, "original.pdf"), bbox_inches="tight")
    plt.close()

    for model_name, ckpts in MODELS.items():
        print(f"\n{'=' * 50}\nmodel: {model_name}\n{'=' * 50}")
        run_comparison(model_name, ckpts, img, DEVICE)