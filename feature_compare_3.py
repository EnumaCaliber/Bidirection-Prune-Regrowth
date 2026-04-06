import torch, os
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from utils.model_loader import model_loader

MODELS = {
    "vgg16": {
        "pretrained":  "vgg16/checkpoint/pretrain_vgg16_ckpt.pth",
        "pruned-high": "vgg16/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.9862.pth",
        "pruned-low":  "vgg16/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.9953.pth",
    },
}

TAGS       = ["pretrained", "pruned-high", "pruned-low"]
COL_LABELS = ["Baseline", "High Sparsity", "Low Sparsity"]
COL_COLORS = ["steelblue", "tomato", "orange"]

IMG_IDX = 0
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
NAMES   = ['airplane', 'automobile', 'bird', 'cat', 'deer',
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
def normalize_to_range(feat, min_val=-1, max_val=1):
    feat_min, feat_max = feat.min(), feat.max()
    if feat_max - feat_min < 1e-8:
        return torch.zeros_like(feat)
    return (feat - feat_min) / (feat_max - feat_min) * (max_val - min_val) + min_val


def all_tags_present(feat_dict):
    return all(t in feat_dict for t in TAGS)


# ── conv comparison ───────────────────────────────────────────────────────────
def show_comparison_feat(feat_dict, name, save_dir, max_ch=5):
    """Baseline | High Sparsity | Low Sparsity"""
    if not all_tags_present(feat_dict):
        return
    feats = {t: feat_dict[t][0] for t in TAGS}
    if feats["pretrained"].dim() != 3:
        return

    n = min(feats["pretrained"].shape[0], max_ch)
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 2.5))
    if n == 1:
        axes = axes.reshape(1, -1)

    for j, label in enumerate(COL_LABELS):
        axes[0, j].set_title(label, fontsize=9, pad=4)

    for i in range(n):
        for j, t in enumerate(TAGS):
            norm = normalize_to_range(feats[t][i])
            axes[i, j].imshow(norm.cpu().numpy(), cmap='viridis', vmin=-1, vmax=1)
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name.replace('.', '_')}_comparison.pdf"),
                bbox_inches="tight")
    plt.close()


# ── linear comparison ─────────────────────────────────────────────────────────
def show_comparison_linear(feat_dict, name, save_dir):
    """Baseline | High Sparsity | Low Sparsity"""
    if not all_tags_present(feat_dict):
        return
    feats = {t: feat_dict[t][0] for t in TAGS}
    if feats["pretrained"].dim() != 1:
        return

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4))
    for j, (t, label, color) in enumerate(zip(TAGS, COL_LABELS, COL_COLORS)):
        norm = normalize_to_range(feats[t])
        axes[j].bar(range(len(norm)), norm.cpu().numpy(), color=color)
        axes[j].set_title(label, fontsize=9)
        axes[j].set_ylim([-1.1, 1.1])
        axes[j].axhline(y=0, color='k', linewidth=0.5)
        axes[j].tick_params(labelbottom=False, labelleft=False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name.replace('.', '_')}_comparison.pdf"),
                bbox_inches="tight")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────
def run_comparison(model_name, ckpts, img, device):
    save_dir = os.path.join(OUT_BASE, model_name)
    os.makedirs(save_dir, exist_ok=True)

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
            show_comparison_feat(feat_dict, layer_name, save_dir)
        elif sample.dim() == 1:
            show_comparison_linear(feat_dict, layer_name, save_dir)

    print(f"[{model_name}] saved to {save_dir}/")


if __name__ == "__main__":
    dataset = datasets.CIFAR10(
        "./data", train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    )
    img, label = dataset[IMG_IDX]
    print(f"[INFO] image {IMG_IDX}, label: {NAMES[label]}")

    os.makedirs(OUT_BASE, exist_ok=True)
    raw = np.clip(
        img.permute(1, 2, 0).numpy() * [.247, .2435, .2616] + [.4914, .4822, .4465], 0, 1
    )
    plt.figure(figsize=(3, 3))
    plt.imshow(raw); plt.axis('off')
    plt.savefig(os.path.join(OUT_BASE, "original.pdf"), bbox_inches="tight")
    plt.close()

    for model_name, ckpts in MODELS.items():
        print(f"\n{'=' * 50}\nmodel: {model_name}\n{'=' * 50}")
        run_comparison(model_name, ckpts, img, DEVICE)