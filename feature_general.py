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
        "pruned": "vgg16/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.995.pth",
    },
    "resnet20": {
        "pretrained": "resnet20/checkpoint/pretrain_resnet20_ckpt.pth",
        "pruned": "resnet20/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.995.pth",
    },
    "densenet": {
        "pretrained": "densenet/checkpoint/pretrain_densenet_ckpt.pth",
        "pruned": "densenet/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.995.pth",
    },
    "effnet": {
        "pretrained": "effnet/checkpoint/pretrain_effnet_ckpt.pth",
        "pruned": "effnet/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.995.pth",
    },
}
IMG_IDX = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
OUT_BASE = "feature"


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
        order.append(name)

    return fn


def register_hooks(model, storage, order):
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(make_hook(name, storage, order)))
    return hooks


def show_feat(feat, name, save_dir, max_ch=16):
    """feat: (C, H, W)"""
    C, H, W = feat.shape
    n = min(C, max_ch)
    cols = min(n, 16)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(f"{name}  |  ch={C}, {H}x{W}", fontsize=14, fontweight='bold')
    for i, ax in enumerate(np.array(axes).flatten()):
        if i < n:
            ax.imshow(feat[i].cpu().numpy(), cmap='viridis')
            ax.set_title(f"ch {i}", fontsize=9)
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f"{name.replace('.', '_')}.png"), dpi=150)
    plt.close()


def run_model(model_name, ckpt_path, tag, img, device):
    """tag: 'pretrained' 或 'pruned'"""
    save_dir = os.path.join(OUT_BASE, model_name, tag)
    os.makedirs(save_dir, exist_ok=True)

    if tag == "pruned":
        model = load_pruned(ckpt_path, device, model_name)
    else:
        model = load_pretrained(ckpt_path, device, model_name)

    storage, order = {}, []
    hooks = register_hooks(model, storage, order)
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
    for h in hooks:
        h.remove()
    print(f"[{model_name}/{tag}] 预测: {NAMES[pred]}")

    for name in order:
        feat = storage[name][0]
        if feat.dim() == 3:
            show_feat(feat, name, save_dir)
        elif feat.dim() == 1:
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(feat)), feat.cpu().numpy())
            plt.xticks(range(len(NAMES)), NAMES, rotation=45)
            plt.title(f"{name}  |  dim={len(feat)}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{name.replace('.', '_')}.png"), dpi=150)
            plt.close()

    print(f"[{model_name}/{tag}] 已保存到 {save_dir}/")


if __name__ == "__main__":

    dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ]))
    img, label = dataset[IMG_IDX]
    print(f"[INFO] 第 {IMG_IDX} 张，真实标签: {NAMES[label]}")

    os.makedirs(OUT_BASE, exist_ok=True)
    raw = np.clip(img.permute(1, 2, 0).numpy() * [.247, .2435, .2616] + [.4914, .4822, .4465], 0, 1)
    plt.figure(figsize=(3, 3))
    plt.imshow(raw)
    plt.title(f"label: {NAMES[label]}")
    plt.axis('off')
    plt.savefig(os.path.join(OUT_BASE, "original.png"), dpi=150)
    plt.close()

    for model_name, ckpts in MODELS.items():
        for tag, ckpt_path in ckpts.items():
            run_model(model_name, ckpt_path, tag, img, DEVICE)
