import torch, math
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from utils.model_loader import model_loader
import os


cfgs = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
model_name = 'resnet20'
OUTPUT_DIR = f"feature/feature_output_{model_name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


NAME_MAP = {
    "features.0": "Conv1_1", "features.3": "Conv1_2", "features.6": "Pool1",
    "features.7": "Conv2_1", "features.10": "Conv2_2", "features.13": "Pool2",
    "features.14": "Conv3_1", "features.17": "Conv3_2", "features.20": "Conv3_3", "features.23": "Pool3",
    "features.24": "Conv4_1", "features.27": "Conv4_2", "features.30": "Conv4_3", "features.33": "Pool4",
    "features.34": "Conv5_1", "features.37": "Conv5_2", "features.40": "Conv5_3", "features.43": "Pool5",
    "classifier": "FC",
}
layer_outputs = {}
layer_order = []


def make_hook(name):
    def fn(module, inp, out):
        layer_outputs[name] = out.detach()
        layer_order.append(name)

    return fn


def register_hooks(model):
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(make_hook(name)))
    return hooks


def show_feat(feat, name, title, max_ch=16):
    """feat: (C, H, W)"""
    C, H, W = feat.shape
    n = min(C, max_ch)
    cols = min(n, 16)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(f"{name}: {title}  |  ch={C}, {H}x{W}", fontsize=13, fontweight='bold')
    for i, ax in enumerate(np.array(axes).flatten()):
        if i < n:
            ax.imshow(feat[i].cpu().numpy(), cmap='viridis')
            ax.set_title(f"ch {i}", fontsize=8)
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}:{title}.png"), dpi=150)
    plt.close()


# ── Main ──
if __name__ == "__main__":
    CKPT = "vgg16/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.995.pth"
    NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_pruned(CKPT, device, model_name)
    print(model)

    # 数据
    dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ]))
    idx = 0
    img, label = dataset[idx]
    print(f"[INFO] 第 {idx} 张，真实标签: {NAMES[label]}")

    # 推理
    hooks = register_hooks(model)
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
    for h in hooks:
        h.remove()
    print(f"[INFO] 预测: {NAMES[pred]}")

    raw = np.clip(img.permute(1, 2, 0).numpy() * [.247, .2435, .2616] + [.4914, .4822, .4465], 0, 1)
    plt.figure(figsize=(3, 3))
    plt.imshow(raw)
    plt.title(f"label: {NAMES[label]}")
    plt.axis('off')
    plt.show()

    for name in layer_order:
        feat = layer_outputs[name][0]
        title = NAME_MAP.get(name, name)  # 映射为语义名
        if feat.dim() == 3:
            show_feat(feat, name, title)
        elif feat.dim() == 1:
            plt.figure(figsize=(8, 5))
            plt.bar(range(len(feat)), feat.cpu().numpy())
            plt.xticks(range(len(NAMES)), NAMES, rotation=45)
            plt.title(f"{name}:{title}  |  dim={len(feat)}")
            plt.savefig(os.path.join(OUTPUT_DIR, f"{name}:{title}.png"), dpi=150)
            plt.close()
