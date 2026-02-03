import torch, math, os
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from utils.model_loader import model_loader

model_name = 'vgg16'
OUTPUT_DIR = f"feature/feature_output_pretrained_{model_name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NAME_MAP = {
    "features.0": "Conv1_1", "features.3": "Conv1_2", "features.6": "Pool1",
    "features.7": "Conv2_1", "features.10": "Conv2_2", "features.13": "Pool2",
    "features.14": "Conv3_1", "features.17": "Conv3_2", "features.20": "Conv3_3", "features.23": "Pool3",
    "features.24": "Conv4_1", "features.27": "Conv4_2", "features.30": "Conv4_3", "features.33": "Pool4",
    "features.34": "Conv5_1", "features.37": "Conv5_2", "features.40": "Conv5_3", "features.43": "Pool5",
    "classifier": "FC",
}


# ── Hook ──
def make_hook(name, storage):
    def fn(module, inp, out):
        storage["outputs"][name] = out.detach()
        storage["order"].append(name)
    return fn

def register_hooks(model, storage):
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(make_hook(name, storage)))
    return hooks


# ── 可视化 ──
def show_feat(feat, name, title, max_ch=16):
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
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_{title}.png"), dpi=150)
    plt.close()


# ── Main ──
if __name__ == "__main__":
    CKPT       = "vgg16/checkpoint/pretrain_vgg16_ckpt.pth"
    NAMES      = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_loader(model_name, device)
    checkpoint = torch.load(CKPT, map_location=device)
    model.load_state_dict(checkpoint['net'])
    model.eval()
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
    storage = {"outputs": {}, "order": []}
    hooks = register_hooks(model, storage)
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
    for h in hooks:
        h.remove()
    print(f"[INFO] 预测: {NAMES[pred]}")

    # 原图
    raw = np.clip(img.permute(1, 2, 0).numpy() * [.247, .2435, .2616] + [.4914, .4822, .4465], 0, 1)
    plt.figure(figsize=(3, 3))
    plt.imshow(raw)
    plt.title(f"label: {NAMES[label]}")
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, "00_original.png"), dpi=150)
    plt.close()

    # 逐层可视化
    for name in storage["order"]:
        feat  = storage["outputs"][name][0]
        title = NAME_MAP.get(name, name)
        if feat.dim() == 3:
            show_feat(feat, name, title)
        elif feat.dim() == 1:
            plt.figure(figsize=(8, 5))
            plt.bar(range(len(feat)), feat.cpu().numpy())
            plt.xticks(range(len(NAMES)), NAMES, rotation=45)
            plt.title(f"{name}:{title}  |  dim={len(feat)}")
            plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_{title}.png"), dpi=150)
            plt.close()

    print(f"[INFO] 所有图片已保存到 {OUTPUT_DIR}/")