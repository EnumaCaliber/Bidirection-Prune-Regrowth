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
OUT_BASE = "distribution"


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

    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'])
    else:
        model.load_state_dict(checkpoint)

    return model.eval()


def register_hooks(model, storage, order):

    hooks = []
    conv_linear_count = 0

    for name, m in model.named_modules():
        # 只处理 Conv2d 和 Linear 层
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            conv_linear_count += 1

            def make_hook(name):
                def fn(module, inp, out):
                    storage[name] = out.detach()
                    if name not in order:
                        order.append(name)

                return fn

            hooks.append(m.register_forward_hook(make_hook(name)))


    return hooks


def collect_features(model_name, device, num_samples=50):
    """收集所有版本的特征"""
    dataset = datasets.CIFAR10("./data", train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.2470, 0.2435, 0.2616))
                               ]))

    all_features = {}
    layer_order = None

    for tag, ckpt_path in MODELS[model_name].items():


        if tag.startswith("pruned"):
            model = load_pruned(ckpt_path, device, model_name)
        else:
            model = load_pretrained(ckpt_path, device, model_name)

        all_features[tag] = {}

        for idx in range(num_samples):
            img, _ = dataset[idx]

            storage, order = {}, []
            hooks = register_hooks(model, storage, order)

            with torch.no_grad():
                model(img.unsqueeze(0).to(device))

            for h in hooks:
                h.remove()

            for name in order:
                feat = storage[name][0].flatten().cpu().numpy()
                if name not in all_features[tag]:
                    all_features[tag][name] = []
                all_features[tag][name].append(feat)

            if layer_order is None:
                layer_order = order

        # 合并数据
        for name in all_features[tag]:
            all_features[tag][name] = np.concatenate(all_features[tag][name])

    return all_features, layer_order


def plot_layer_comparison(model_name, all_features, layer_order):

    save_dir = os.path.join(OUT_BASE, model_name)
    os.makedirs(save_dir, exist_ok=True)


    for layer_idx, layer_name in enumerate(layer_order):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


        pretrain_data = all_features["pretrained"][layer_name]
        oneshot_data = all_features["pruned-oneshot"][layer_name]
        iterative_data = all_features["pruned-iterative"][layer_name]

        all_data = np.concatenate([pretrain_data, oneshot_data, iterative_data])
        data_min, data_max = np.min(all_data), np.max(all_data)
        bins = np.linspace(data_min, data_max, 50)

        # 左图：Pretrain vs Oneshot
        ax1.hist(pretrain_data, bins=bins, alpha=0.6, color='#2E86AB',
                 label=f'Pretrained (μ={np.mean(pretrain_data):.3f}, σ={np.std(pretrain_data):.3f})',
                 density=True, edgecolor='black', linewidth=0.5)
        ax1.hist(oneshot_data, bins=bins, alpha=0.6, color='#E63946',
                 label=f'Oneshot (μ={np.mean(oneshot_data):.3f}, σ={np.std(oneshot_data):.3f})',
                 density=True, edgecolor='black', linewidth=0.5)

        ax1.set_xlabel('Activation Value', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax1.set_title('Pretrained vs Oneshot', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')

        ax2.hist(pretrain_data, bins=bins, alpha=0.6, color='#2E86AB',
                 label=f'Pretrained (μ={np.mean(pretrain_data):.3f}, σ={np.std(pretrain_data):.3f})',
                 density=True, edgecolor='black', linewidth=0.5)
        ax2.hist(iterative_data, bins=bins, alpha=0.6, color='#F18F01',
                 label=f'Iterative (μ={np.mean(iterative_data):.3f}, σ={np.std(iterative_data):.3f})',
                 density=True, edgecolor='black', linewidth=0.5)

        ax2.set_xlabel('Activation Value', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax2.set_title('Pretrained vs Iterative', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        fig.suptitle(f'Layer {layer_idx}: {layer_name}',
                     fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        safe_name = layer_name.replace('.', '_')
        plt.savefig(os.path.join(save_dir, f'layer_{layer_idx:02d}_{safe_name}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close()
        if (layer_idx + 1) % 5 == 0:
            print(f" Finish: {layer_idx + 1}/{len(layer_order)} 层")



if __name__ == "__main__":
    os.makedirs(OUT_BASE, exist_ok=True)

    # 选择要分析的模型
    models_to_analyze = ["vgg16", "resnet20", "densenet", "effnet"]

    for model_name in models_to_analyze:
        all_features, layer_order = collect_features(model_name, DEVICE, num_samples=50)
        for idx, name in enumerate(layer_order):
            print(f"  {idx:2d}. {name}")
        plot_layer_comparison(model_name, all_features, layer_order)
