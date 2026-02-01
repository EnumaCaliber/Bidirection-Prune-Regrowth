import torch
import matplotlib.pyplot as plt
import os


def visualize_pruning_heatmap(checkpoint_path, save_path='prune_visual.png'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    state_dict = checkpoint
    masks = {}
    for key, value in state_dict.items():
        if 'weight_mask' in key:
            layer_name = key.replace('.weight_mask', '')
            masks[layer_name] = value.cpu().numpy()

    n_layers = len(masks)
    fig, axes = plt.subplots(n_layers, 1, figsize=(14, 2.5 * n_layers))

    if n_layers == 1:
        axes = [axes]

    for idx, (name, mask) in enumerate(masks.items()):
        if mask.ndim == 4:  # Conv: (out_ch, in_ch, h, w)
            mask_2d = mask.reshape(mask.shape[0], -1)
        elif mask.ndim == 2:  # Linear
            mask_2d = mask
        elif mask.ndim == 1:
            mask_2d = mask.reshape(1, -1)
        else:
            mask_2d = mask.reshape(-1, mask.shape[-1])
        if mask_2d.shape[1] > 1000:
            step = mask_2d.shape[1] // 1000
            mask_2d = mask_2d[:, ::step]
        if mask_2d.shape[0] > 200:
            step = mask_2d.shape[0] // 200
            mask_2d = mask_2d[::step, :]

        axes[idx].imshow(mask_2d, cmap='RdBu_r', aspect='auto',
                         interpolation='nearest', vmin=0, vmax=1)
        sparsity = (1 - mask.mean()) * 100
        layer_name = name.replace('module.', '').replace('.weight', '')
        axes[idx].set_title(f'{layer_name}:Sparsity: {sparsity:.2f}%',
                            fontsize=11, fontweight='bold', pad=10)


    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    model_name = "resnet20"
    root_path = 'ckpt_after_prune_oneshot'
    prun_name = 'pruned_oneshot_mask_0.98.pth'
    checkpoint_path = os.path.join(model_name, root_path, prun_name)
    visualize_pruning_heatmap(checkpoint_path)
