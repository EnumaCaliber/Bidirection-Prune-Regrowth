"""
structured_channel_plot.py
──────────────────────────────────────────────────────────────────────────────
For each Conv2d layer, plots a bar chart showing which channels are kept
(active) vs pruned, styled like the reference image.

Active channels  → bar height +1  (warm sand)
Pruned channels  → bar height -1  (soft coral/red)

Usage:
    python structured_channel_plot.py
──────────────────────────────────────────────────────────────────────────────
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME   = "vgg16"
DENSE_CKPT   = "vgg16/checkpoint/pretrain_vgg16_ckpt.pth"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# One structured pruned model to inspect
# (change to whichever checkpoint you want to visualise)
PRUNED_CKPT  = "vgg16/ckpt_structured_iterative/step22_sp0.874.pth"
PRUNED_LABEL = "Structured (87% sparsity)"

OUT_DIR      = "channel_plots"   # output folder

COLOR_ACTIVE = "#E8A94A"   # warm amber — kept channels
COLOR_PRUNED = "#7BAFD4"   # soft cornflower blue — removed channels

# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_dense(ckpt_path, model_name, device):
    from utils.model_loader import model_loader
    model = model_loader(model_name, device)
    raw   = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = raw["net"] if isinstance(raw, dict) and "net" in raw else raw
    model.load_state_dict(state)
    return model.eval()

def load_structured(ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval()
    raise ValueError(f"{ckpt_path} must be saved with torch.save(model, path).")

# ─────────────────────────────────────────────────────────────────────────────
# Channel activity inference
# ─────────────────────────────────────────────────────────────────────────────

def get_channel_activity(dense_model, pruned_model):
    """
    For each Conv2d layer, return a bool array of length = dense out_channels.
    True  = channel kept in pruned model (matched by cosine similarity)
    False = channel pruned away
    """
    results = {}

    dense_convs  = {n: m for n, m in dense_model.named_modules()
                    if isinstance(m, nn.Conv2d)}
    pruned_convs = {n: m for n, m in pruned_model.named_modules()
                    if isinstance(m, nn.Conv2d)}

    for name, d_conv in dense_convs.items():
        d_out = d_conv.out_channels

        if name not in pruned_convs:
            results[name] = np.zeros(d_out, dtype=bool)
            continue

        p_conv = pruned_convs[name]
        p_out  = p_conv.out_channels

        if p_out == d_out:
            # Layer untouched
            results[name] = np.ones(d_out, dtype=bool)
            continue

        # Match pruned channels back to dense channels via cosine similarity
        d_w = d_conv.weight.detach().reshape(d_out, -1).float()
        p_w = p_conv.weight.detach().reshape(p_out, -1).float()
        k   = min(d_w.shape[1], p_w.shape[1])
        d_w, p_w = d_w[:, :k], p_w[:, :k]

        d_norm = d_w / (d_w.norm(dim=1, keepdim=True) + 1e-8)
        p_norm = p_w / (p_w.norm(dim=1, keepdim=True) + 1e-8)
        sim    = (p_norm @ d_norm.T).cpu().numpy()   # (p_out, d_out)

        used   = set()
        kept   = set()
        for pi in range(p_out):
            for di in np.argsort(-sim[pi]):
                if di not in used:
                    used.add(di)
                    kept.add(int(di))
                    break

        activity = np.array([i in kept for i in range(d_out)], dtype=bool)
        results[name] = activity

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Plot one layer
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer(layer_name, activity, pruned_label, out_path):
    d_out    = len(activity)
    n_kept   = activity.sum()
    n_pruned = d_out - n_kept
    pct      = n_pruned / d_out * 100

    heights = np.where(activity, 1.0, -1.0)
    colors  = np.where(activity, COLOR_ACTIVE, COLOR_PRUNED)
    xs      = np.arange(d_out)

    fig, ax = plt.subplots(figsize=(max(6, d_out / 30), 3.2))
    fig.patch.set_facecolor("#FDFAF5")
    ax.set_facecolor("#FDFAF5")

    ax.bar(xs, heights, color=colors, width=1.0, linewidth=0)
    ax.axhline(0, color="#999999", linewidth=0.6, zorder=5)

    ax.set_xlim(-1, d_out)
    ax.set_ylim(-1.25, 1.25)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["-1", "", "0", "", "1"], fontsize=8)
    ax.set_xlabel("Channel Index", fontsize=9)
    ax.set_ylabel("Active (1) / Pruned (-1)", fontsize=9)

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#555555")

    ax.set_title(
        f"{pruned_label}: {n_pruned}/{d_out} channels pruned ({pct:.1f}%)  —  {layer_name}",
        fontsize=9.5, pad=8, color="#222222"
    )

    handles = [
        mpatches.Patch(facecolor=COLOR_ACTIVE, edgecolor="none",
                       label=f"Active ({n_kept})"),
        mpatches.Patch(facecolor=COLOR_PRUNED, edgecolor="none",
                       label=f"Pruned ({n_pruned})"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.9, edgecolor="#dddddd")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=180, facecolor="white")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading models …")
    dense  = load_dense(DENSE_CKPT, MODEL_NAME, DEVICE)
    pruned = load_structured(PRUNED_CKPT, DEVICE)

    print("Inferring channel activity …")
    activity = get_channel_activity(dense, pruned)

    print(f"Saving plots to '{OUT_DIR}/' …")
    for layer_name, act in activity.items():
        fname    = layer_name.replace(".", "_") + ".pdf"
        out_path = os.path.join(OUT_DIR, fname)
        plot_layer(layer_name, act, PRUNED_LABEL, out_path)
        n_kept = act.sum()
        print(f"  {layer_name:25s}  {n_kept}/{len(act)} kept  →  {fname}")

    print("Done.")

if __name__ == "__main__":
    main()