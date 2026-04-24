"""
generate_pruning_heatmap.py
Two heatmaps with fully discrete colours — one flat colour per unique value.
  heatmap_structured.pdf   — Structured:   Low vs High sparsity
  heatmap_unstructured.pdf — Unstructured: Low vs High sparsity
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "vgg16"
DENSE_CKPT = "vgg16/checkpoint/pretrain_vgg16_ckpt.pth"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR    = "."

STRUCTURED = {
    "low":  {"ckpt": "vgg16/ckpt_after_prune_structured_oneshot/pruned_structured_l1_sp0.9_it1.pth",
             "label": "Low Sparsity (90%)"},
    "high": {"ckpt": "vgg16/ckpt_after_prune_structured_oneshot/pruned_structured_l1_sp0.94_it1.pth",
             "label": "High Sparsity (94%)"},
}

UNSTRUCTURED = {
    "low":  {"ckpt": "vgg16/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.94.pth",
             "label": "Low Sparsity (94%)"},
    "high": {"ckpt": "vgg16/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.99.pth",
             "label": "High Sparsity (99%)"},
}

# 12 maximally distinct colours from SciPainter — no two adjacent ones look alike
PALETTE = [
    "#4F99C9",  # steel blue
    "#FFCB5B",  # warm yellow
    "#8582BD",  # soft purple
    "#A8D3A0",  # sage green
    "#E87B1E",  # orange
    "#A6D0E6",  # sky blue
    "#731A73",  # deep purple
    "#4DAF4A",  # bright green
    "#F8B072",  # peach
    "#0098B2",  # teal
    "#D65190",  # pink
    "#3D505A",  # dark slate
]

# Round values to this many decimal places before comparing
ROUND_DP = 3

# ─────────────────────────────────────────────────────────────────────────────
# Colour assignment — one flat colour per unique value
# ─────────────────────────────────────────────────────────────────────────────

import colorsys

COLOR_100 = "#5C6B73"  # muted slate — fixed for 100%

# 16 soft muted colours, alternating warm/cool so adjacent values never clash
SOFT_PALETTE = [
    "#F2C97E",  # warm sand
    "#8FC4D4",  # soft teal-blue
    "#F0A787",  # soft coral
    "#9DC8A0",  # sage green
    "#C4A8D8",  # soft lavender
    "#F5DE9A",  # pale yellow
    "#7AB8C8",  # medium sky blue
    "#E8A8B8",  # dusty rose
    "#B4D4A0",  # light mint green
    "#D4A8C0",  # soft mauve
    "#A8C8E0",  # powder blue
    "#F0C090",  # peach
    "#A8D4BC",  # seafoam
    "#D8B8E8",  # soft purple
    "#B8D8A8",  # pale green
    "#E8C8A0",  # warm cream-orange
]

def make_value_color_dict(all_values: list) -> dict:
    """
    Each unique non-100% value gets one colour from SOFT_PALETTE in order.
    Palette is ordered warm/cool alternating → adjacent unique values always
    look clearly different.
    """
    uniq     = sorted(set(round(v, ROUND_DP) for v in all_values))
    non_full = [v for v in uniq if v < 0.999]
    color_map = {}
    for i, v in enumerate(non_full):
        color_map[v] = SOFT_PALETTE[i % len(SOFT_PALETTE)]
    for v in uniq:
        if v >= 0.999:
            color_map[v] = COLOR_100
    return color_map, uniq

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

def load_unstructured_sd(ckpt_path, device):
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    return raw["net"] if isinstance(raw, dict) and "net" in raw else raw

# ─────────────────────────────────────────────────────────────────────────────
# Retention rates
# ─────────────────────────────────────────────────────────────────────────────

def structured_retention(dense_model, pruned_model):
    d = {n: m for n, m in dense_model.named_modules()  if isinstance(m, nn.Conv2d)}
    p = {n: m for n, m in pruned_model.named_modules() if isinstance(m, nn.Conv2d)}
    return {n: (p[n].out_channels / dm.out_channels
                if n in p and dm.out_channels > 0 else 1.0)
            for n, dm in d.items()}

def unstructured_retention(dense_model, pruned_sd):
    result = {}
    for name, mod in dense_model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        mk, wk = f"{name}.weight_mask", f"{name}.weight"
        if mk in pruned_sd:
            m = pruned_sd[mk].float()
            result[name] = m.sum().item() / m.numel()
        elif wk in pruned_sd:
            w = pruned_sd[wk].float()
            result[name] = (w != 0).sum().item() / w.numel()
        else:
            result[name] = 1.0
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(layer_names, rates_low, rates_high,
                 label_low, label_high, col_subtitle, title,
                 color_dict, uniq_vals, output_path):

    n     = len(layer_names)
    fig_h = max(4.5, n * 0.46)
    fig, ax = plt.subplots(figsize=(7.4, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()

    cw, ch = 1.0, 1.0

    for i, ln in enumerate(layer_names):
        for j, val in enumerate([rates_low.get(ln, 1.0), rates_high.get(ln, 1.0)]):
            key = round(val, ROUND_DP)
            hex_c = color_dict.get(key, "#cccccc")
            rgb   = mcolors.to_rgb(hex_c)

            ax.add_patch(plt.Rectangle(
                (j - cw/2, i - ch/2), cw, ch,
                facecolor=rgb, edgecolor="none"))

            lum = 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]
            tc  = "#1a1a1a" if lum > 0.45 else "white"
            ax.text(j, i, f"{val*100:.1f}%",
                    ha="center", va="center",
                    fontsize=9.0, fontweight="bold", color=tc)

    # Column headers
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"{label_low}\n({col_subtitle})",
         f"{label_high}\n({col_subtitle})"],
        fontsize=9.5, fontweight="bold", color="#2c2c2c")
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_tick_params(length=0, pad=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(layer_names, fontsize=8.5, color="#2c2c2c")
    ax.yaxis.set_tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Footer averages
    avg_lo = np.mean([rates_low.get(ln,  1.0) for ln in layer_names]) * 100
    avg_hi = np.mean([rates_high.get(ln, 1.0) for ln in layer_names]) * 100
    fig.text(0.38, 0.004,
             f"Layer avg  ·  Low: {avg_lo:.1f}%   High: {avg_hi:.1f}%",
             ha="center", fontsize=8, color="#888888")

    ax.set_title(title, fontsize=12, pad=15, fontweight="bold", color="#1a1a1a")
    plt.savefig(output_path, bbox_inches="tight", dpi=200, facecolor="white")
    plt.close()
    print(f"  Saved → {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading dense baseline …")
    dense  = load_dense(DENSE_CKPT, MODEL_NAME, DEVICE)
    layers = [n for n, m in dense.named_modules() if isinstance(m, nn.Conv2d)]
    print(f"  {len(layers)} Conv2d layers.")

    print("\n[Structured] loading …")
    r_st_lo = structured_retention(dense, load_structured(STRUCTURED["low"]["ckpt"],  DEVICE))
    r_st_hi = structured_retention(dense, load_structured(STRUCTURED["high"]["ckpt"], DEVICE))

    print("[Unstructured] loading …")
    r_un_lo = unstructured_retention(dense, load_unstructured_sd(UNSTRUCTURED["low"]["ckpt"],  DEVICE))
    r_un_hi = unstructured_retention(dense, load_unstructured_sd(UNSTRUCTURED["high"]["ckpt"], DEVICE))

    # Build shared colour dict from ALL values across both plots
    all_vals = (list(r_st_lo.values()) + list(r_st_hi.values()) +
                list(r_un_lo.values()) + list(r_un_hi.values()))
    color_dict, uniq_vals = make_value_color_dict(all_vals)
    print(f"\n  {len(uniq_vals)} unique values → {len(uniq_vals)} distinct colours")

    print()
    plot_heatmap(layers, r_st_lo, r_st_hi,
                 STRUCTURED["low"]["label"], STRUCTURED["high"]["label"],
                 "channel retention",
                 "Structured Pruning — Channel Retention per Layer",
                 color_dict, uniq_vals,
                 os.path.join(OUT_DIR, "heatmap_structured.pdf"))

    plot_heatmap(layers, r_un_lo, r_un_hi,
                 UNSTRUCTURED["low"]["label"], UNSTRUCTURED["high"]["label"],
                 "weight mask retention",
                 "Unstructured Pruning — Weight Retention per Layer",
                 color_dict, uniq_vals,
                 os.path.join(OUT_DIR, "heatmap_unstructured.pdf"))

    print("Done.")

if __name__ == "__main__":
    main()