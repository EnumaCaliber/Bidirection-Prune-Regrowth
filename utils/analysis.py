import matplotlib.pyplot as plt
import os
def plot_weight_distributions(model, model_name, save_path="./weight_distributions", title="Weight Distributions"):
    os.makedirs(save_path, exist_ok=True)

    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_weights.extend(param.data.cpu().numpy().flatten())
            

    plt.hist(all_weights, bins=200)
    plt.title("All Weights")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    title = f"{model_name} Weight Distribution"
    plt.title(title)
#     plt.show()
    
    filename = f"{model_name}_hist.png"
    plt.tight_layout()
    target_folder = os.path.join(save_path, filename)
    plt.savefig(target_folder, dpi=300)
    plt.close()

    print(f"Saved weight distribution plots to: {save_path}")

def get_pruned_weights(model):
    weights = []
    for m in model.modules():
        if hasattr(m, 'weight_orig') and hasattr(m, 'weight_mask'):
            w_eff = m.weight_orig.data * m.weight_mask.data  # Elementwise multiply
            weights.append(w_eff.view(-1))  # flatten
    if len(weights) == 0:
        raise ValueError("No pruned weights found (missing weight_orig and weight_mask?)")
    all_weights = torch.cat(weights)
    return all_weights

def pruned_weight_distribution(model, save_path="./weight_distributions", title="Weight Distributions"):
    os.makedirs(save_path, exist_ok=True)

    all_weights = get_pruned_weights(model)
    nonzero_weights = all_weights[all_weights != 0].cpu().numpy()
    
    plt.hist(nonzero_weights, bins=200)
    
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.title('Histogram of Survived Pruned Weights')

    plt.grid(True)
    
    filename = f"pruned_weight_hist.png"
    plt.tight_layout()
    target_folder = os.path.join(save_path, filename)
    plt.savefig(target_folder, dpi=300)
    plt.close()

    print(f"Saved weight distribution plots to: {save_path}")