"""
RL-based Regrowth with Saliency-Based Selection
Combines:
1. RL allocation (from nas_rl) - LSTM controller decides how much to regrow per layer
2. Saliency-based selection (from RigL + FairPrune) - gradient magnitude determines which weights to regrow

Key improvements over reference mask approach:
- Principled parameter selection based on gradient importance
- RigL-style direct mask manipulation
- Flexible weight initialization strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import copy
import os
import time
import wandb
import random
from torch.distributions import Categorical
from transformers import get_cosine_schedule_with_warmup
from collections import deque, defaultdict
from tqdm import tqdm

from utils.model_loader import model_loader
from utils.data_loader import data_loader
from utils.analysis_utils import (
    BlockwiseFeatureExtractor, compute_block_ssim,
    load_model, load_model_name, prune_weights_reparam, count_pruned_params,
    get_cifar_resnet20_blocks
)

# Start wandb run
run = wandb.init(
    project="DAC26_solu26_saliency",
    name="solu26_saliency_regrowth_from99",
    config={
        "learning_rate": 5e-3,
        "architecture": "VGG16",
        "regrow_step": 0.015,
        "epochs": 500,
    },
)

method = "oneshot"
model_name = "vgg16"


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


class SaliencyComputer:
    """
    Computes gradient-based saliency scores for parameters using FairPrune formula.
    
    Saliency measures parameter importance using the formula:
    - Saliency(θ) = H_ii * θ² 
    - where H_ii ≈ (∂L/∂θ)² (Hessian diagonal approximation)
    - and θ is the parameter value
    
    This combines:
    - Curvature (H_ii): How sensitive loss is to parameter changes
    - Magnitude (θ²): How large the parameter value is
    
    High saliency = important for accuracy
    Low saliency = less important, safe to prune
    
    Implementation follows:
    - RigL's gradient accumulation strategy
    - FairPrune's H*θ² importance formula
    - Optimal Brain Damage's Taylor expansion analysis
    """

    def __init__(self, model, criterion, device='cuda'):
        self.model = model
        self.criterion = criterion
        self.device = device
        # Storage for accumulated gradients (like RigL's dense_grad)
        self.accumulated_grads = {}
        self.grad_count = 0

    def reset_accumulated_grads(self):
        """Reset accumulated gradients (call before new saliency computation)"""
        self.accumulated_grads = {}
        self.grad_count = 0

    def compute_saliency_scores(self, data_loader, target_layers, max_batches=None, num_classes=10):
        """
        Compute saliency scores for target layers using squared gradients.
        
        Methodology:
        1. Accumulate gradients over multiple batches
        2. Use squared gradient magnitude as saliency: S(θ) = (∂L/∂θ)²
        3. Average across all batches and classes
        
        This approximates the Hessian diagonal (Fisher Information), measuring
        the curvature of the loss landscape with respect to each parameter.
        
        Args:
            data_loader: DataLoader for gradient computation
            target_layers: List of layer names to compute saliency for
            max_batches: Maximum batches to process
            num_classes: Number of classes
        
        Returns:
            saliency_dict: Dict[layer_name] -> saliency tensor (same shape as weights)
        """
        self.model.eval()
        self.reset_accumulated_grads()

        print(f"\nComputing saliency scores (RigL-style)...")
        print(f"  Target layers: {len(target_layers)}")
        print(f"  Max batches: {max_batches}")

        # Get module dict for quick access
        module_dict = dict(self.model.named_modules())

        # Initialize accumulators
        for layer_name in target_layers:
            module = module_dict.get(layer_name)
            if module is not None and hasattr(module, 'weight'):
                weight_shape = module.weight.shape
                self.accumulated_grads[layer_name] = torch.zeros(weight_shape, device=self.device)

        batch_count = 0

        # Accumulate gradients across batches and classes
        for inputs, labels in tqdm(data_loader, desc="Accumulating gradients"):
            # if batch_count >= max_batches:
            #     break

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Compute gradients using torch.autograd.grad (works with non-leaf tensors)
            # This is needed because with pruning, module.weight is not a leaf tensor
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)

            # Accumulate gradients for target layers
            # Match parameters by identity (same approach as saliency_analysis.py)
            for param, grad in zip(self.model.parameters(), grads):
                if grad is None:
                    continue

                # Find which layer this parameter belongs to
                for name, p in self.model.named_parameters():
                    if p is param:
                        # Check if this parameter belongs to one of our target layers
                        # Since model_pretrained is unpruned, parameter names are "layer.weight"
                        for layer_name in target_layers:
                            if name == f"{layer_name}.weight":
                                # FairPrune formula: Saliency = H_ii * θ²
                                # where H_ii ≈ (∂L/∂θ)² (Hessian diagonal approximation)
                                hessian_approx = grad.pow(2).detach()  # H_ii
                                param_squared = param.data.pow(2).detach()  # θ²
                                saliency_contribution = hessian_approx * param_squared  # H_ii * θ²
                                self.accumulated_grads[layer_name] += saliency_contribution
                                break
                        break

            batch_count += 1
            self.grad_count += 1

        # Average accumulated gradients
        saliency_dict = {}
        for layer_name in target_layers:
            if layer_name in self.accumulated_grads:
                # Average over batches
                saliency = self.accumulated_grads[layer_name] / max(self.grad_count, 1)
                saliency_dict[layer_name] = saliency.cpu()

                print(f"  {layer_name}: mean={saliency.mean().item():.6e}, "
                      f"std={saliency.std().item():.6e}, "
                      f"max={saliency.max().item():.6e}")

        return saliency_dict


class RegrowthAgent(nn.Module):
    """LSTM-based controller for allocation decisions (same as before)"""

    def __init__(self, action_dim, hidden_size, num_layers, context_dim, device='cuda'):
        super(RegrowthAgent, self).__init__()
        self.DEVICE = device
        self.num_layers = num_layers
        self.nhid = hidden_size
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.input_dim = action_dim + context_dim

        self.lstm = nn.LSTMCell(self.input_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, action_dim)
        self.hidden = self.init_hidden()

    def forward(self, prev_logits, context_vec):
        if prev_logits.dim() == 1:
            prev_logits = prev_logits.unsqueeze(0)
        if context_vec.dim() == 1:
            context_vec = context_vec.unsqueeze(0)

        lstm_input = torch.cat([prev_logits, context_vec], dim=-1)
        h_t, c_t = self.lstm(lstm_input, self.hidden)
        self.hidden = (h_t, c_t)
        logits = self.decoder(h_t)
        return logits

    def init_hidden(self):
        h_t = torch.zeros(1, self.nhid, dtype=torch.float, device=self.DEVICE)
        c_t = torch.zeros(1, self.nhid, dtype=torch.float, device=self.DEVICE)
        return (h_t, c_t)


class SaliencyBasedRegrowth:
    """
    Core regrowth logic using saliency scores.
    
    RigL-inspired approach:
    1. Rank pruned parameters by saliency (gradient magnitude)
    2. Select top-K with highest saliency
    3. Update mask directly (no weight copying)
    4. Optionally reinitialize regrown weights
    """

    @staticmethod
    @torch.no_grad()
    def apply_regrowth(model, layer_name, saliency_tensor, num_weights,
                       init_strategy='zero', device='cuda'):
        """
        Apply saliency-based regrowth to a single layer.
        
        RigL-style implementation:
        - Select top-K pruned parameters by saliency
        - Set mask[top_k] = 1.0
        - Initialize regrown weights
        
        Args:
            model: Neural network model
            layer_name: Name of layer to regrow
            saliency_tensor: Saliency scores (same shape as layer weights)
            num_weights: Number of weights to regrow
            init_strategy: 'zero', 'kaiming', 'xavier', or 'magnitude' (keep existing)
            device: Device
        
        Returns:
            actual_regrown: Number of weights actually regrown
            regrown_indices: List of regrown position tuples
        """
        module_dict = dict(model.named_modules())
        module = module_dict.get(layer_name)

        if module is None or not hasattr(module, 'weight_mask'):
            return 0, []

        current_mask = module.weight_mask
        saliency = saliency_tensor.to(device)

        # Find pruned positions (mask == 0)
        pruned_positions = (current_mask == 0)

        if not pruned_positions.any():
            return 0, []

        # RigL approach: mask out active weights, rank pruned by saliency
        saliency_masked = saliency.clone()
        saliency_masked[~pruned_positions] = -float('inf')  # Exclude active weights

        # Flatten and get top-K
        flat_saliency = saliency_masked.flatten()
        k = min(num_weights, (flat_saliency > -float('inf')).sum().item())

        if k == 0:
            return 0, []

        # Get top-K indices by saliency
        _, top_k_flat_indices = torch.topk(flat_saliency, k=k)

        # Convert flat indices to multi-dimensional
        shape = saliency.shape
        regrown_indices = []

        for flat_idx in top_k_flat_indices:
            multi_idx = np.unravel_index(flat_idx.cpu().item(), shape)
            regrown_indices.append(multi_idx)

            # Update mask (RigL-style)
            current_mask[multi_idx] = 1.0

            # Initialize weight
            if hasattr(module, 'weight_orig'):
                weight_param = module.weight_orig
            else:
                weight_param = module.weight

            if init_strategy == 'zero':
                weight_param.data[multi_idx] = 0.0
            elif init_strategy == 'kaiming':
                # Kaiming initialization (He et al.)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight_param)
                bound = np.sqrt(6.0 / fan_in)
                weight_param.data[multi_idx] = torch.empty(1).uniform_(-bound, bound).item()
            elif init_strategy == 'xavier':
                # Xavier/Glorot initialization
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight_param)
                bound = np.sqrt(6.0 / (fan_in + fan_out))
                weight_param.data[multi_idx] = torch.empty(1).uniform_(-bound, bound).item()
            elif init_strategy == 'magnitude':
                # Keep existing magnitude (don't modify weight value)
                pass

        return len(regrown_indices), regrown_indices


class RegrowthPolicyGradient:
    """
    RL Policy Gradient with Saliency-Based Regrowth
    
    Architecture:
    1. LSTM controller samples allocation per layer (RL)
    2. Saliency computer ranks parameters by importance (Gradient-based)
    3. Regrowth applies top-K selection (RigL-style)
    4. Finetuning recovers accuracy (AdamW)
    """

    def __init__(self, config, model_pretrained, model_99,
                 target_layers, train_loader, test_loader, device):
        # Training hyperparameters
        self.NUM_EPOCHS = config['num_epochs']
        self.ALPHA = config['learning_rate']
        self.BATCH_SIZE = config['batch_size']
        self.HIDDEN_SIZE = config['hidden_size']
        self.BETA = config['entropy_coef']
        self.REWARD_TEMPERATURE = config.get('reward_temperature', 0.01)

        self.DEVICE = device
        self.ACTION_SPACE = config['action_space_size']
        self.NUM_STEPS = len(target_layers)
        self.CONTEXT_DIM = config.get('context_dim', 3)
        self.BASELINE_DECAY = config.get('baseline_decay', 0.9)

        # Models and data
        self.model_pretrained = model_pretrained.to(device)
        self.model_99 = model_99.to(device)
        self.target_layers = target_layers
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Regrowth configuration
        self.target_regrow = config['target_regrow']
        self.layer_capacities = config['layer_capacities']
        self.total_capacity = max(sum(self.layer_capacities), 1)
        self.init_strategy = config.get('init_strategy', 'zero')
        self.saliency_max_batches = config.get('saliency_max_batches', 50)

        # Saliency computer
        self.saliency_computer = SaliencyComputer(
            model=self.model_pretrained,
            criterion=nn.CrossEntropyLoss(),
            device=self.DEVICE
        )

        # Compute initial saliency scores (will be reused during training)
        print("\n" + "=" * 70)
        print("Computing Initial Saliency Scores")
        print("=" * 70)
        self.saliency_dict = self.saliency_computer.compute_saliency_scores(
            data_loader=self.train_loader,
            target_layers=self.target_layers,
            max_batches=self.saliency_max_batches,
            num_classes=10
        )
        print("=" * 70 + "\n")

        # Controller (LSTM-based)
        self.agent = RegrowthAgent(
            action_dim=self.ACTION_SPACE,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_STEPS,
            context_dim=self.CONTEXT_DIM,
            device=self.DEVICE
        ).to(self.DEVICE)

        # Optimizer
        self.adam = optim.Adam(params=self.agent.parameters(), lr=self.ALPHA)

        # Reward baseline
        self.reward_baseline = None
        self.total_rewards = deque([], maxlen=100)

        # Model name for checkpointing
        self._model_name = config.get('model_name')

        # Checkpoint settings
        self.checkpoint_dir = config.get('checkpoint_dir', './rl_saliency_checkpoints')
        self.save_freq = config.get('save_freq', 1)

        # Entropy schedule
        self.use_entropy_schedule = config.get('use_entropy_schedule', True)
        self.start_beta = config.get('start_beta', 0.4)
        self.end_beta = config.get('end_beta', 0.004)
        self.decay_fraction = config.get('decay_fraction', 0.4)

        # Use target_layers in original order (no SSIM ranking)
        self.layer_priority = []
        for idx, layer_name in enumerate(self.target_layers):
            self.layer_priority.append((layer_name, idx))

        print(f"\nLayer regrowth order (following target_layers):")
        for layer_name, orig_idx in self.layer_priority:
            capacity = int(self.layer_capacities[orig_idx])
            print(f"  {orig_idx + 1}. {layer_name}: capacity={capacity}")

    def get_entropy_coef(self, epoch):
        """Get entropy coefficient with decay schedule"""
        if not self.use_entropy_schedule:
            return self.BETA

        decay_epochs = self.NUM_EPOCHS * self.decay_fraction

        if epoch < decay_epochs:
            beta = self.start_beta - (self.start_beta - self.end_beta) * (epoch / decay_epochs)
        else:
            beta = self.end_beta

        return beta

    def _create_model_copy(self, source_model):
        """Create proper copy of pruned model"""
        new_model = model_loader(self._model_name, self.DEVICE)
        prune_weights_reparam(new_model)
        new_model.load_state_dict(source_model.state_dict())
        return new_model

    def solve_environment(self, resume_from=None):
        """Main RL training loop"""
        # Record start time
        solve_start_time = time.time()

        print(f"\n{'=' * 70}")
        print("RL Training with Saliency-Based Regrowth")
        print(f"{'=' * 70}")
        print(f"Configuration:")
        print(f"  Epochs: {self.NUM_EPOCHS}")
        print(f"  Learning rate: {self.ALPHA}")
        print(f"  Entropy coef: {self.BETA}")
        print(f"  Init strategy: {self.init_strategy}")
        print(f"  Saliency method: Gradient magnitude (RigL-style)")
        print(f"{'=' * 70}\n")

        best_reward = float('-inf')
        best_allocation = None
        best_regrow_indices = None
        start_epoch = 0

        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming from: {resume_from}")
            checkpoint = torch.load(resume_from)
            self.agent.load_state_dict(checkpoint['agent_state_dict'])
            self.adam.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_reward = checkpoint['best_reward']
            best_allocation = checkpoint['best_allocation']
            best_regrow_indices = checkpoint['best_regrow_indices']
            if 'reward_baseline' in checkpoint:
                self.reward_baseline = checkpoint['reward_baseline']
            print(f"  Resumed from epoch {start_epoch}, best reward: {best_reward:.4f}\n")

        epoch = start_epoch
        while epoch < self.NUM_EPOCHS:
            # Play episode (pass solve_start_time for relative timing)
            (episode_weighted_log_prob,
             episode_logits,
             reward,
             allocation,
             sparsity,
             regrow_indices) = self.play_episode(solve_start_time)

            # Track best
            if reward > best_reward:
                best_reward = reward
                best_allocation = allocation
                best_regrow_indices = copy.deepcopy(regrow_indices)
                self._save_best_allocation(epoch, best_reward, best_allocation, best_regrow_indices)

            # Update controller
            current_beta = self.get_entropy_coef(epoch)
            loss, entropy = self.calculate_loss(episode_logits, episode_weighted_log_prob, beta=current_beta)

            self.adam.zero_grad()
            loss.backward()
            self.adam.step()

            # Logging
            pg_norm = torch.norm(episode_weighted_log_prob).item()

            run.log({
                "epoch": epoch + 1,
                "epoch_loss": loss.item(),
                "epoch_entropy": entropy.item(),
                "epoch_pg_norm": pg_norm,
                "current_beta": current_beta,
                "reward": reward,
                "acc": reward * 100.0,
                "reward_baseline": self.reward_baseline,
                "sparsity": sparsity,
            })

            print(f"Epoch {epoch + 1:3d}/{self.NUM_EPOCHS} | "
                  f"Reward: {reward:.4f} ({reward * 100:.2f}%) | "
                  f"Best: {best_reward:.4f} ({best_reward * 100:.2f}%) | "
                  f"Loss: {loss.item():.4f} | "
                  f"Entropy: {entropy:.4f} | "
                  f"Beta: {current_beta:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.save_freq == 0:
                self._save_checkpoint(epoch, reward, allocation, regrow_indices)

            epoch += 1

        print(f"\n{'=' * 70}")
        print(f"Training Completed!")
        print(f"Best Reward: {best_reward:.4f} ({best_reward * 100:.2f}%)")
        print(f"{'=' * 70}\n")

        return best_allocation, best_reward, best_regrow_indices

    def play_episode(self, solve_start_time):
        """
        Play one episode with saliency-based regrowth.
        
        Process:
        1. LSTM samples allocation per layer
        2. Apply saliency-based regrowth (top-K by gradient)
        3. Mini finetune
        4. Evaluate reward
        """
        # Episode start time
        episode_start_time = time.time()

        # Reset LSTM
        self.agent.hidden = self.agent.init_hidden()

        # Sample allocation
        prev_logits = torch.zeros(1, self.ACTION_SPACE, device=self.DEVICE)

        max_ratio = 1.0
        step_size = max_ratio / (self.ACTION_SPACE - 1)
        ratio_options = torch.arange(self.ACTION_SPACE, device=self.DEVICE, dtype=torch.float) * step_size

        total_budget = int(self.target_regrow)
        remaining_budget = total_budget
        masked_logits_list = []
        log_prob_list = []
        selected_counts = []
        priority_layer_names = []

        # Sample actions for each layer (in original target_layers order)
        for priority_idx, (layer_name, orig_idx) in enumerate(self.layer_priority):
            layer_capacity = int(self.layer_capacities[orig_idx])
            layer_position = priority_idx / max(self.NUM_STEPS - 1, 1) if self.NUM_STEPS > 1 else 0.0
            capacity_fraction = layer_capacity / self.total_capacity if self.total_capacity > 0 else 0.0
            remaining_fraction = remaining_budget / total_budget if total_budget > 0 else 0.0

            context_vec = torch.tensor(
                [layer_position, capacity_fraction, remaining_fraction],
                dtype=torch.float,
                device=self.DEVICE
            ).unsqueeze(0)

            logits_layer = self.agent(prev_logits, context_vec).squeeze(0)

            effective_max = min(layer_capacity, remaining_budget)
            counts_options = torch.round(ratio_options * effective_max).to(dtype=torch.long)
            feasible = (counts_options <= remaining_budget)

            if not torch.any(feasible):
                feasible[0] = True  # Allow zero allocation

            masked_logits = torch.where(
                feasible,
                logits_layer,
                torch.full_like(logits_layer, fill_value=-1e9)
            )

            probs = F.softmax(masked_logits, dim=0)
            dist = Categorical(probs=probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)

            chosen_count = int(counts_options[action_idx].item())
            chosen_count = min(chosen_count, layer_capacity, remaining_budget)
            remaining_budget = max(remaining_budget - chosen_count, 0)

            selected_counts.append(chosen_count)
            priority_layer_names.append(layer_name)
            log_prob_list.append(log_prob)
            masked_logits_list.append(masked_logits)
            prev_logits = logits_layer.unsqueeze(0)

        sampling_end_time = time.time()
        time_from_solve_start = sampling_end_time - solve_start_time
        print(f"  [Timing] Time from solve_environment start: {time_from_solve_start:.3f}s")

        episode_log_probs = torch.stack(log_prob_list)

        # Build allocation
        allocation = {}
        for i, layer_name in enumerate(priority_layer_names):
            count = int(selected_counts[i])
            if count > 0:
                allocation[layer_name] = count

        # Log allocation amounts
        print(f"\n  Sampled allocation:")
        total_allocated = 0
        for i, layer_name in enumerate(priority_layer_names):
            count = selected_counts[i]
            capacity = self.layer_capacities[self.layer_priority[i][1]]
            if count > 0:
                print(f"    {layer_name}: {count} weights (capacity: {capacity})")
            total_allocated += count
        print(f"  Total allocated: {total_allocated}/{total_budget} ({100 * total_allocated / total_budget:.1f}%)")

        # Create model copy and apply saliency-based regrowth
        model_copy = self._create_model_copy(self.model_99)
        model_copy.eval()

        actual_regrown = {}
        regrow_indices = {}

        for layer_name in priority_layer_names:
            if layer_name in allocation:
                num_weights = allocation[layer_name]
                saliency_tensor = self.saliency_dict.get(layer_name)

                if saliency_tensor is not None:
                    actual, indices = SaliencyBasedRegrowth.apply_regrowth(
                        model=model_copy,
                        layer_name=layer_name,
                        saliency_tensor=saliency_tensor,
                        num_weights=num_weights,
                        init_strategy=self.init_strategy,
                        device=self.DEVICE
                    )
                    actual_regrown[layer_name] = actual
                    regrow_indices[layer_name] = indices

        # # Log actual regrowth results
        # print(f"\n  Actual regrowth applied:")
        # total_regrown = 0
        # for layer_name in priority_layer_names:
        #     if layer_name in actual_regrown:
        #         actual = actual_regrown[layer_name]
        #         requested = allocation[layer_name]
        #         print(f"    {layer_name}: {actual} weights (requested: {requested})")
        #         total_regrown += actual
        # print(f"  Total regrown: {total_regrown}")

        # Mini finetune
        self.mini_finetune(model_copy, epochs=50)

        # Evaluate
        # may change and see how to happen
        accuracy = self.evaluate_model(model_copy, full_eval=True)
        sparsity, _, _ = self.calculate_sparsity(model_copy)

        # Compute reward
        reward = accuracy / 100.0

        # Update baseline and compute advantage
        if self.reward_baseline is None:
            self.reward_baseline = reward

        baseline = self.reward_baseline
        temperature = max(self.REWARD_TEMPERATURE, 1e-6)
        centered_reward = reward - baseline
        normalized_advantage = centered_reward / temperature
        normalized_advantage = float(np.clip(normalized_advantage, -100.0, 100.0))

        # Log metrics
        run.log({
            "acc": accuracy,
            "reward": reward,
            "reward_baseline": baseline,
            "centered_reward": centered_reward,
            "normalized_advantage": normalized_advantage,
            "sparsity": sparsity,
        })

        self.reward_baseline = self.BASELINE_DECAY * self.reward_baseline + (1 - self.BASELINE_DECAY) * reward

        advantage_tensor = torch.tensor(normalized_advantage, device=self.DEVICE, dtype=torch.float)
        episode_weighted_log_probs = episode_log_probs * advantage_tensor
        sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)

        episode_logits = torch.stack(masked_logits_list) if masked_logits_list else None

        episode_end_time = time.time()
        total_episode_duration = episode_end_time - episode_start_time
        print(f"  [Timing] Total episode duration: {total_episode_duration:.3f}s")

        # Log timing metrics
        run.log({
            "timing/time_from_solve_start": time_from_solve_start,
            "timing/total_episode": total_episode_duration,
        })

        return sum_weighted_log_probs, episode_logits, reward, allocation, sparsity, regrow_indices

    def calculate_loss(self, epoch_logits, weighted_log_probs, beta=None):
        """Calculate policy loss + entropy bonus"""
        if beta is None:
            beta = self.BETA

        policy_loss = -torch.mean(weighted_log_probs)

        if epoch_logits.numel() == 0:
            entropy = torch.tensor(0.0, device=self.DEVICE)
        else:
            p = F.softmax(epoch_logits, dim=1)
            log_p = F.log_softmax(epoch_logits, dim=1)
            entropy = -torch.mean(torch.sum(p * log_p, dim=1), dim=0)

        entropy_bonus = -beta * entropy
        return policy_loss + entropy_bonus, entropy

    def calculate_sparsity(self, model):
        """Calculate model sparsity"""
        total_params = 0
        pruned_params = 0

        for name, module in model.named_modules():
            if hasattr(module, 'weight_mask'):
                mask = module.weight_mask
                total_params += mask.numel()
                pruned_params += (mask == 0).sum().item()

        sparsity = 100.0 * pruned_params / total_params if total_params > 0 else 0.0
        return sparsity, total_params, pruned_params

    def mini_finetune(self, model, epochs=50, lr=0.0003):
        """Mini finetuning with AdamW"""
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        best_accuracy = 0.0
        best_model_state = None

        for epoch in range(epochs):
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # Quick eval
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                    if batch_idx >= 20:
                        break
                    inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            accuracy = 100.0 * correct / total
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = copy.deepcopy(model.state_dict())

            model.train()

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        model.eval()

    def evaluate_model(self, model, full_eval=False):
        """Evaluate model accuracy"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                if not full_eval and batch_idx >= 20:
                    break
                inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy

    def _save_checkpoint(self, epoch, best_reward, best_allocation, best_regrow_indices):
        """Save training checkpoint"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # change file dir
        # TODO
        checkpoint_path = os.path.join(self.checkpoint_dir,
                                       f'{model_name}/{method}/saliency_rl_checkpoint_epoch_{epoch + 1}.pth')

        checkpoint = {
            'epoch': epoch,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.adam.state_dict(),
            'best_reward': best_reward,
            'best_allocation': best_allocation,
            'best_regrow_indices': best_regrow_indices,
            'reward_baseline': self.reward_baseline,
            'total_rewards': list(self.total_rewards),
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ Checkpoint saved: {checkpoint_path}")

    def _save_best_allocation(self, epoch, best_reward, best_allocation, best_regrow_indices):
        """Save best allocation"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # TODO
        best_path = os.path.join(self.checkpoint_dir, f'{model_name}/{method}/best_saliency_allocation.pth')

        best_data = {
            'epoch': epoch,
            'reward': best_reward,
            'accuracy': best_reward * 100.0,
            'allocation': best_allocation,
            'regrow_indices': best_regrow_indices,
            'timestamp': time.time(),
        }

        torch.save(best_data, best_path)
        print(f"  ✓ New best! Reward: {best_reward:.4f} ({best_reward * 100:.2f}%) at epoch {epoch + 1}")


def full_finetune(model, train_loader, test_loader, device,
                  epochs=1500, lr=0.0003, save_path=None, patience=30):
    """Full finetuning with early stopping"""
    print(f"\n{'=' * 70}")
    print("Final Finetuning")
    print(f"{'=' * 70}\n")

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_accuracy = 100.0 * correct / total

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            if save_path:
                torch.save(best_model_state, save_path)
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Test Acc: {test_accuracy:.2f}% | Best: {best_accuracy:.2f}%")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(f"\nBest accuracy: {best_accuracy:.2f}% (epoch {best_epoch})")
    return best_accuracy, best_model_state


def main():
    parser = argparse.ArgumentParser(description='RL with Saliency-Based Regrowth')
    parser.add_argument('--m_name', type=str, default='vgg16')
    parser.add_argument('--data_dir', type=str, default='./data')

    # RL hyperparameters
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--entropy_coef', type=float, default=0.5)
    parser.add_argument('--reward_temperature', type=float, default=0.005)
    parser.add_argument('--start_beta', type=float, default=0.40)
    parser.add_argument('--end_beta', type=float, default=0.04)
    parser.add_argument('--decay_fraction', type=float, default=0.4)

    # Action space
    parser.add_argument('--action_space_size', type=int, default=11)

    # Regrowth parameters
    parser.add_argument('--regrow_step', type=float, default=0.035)
    parser.add_argument('--init_strategy', type=str, default='zero',
                        choices=['zero', 'kaiming', 'xavier', 'magnitude'])
    parser.add_argument('--saliency_max_batches', type=int, default=50)

    # Finetuning
    parser.add_argument('--finetune_epochs', type=int, default=400)
    parser.add_argument('--finetune_lr', type=float, default=0.0003)
    parser.add_argument('--save_dir', type=str, default='./rl_saliency_checkpoints')

    # Checkpoint
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    train_loader, val_loader, test_loader = data_loader(data_dir=args.data_dir)

    # Load models
    print("Loading models...")
    model_pretrained = model_loader(args.m_name, device)
    load_model_name(model_pretrained, f'./{args.m_name}/checkpoint', args.m_name)

    model_99 = model_loader(args.m_name, device)
    prune_weights_reparam(model_99)
    # Note: model_pretrained should stay unpruned for saliency computation

    if args.m_name == 'resnet20':
        checkpoint_99 = torch.load(
            f'./{args.m_name}/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.9903.pth')
        target_layers = ["layer3.0.conv2", "layer3.1.conv1", "layer3.1.conv2",
                         "layer3.2.conv1"]

    elif args.m_name == 'vgg16':
        checkpoint_99 = torch.load(
            f'./{args.m_name}/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.995.pth')
        target_layers = ["features.10", "features.14", "features.17", "features.20", "features.24"]
    elif args.m_name == 'alexnet':
        checkpoint_99 = torch.load(f'./{args.m_name}/ckpt_after_prune/pruned_finetuned_mask_0.99.pth')
        target_layers = ['features.3', 'features.6', 'features.8', 'features.10', 'classifier.1']
    else:
        raise ValueError(f"Unknown model: {args.m_name}")

    model_99.load_state_dict(checkpoint_99)

    # Get layer capacities (regrowable positions)
    # Capacity = positions that are currently pruned in model_99
    # Since model_pretrained is unpruned, all positions in it are available
    layer_capacities = []
    for layer_name in target_layers:
        module_99 = dict(model_99.named_modules())[layer_name]

        if hasattr(module_99, 'weight_mask'):
            mask_99 = module_99.weight_mask
            # Capacity = currently pruned positions (mask == 0)
            capacity = (mask_99 == 0).sum().item()
            layer_capacities.append(capacity)

    # Calculate target regrowth
    total_weights, _, _ = count_pruned_params(model_99)
    target_regrow = int(total_weights * args.regrow_step)
    target_regrow = min(target_regrow, sum(layer_capacities))

    print(f"\nRegrowth configuration:")
    print(f"  Total weights: {total_weights}")
    print(f"  Target regrowth: {target_regrow}")
    print(f"  Total capacity: {sum(layer_capacities)}")
    print(f"  Init strategy: {args.init_strategy}")

    # Setup config
    config = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'hidden_size': args.hidden_size,
        'entropy_coef': args.entropy_coef,
        'action_space_size': args.action_space_size,
        'target_regrow': target_regrow,
        'layer_capacities': layer_capacities,
        'model_name': args.m_name,
        'reward_temperature': args.reward_temperature,
        'checkpoint_dir': args.save_dir,
        'save_freq': args.save_freq,
        'start_beta': args.start_beta,
        'end_beta': args.end_beta,
        'decay_fraction': args.decay_fraction,
        'init_strategy': args.init_strategy,
        'saliency_max_batches': args.saliency_max_batches,
    }

    # Initialize Policy Gradient
    pg = RegrowthPolicyGradient(
        config=config,
        model_pretrained=model_pretrained,
        model_99=model_99,
        target_layers=target_layers,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device
    )

    # Evaluate before
    before_accuracy = pg.evaluate_model(model_99, full_eval=True)
    before_sparsity, _, _ = pg.calculate_sparsity(model_99)
    print(f"\nBefore regrowth:")
    print(f"  Accuracy: {before_accuracy:.2f}%")
    print(f"  Sparsity: {before_sparsity:.2f}%")

    # Run RL training
    best_allocation, best_reward, best_regrow_indices = pg.solve_environment(resume_from=args.resume)

    # Apply best allocation
    print("\nApplying best allocation...")
    for layer_name, num_weights in best_allocation.items():
        if num_weights > 0:
            saliency_tensor = pg.saliency_dict.get(layer_name)
            if saliency_tensor is not None:
                SaliencyBasedRegrowth.apply_regrowth(
                    model=model_99,
                    layer_name=layer_name,
                    saliency_tensor=saliency_tensor,
                    num_weights=num_weights,
                    init_strategy=args.init_strategy,
                    device=device
                )

    # Evaluate after regrowth
    after_accuracy = pg.evaluate_model(model_99, full_eval=True)
    after_sparsity, _, _ = pg.calculate_sparsity(model_99)
    print(f"\nAfter regrowth:")
    print(f"  Accuracy: {after_accuracy:.2f}%")
    print(f"  Sparsity: {after_sparsity:.2f}%")
    print(f"  Improvement: {after_accuracy - before_accuracy:+.2f}%")

    # Final finetuning
    # file join

    final_save_path = os.path.join(args.save_dir, f'{args.m_name}/{method}/final_model.pth')
    final_accuracy, final_state = full_finetune(
        model=model_99,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        save_path=final_save_path,
        patience=50
    )

    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Before: {before_accuracy:.2f}%")
    print(f"After regrowth: {after_accuracy:.2f}%")
    print(f"After finetuning: {final_accuracy:.2f}%")
    print(f"Total improvement: {final_accuracy - before_accuracy:+.2f}%")
    print(f"{'=' * 70}\n")

    run.finish()


if __name__ == '__main__':
    main()
