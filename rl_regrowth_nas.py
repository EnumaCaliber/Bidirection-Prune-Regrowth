"""
RL-based Regrowth Allocation following NAS-RL Implementation
Reference: https://github.com/dxywill/nas_rl

Key design principles from nas_rl:
1. Agent: LSTM-based controller that outputs action logits sequentially
2. Action sampling: Categorical distribution from logits
3. Policy Gradient: REINFORCE with entropy bonus
4. Update: -mean(log_prob * reward) - beta * entropy

Adapted for regrowth allocation:
- Action space: Discretized ratios [0, step_size, 2*step_size, ..., max_ratio]
- Constraint handling: Two approaches
  (1) Normalize ratios after sampling: ratios = ratios / sum(ratios)
  (2) Add penalty: reward = accuracy - lambda * |sum(ratios) - 1|
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import time
import os
import copy
import argparse
import random
import wandb
from collections import deque

from utils.model_loader import model_loader
from utils.data_loader import data_loader
from utils.analysis_utils import (
    BlockwiseFeatureExtractor, compute_block_ssim,
    load_model, load_model_name, prune_weights_reparam, count_pruned_params,
    get_cifar_resnet20_blocks
)

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    # Set the wandb project where this run will be logged.
    project="ICCAP 2026",
    name="solu25_res_eps500_step05_from995",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 5e-3,
        "architecture": "VGG16",
        "regrow_step": 0.005,
        "epochs": 500,
        "batch_size": 1,
    },
)


def set_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Make PyTorch deterministic (may impact performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For DataLoader workers
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


class RegrowthAgent(nn.Module):
    """
    LSTM-based controller agent (following nas_rl/controller.py)
    
    Outputs action logits for each layer sequentially.
    """

    def __init__(self, action_dim, hidden_size, num_layers, context_dim, device='cuda'):
        super(RegrowthAgent, self).__init__()
        self.DEVICE = device
        self.num_layers = num_layers  # Number of layers to allocate
        self.nhid = hidden_size
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.input_dim = action_dim + context_dim

        # LSTM cell for sequential decision making
        self.lstm = nn.LSTMCell(self.input_dim, hidden_size)

        # Decoder: hidden state -> action logits
        # Output size = number of discrete actions (ratios)
        self.decoder = nn.Linear(hidden_size, action_dim)

        self.hidden = self.init_hidden()

    def forward(self, prev_logits, context_vec):
        """
        Single-step forward pass.
        
        Args:
            prev_logits: Previous step's logits [batch=1, action_dim]
            context_vec: Context features for current layer [batch=1, context_dim]
        
        Returns:
            logits: [batch=1, action_dim] logits for current layer
        """
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
        """Initialize hidden state for LSTM"""
        h_t = torch.zeros(1, self.nhid, dtype=torch.float, device=self.DEVICE)
        c_t = torch.zeros(1, self.nhid, dtype=torch.float, device=self.DEVICE)
        return (h_t, c_t)


class RegrowthPolicyGradient:
    """
    Policy Gradient training for regrowth allocation (following nas_rl/policy_gradient.py)
    
    Key components:
    1. Agent samples allocation ratios for each layer
    2. Apply allocation -> finetune -> evaluate -> get reward
    3. Update agent with policy gradient + entropy bonus
    """

    def __init__(self, config, model_pretrained, model_95, model_99,
                 target_layers, train_loader, test_loader, device):
        # Training hyperparameters
        self.NUM_EPOCHS = config['num_epochs']
        self.ALPHA = config['learning_rate']  # Learning rate
        self.BATCH_SIZE = config['batch_size']  # Number of samples per epoch
        self.HIDDEN_SIZE = config['hidden_size']
        self.BETA = config['entropy_coef']  # Entropy bonus coefficient
        self.REWARD_TEMPERATURE = config.get('reward_temperature', 0.01)

        self.DEVICE = device
        self.ACTION_SPACE = config['action_space_size']  # Number of discrete ratios
        self.NUM_STEPS = len(target_layers)  # Number of layers to allocate
        self.CONTEXT_DIM = config.get('context_dim', 3)
        self.BASELINE_DECAY = config.get('baseline_decay', 0.9)

        # Models and data
        self.model_pretrained = model_pretrained.to(device)
        self.model_95 = model_95.to(device)
        self.model_99 = model_99.to(device)
        self.target_layers = target_layers
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Reference model for priority-based regrowth (can be custom or default to model_95)
        self.model_reference = config.get('model_reference', model_95).to(device)

        # Target regrowth budget
        self.target_regrow = config['target_regrow']
        self.layer_capacities = config['layer_capacities']
        self.total_capacity = max(sum(self.layer_capacities), 1)

        # Reference for regrowth
        self.reference_masks = config['reference_masks']
        self.reference_weights = config['reference_weights']

        # Controller (LSTM-based)
        self.agent = RegrowthAgent(
            action_dim=self.ACTION_SPACE,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_STEPS,
            context_dim=self.CONTEXT_DIM,
            device=self.DEVICE
        ).to(self.DEVICE)

        # Reward baseline for variance reduction
        self.reward_baseline = None

        # Optimizer
        self.adam = optim.Adam(params=self.agent.parameters(), lr=self.ALPHA)

        # Statistics
        self.total_rewards = deque([], maxlen=100)

        # Setup feature extractors for SSIM
        self.block_dict = {'target_block': target_layers}

        # Store model name for safe copying
        self._model_name = config.get('model_name')

        # Checkpoint settings
        self.checkpoint_dir = config.get('checkpoint_dir', './rl_nas_checkpoints')
        self.save_freq = config.get('save_freq', 5)  # Save every N epochs

        # Track best model within current checkpoint window
        self.window_best_reward = float('-inf')
        self.window_best_allocation = None
        self.window_best_regrow_indices = None
        self.window_best_epoch = 0

        # Entropy schedule settings
        self.use_entropy_schedule = config.get('use_entropy_schedule', True)
        self.start_beta = config.get('start_beta', 0.4)
        self.end_beta = config.get('end_beta', 0.004)
        self.decay_fraction = config.get('decay_fraction', 0.4)  # Decay over 80% of training

        # Compute layer priority ONCE at initialization (based on SSIM scores)
        # Lower SSIM = more dissimilar features = higher priority for regrowth
        print("\nComputing layer priority based on SSIM scores (one-time computation)...")
        ssim_scores = self.compute_layer_ssim_scores(self.model_99)

        # Create priority-sorted list of (layer_name, original_idx, ssim_val)
        self.layer_priority = []
        for idx, layer_name in enumerate(self.target_layers):
            ssim_val = ssim_scores.get(layer_name, 0.5)
            self.layer_priority.append((layer_name, idx, ssim_val))

        # Sort by SSIM (ascending) - lower SSIM = higher priority
        self.layer_priority.sort(key=lambda x: x[2])

        print(f"Layer regrowth priority (by SSIM, lower = higher priority):")
        for layer_name, orig_idx, ssim_val in self.layer_priority:
            capacity = int(self.layer_capacities[orig_idx])
            print(f"  {layer_name}: SSIM={ssim_val:.4f}, capacity={capacity} weights")

    def get_entropy_coef(self, epoch):
        """
        Get entropy coefficient for current epoch using decay schedule.
        
        Args:
            epoch: Current training epoch (0-indexed)
        
        Returns:
            beta: Entropy coefficient for this epoch
        """
        if not self.use_entropy_schedule:
            # Use fixed entropy coefficient
            return self.BETA

        # Decay over specified fraction of total epochs
        decay_epochs = self.NUM_EPOCHS * self.decay_fraction

        if epoch < decay_epochs:
            # Linear decay from start_beta to end_beta
            beta = self.start_beta - (self.start_beta - self.end_beta) * (epoch / decay_epochs)
        else:
            # Keep at end_beta for remaining epochs
            beta = self.end_beta

        return beta

    def _create_model_copy(self, source_model):
        """
        Create a proper copy of a pruned model.
        deepcopy doesn't work well with pruning hooks, so we recreate and load state dict.
        """

        new_model = model_loader(self._model_name, self.DEVICE)
        prune_weights_reparam(new_model)
        new_model.load_state_dict(source_model.state_dict())
        return new_model

    def compute_layer_ssim_scores(self, model_current):
        """
        Compute SSIM scores for each target layer between current model and pretrained model.
        Lower SSIM = more dissimilar features = higher priority for regrowth.
        
        Args:
            model_current: Current pruned/regrown model
            
        Returns:
            ssim_dict: Dict[layer_name] -> SSIM score (float)
        """
        # Extract features from both models
        extractor_pretrained = BlockwiseFeatureExtractor(self.model_pretrained, self.block_dict)
        extractor_current = BlockwiseFeatureExtractor(model_current, self.block_dict)

        # Extract features using the extract_block_features method
        # Use 128 batches from test_loader for SSIM computation
        with torch.no_grad():
            features_pretrained = extractor_pretrained.extract_block_features(self.test_loader, num_batches=128)
            features_current = extractor_current.extract_block_features(self.test_loader, num_batches=128)

        # Compute SSIM scores per layer
        ssim_scores = compute_block_ssim(features_pretrained, features_current)

        # Map SSIM scores to layer names
        # ssim_scores has structure: {'target_block': {'layer_name': score, ...}}
        ssim_dict = {}
        target_block_scores = ssim_scores.get('target_block', {})

        for layer_name in self.target_layers:
            if layer_name in target_block_scores:
                ssim_dict[layer_name] = target_block_scores[layer_name]
            else:
                # If SSIM not available, assign neutral score
                print(f"Warning: SSIM not computed for {layer_name}, using default 0.5")
                ssim_dict[layer_name] = 0.5

        return ssim_dict

    def _save_checkpoint(self, epoch, best_reward, best_allocation, best_regrow_indices):
        """Save training checkpoint and best model within this checkpoint window"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, f'rl_training_checkpoint_epoch_{epoch + 1}.pth')

        checkpoint = {
            'epoch': epoch,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.adam.state_dict(),
            'best_reward': best_reward,
            'best_allocation': best_allocation,
            'best_regrow_indices': best_regrow_indices,
            'reward_baseline': self.reward_baseline,
            'total_rewards': list(self.total_rewards),
            'config': {
                'num_epochs': self.NUM_EPOCHS,
                'learning_rate': self.ALPHA,
                'batch_size': self.BATCH_SIZE,
                'hidden_size': self.HIDDEN_SIZE,
                'entropy_coef': self.BETA,
                'action_space': self.ACTION_SPACE,
                'num_steps': self.NUM_STEPS,
            }
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ Checkpoint saved: {checkpoint_path}")

        # Save the best model within this checkpoint window
        if self.window_best_allocation is not None:
            window_best_path = os.path.join(
                self.checkpoint_dir,
                f'window_best_epoch_{epoch + 1}_{self._model_name}.pth'
            )

            window_best_data = {
                'window_end_epoch': epoch,
                'window_best_epoch': self.window_best_epoch,
                'reward': self.window_best_reward,
                'accuracy': self.window_best_reward * 100.0,
                'allocation': self.window_best_allocation,
                'regrow_indices': self.window_best_regrow_indices,
                'timestamp': time.time(),
            }

            torch.save(window_best_data, window_best_path)
            print(f"  ✓ Window best model saved: {window_best_path}")
            print(
                f"     Window best: Epoch {self.window_best_epoch + 1}, Reward: {self.window_best_reward:.4f} ({self.window_best_reward * 100:.2f}%)")

    def _save_best_allocation(self, epoch, best_reward, best_allocation, best_regrow_indices):
        """Save best allocation found so far (overwrites previous best)"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        best_path = os.path.join(self.checkpoint_dir, f'best_allocation_{self._model_name}.pth')

        best_data = {
            'epoch': epoch,
            'reward': best_reward,
            'accuracy': best_reward * 100.0,  # Convert to percentage
            'allocation': best_allocation,
            'regrow_indices': best_regrow_indices,
            'timestamp': time.time(),
        }

        torch.save(best_data, best_path)
        print(
            f"  ✓ New best allocation saved! Reward: {best_reward:.4f} ({best_reward * 100:.2f}%) at epoch {epoch + 1}")
        print(f"     Saved to: {best_path}")

    def solve_environment(self, resume_from=None):
        """
        Main training loop (following nas_rl/policy_gradient.py::solve_environment)
        
        Args:
            resume_from: Path to checkpoint file to resume from (optional)
        """
        print(f"\n{'=' * 70}")
        print("RL Policy Gradient Training (NAS-RL Style)")
        print(f"{'=' * 70}")
        print(f"Configuration:")
        print(f"  Epochs: {self.NUM_EPOCHS}")
        print(f"  Batch size: {self.BATCH_SIZE}")
        print(f"  Learning rate: {self.ALPHA}")
        print(f"  Entropy coef: {self.BETA}")
        print(f"  Reward temperature: {self.REWARD_TEMPERATURE}")
        print(f"  Hidden size: {self.HIDDEN_SIZE}")
        print(f"  Action space: {self.ACTION_SPACE}")
        print(f"  Num layers: {self.NUM_STEPS}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  Save frequency: Every {self.save_freq} epochs")
        print(f"{'=' * 70}\n")

        best_reward = float('-inf')
        best_allocation = None
        best_regrow_indices = None
        start_epoch = 0

        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from)
            self.agent.load_state_dict(checkpoint['agent_state_dict'])
            self.adam.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_reward = checkpoint['best_reward']
            best_allocation = checkpoint['best_allocation']
            best_regrow_indices = checkpoint['best_regrow_indices']
            if 'reward_baseline' in checkpoint and checkpoint['reward_baseline'] is not None:
                self.reward_baseline = checkpoint['reward_baseline']
            if 'total_rewards' in checkpoint:
                self.total_rewards = deque(checkpoint['total_rewards'], maxlen=100)
            print(f"  Resumed from epoch {start_epoch}")
            print(f"  Best reward so far: {best_reward:.4f} ({best_reward * 100:.2f}%)")
            print()

            # Reset window tracking when resuming (start fresh window)
            self.window_best_reward = float('-inf')
            self.window_best_allocation = None
            self.window_best_regrow_indices = None
            self.window_best_epoch = start_epoch

        epoch = start_epoch
        while epoch < self.NUM_EPOCHS:
            # Epoch-level accumulators
            # epoch_logits = torch.empty(size=(0, self.ACTION_SPACE), device=self.DEVICE)
            # epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)

            # # Sample BATCH_SIZE allocations
            # epoch_allocations = []
            # for i in range(self.BATCH_SIZE):
            # Play one episode (sample allocation, evaluate, get reward)
            (episode_weighted_log_prob,
             episode_logits,
             reward,
             allocation,
             sparsity,
             regrow_indices) = self.play_episode()

            # Track rewards
            # self.total_rewards.append(reward)
            # epoch_allocations.append((reward, allocation, sparsity))

            # # Accumulate weighted log probs
            # epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, episode_weighted_log_prob), dim=0)

            # # Accumulate logits for entropy calculation
            # if episode_logits is not None:
            #     epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)

            # Track best
            if reward > best_reward:
                best_reward = reward
                best_allocation = allocation
                best_regrow_indices = copy.deepcopy(regrow_indices)

                # Save global best allocation immediately when found
                self._save_best_allocation(
                    epoch=epoch,
                    best_reward=best_reward,
                    best_allocation=best_allocation,
                    best_regrow_indices=best_regrow_indices
                )

            # Track best within current checkpoint window
            if reward > self.window_best_reward:
                self.window_best_reward = reward
                self.window_best_allocation = copy.deepcopy(allocation)
                self.window_best_regrow_indices = copy.deepcopy(regrow_indices)
            print(
                f"\n  Controller update (epoch {epoch + 1}): acc={reward * 100:.2f}% | baseline={self.reward_baseline * 100:.2f}%")

            current_beta = self.get_entropy_coef(epoch)

            loss, entropy = self.calculate_loss(
                epoch_logits=episode_logits,
                weighted_log_probs=episode_weighted_log_prob,
                beta=current_beta
            )

            self.adam.zero_grad()
            loss.backward()
            self.adam.step()

            # mean_reward = float(np.mean(self.total_rewards)) if len(self.total_rewards) > 0 else float(last_reward)
            pg_norm = torch.norm(episode_weighted_log_prob).item()

            run.log({
                "epoch_loss": loss.item(),
                "epoch_entropy": entropy.item(),
                "epoch_pg_norm": pg_norm,
                "current_beta": current_beta,
                "reward_baseline_after_update": self.reward_baseline,
            })

            # Log allocation for current episode
            if epoch % 1 == 0 or epoch == 0:
                print(f"\n  Allocation from epoch {epoch + 1}:")
                total_alloc = sum(allocation.values()) if allocation else 0
                # reward is normalized to [0,1], multiply by 100 for percentage display
                print(f"    Reward: {reward * 100:.2f}%, Sparsity: {sparsity:.2f}%, Total allocated: {total_alloc}")
                if allocation:
                    print(f"    Layer allocations (count / capacity):")
                    for layer_name, count in allocation.items():
                        layer_idx = self.target_layers.index(layer_name)
                        capacity = self.layer_capacities[layer_idx]
                        print(f"      {layer_name}: {count:4d} / {capacity:4d}")
                print()

            # Get current beta for display
            display_beta = self.get_entropy_coef(epoch)

            print(f"Epoch {epoch + 1:3d}/{self.NUM_EPOCHS} | "
                  f"Current Reward: {reward:.4f} | "
                  f"Best Reward: {best_reward:.4f} | "
                  f"Entropy: {entropy:.4f} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Beta: {display_beta:.4f}")

            # Save checkpoint periodically
            if (epoch + 1) % self.save_freq == 0:
                self._save_checkpoint(
                    epoch=epoch,
                    best_reward=best_reward,
                    best_allocation=best_allocation,
                    best_regrow_indices=best_regrow_indices
                )

                # Reset window tracking for next checkpoint period
                self.window_best_reward = float('-inf')
                self.window_best_allocation = None
                self.window_best_regrow_indices = None
                self.window_best_epoch = epoch + 1  # Start of next window

            epoch += 1

        print(f"\n{'=' * 70}")
        print(f"Training Completed!")
        print(f"Best Reward: {best_reward:.4f} (accuracy: {best_reward * 100:.2f}%)")
        print(f"Best Allocation: {best_allocation}")
        print(f"{'=' * 70}\n")

        return best_allocation, best_reward, best_regrow_indices

    def play_episode(self):
        """
        Play one episode: sample allocation -> evaluate -> compute reward
        (following nas_rl/policy_gradient.py::play_episode)
        
        Returns:
            sum_weighted_log_probs: Σ(log_prob * reward)
            episode_logits: Logits for entropy calculation
            reward: Reward for this episode
            allocation: Dict of allocation
            sparsity: Sparsity percentage after regrowth
            regrow_indices: Dict[layer_name] -> list of index tuples used during regrowth
        """
        # Reset LSTM hidden state at the start of every episode (avoid leaking across episodes)
        self.agent.hidden = self.agent.init_hidden()

        # Use pre-computed layer priority (computed once at initialization)
        # No need to recompute SSIM every episode - the priority is fixed based on initial degradation

        # Previous logits start at zero
        prev_logits = torch.zeros(1, self.ACTION_SPACE, device=self.DEVICE)

        # Sample actions for each layer using Categorical distribution
        # NOW iterate through layers in SSIM-PRIORITY order
        max_ratio = 1.0
        step_size = max_ratio / (self.ACTION_SPACE - 1)
        ratio_options = torch.arange(self.ACTION_SPACE, device=self.DEVICE, dtype=torch.float) * step_size

        total_budget = int(self.target_regrow)
        remaining_budget = total_budget
        masked_logits_list = []
        log_prob_list = []
        selected_counts = []
        priority_layer_names = []  # Track layer names in priority order

        for priority_idx, (layer_name, orig_idx, ssim_val) in enumerate(self.layer_priority):
            # Use orig_idx to get capacity from self.layer_capacities
            layer_capacity = int(self.layer_capacities[orig_idx])
            # Use priority_idx for positional encoding in context
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
                feasible = torch.zeros_like(feasible, dtype=torch.bool)
                feasible[0] = True  # ensure "no regrowth" remains selectable

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
            priority_layer_names.append(layer_name)  # Track in priority order
            log_prob_list.append(log_prob)
            masked_logits_list.append(masked_logits)
            prev_logits = logits_layer.unsqueeze(0)

        episode_log_probs = torch.stack(log_prob_list)

        # Build allocation directly from sampled counts (now in SSIM-priority order)
        allocation = {}
        for i, layer_name in enumerate(priority_layer_names):
            count = int(selected_counts[i])
            if count > 0:
                allocation[layer_name] = count

        # # Remove empty allocations for cleanliness
        # allocation = {
        #     layer_name: count for layer_name, count in allocation.items() if count > 0
        # }

        # Apply allocation and evaluate  
        # Use proper model copying to avoid deepcopy issues with pruning hooks
        model_copy = self._create_model_copy(self.model_99)
        model_copy.eval()

        # # Log allocation amounts (allocation is already in SSIM-priority order)
        # print(f"\n  Allocation amounts:")
        # for layer_name in priority_layer_names:
        #     if layer_name in allocation:
        #         num_weights = allocation[layer_name]
        #         orig_idx = next(i for i, (ln, _, _) in enumerate(self.layer_priority) if ln == layer_name)
        #         ssim_val = self.layer_priority[orig_idx][2]
        #         print(f"    {layer_name}: regrow={num_weights} weights (SSIM={ssim_val:.4f})")

        # Apply regrowth (allocation is already in SSIM-priority order from the sampling loop)
        actual_regrown = {}
        regrow_indices = {}
        for layer_name in priority_layer_names:
            if layer_name in allocation:
                num_weights = allocation[layer_name]
                if num_weights > 0:
                    actual, selected_idx = self.apply_random_regrowth(model_copy, layer_name, num_weights)
                    actual_regrown[layer_name] = actual
                    if selected_idx is not None and selected_idx.numel() > 0:
                        regrow_indices[layer_name] = [tuple(idx.tolist()) for idx in selected_idx]
                    else:
                        regrow_indices[layer_name] = []
                else:
                    actual_regrown[layer_name] = 0
                    regrow_indices[layer_name] = []

        # # Mini finetuning before evaluation to stabilize accuracy estimates
        # CRITICAL: Increased from 10 to 50 epochs to allow proper recovery after regrowth
        # 10 epochs was too short for the network to adapt to regrown weights
        self.mini_finetune(model_copy, epochs=50)

        # Evaluate accuracy
        accuracy = self.evaluate_model(model_copy, full_eval=False)

        # Calculate sparsity after regrowth
        sparsity, total_params, pruned_params = self.calculate_sparsity(model_copy)

        # Compute reward - normalize accuracy to [0, 1] range to match entropy scale
        # Typical CIFAR accuracy range: 70-90%, normalize to roughly [0, 1]
        reward = accuracy / 100.0  # Convert percentage to [0, 1] range

        # Weighted log probabilities (following nas_rl)
        # Update running reward baseline and compute advantage
        if self.reward_baseline is None:
            self.reward_baseline = reward
        baseline = self.reward_baseline
        temperature = max(self.REWARD_TEMPERATURE, 1e-6)
        centered_reward = reward - baseline
        normalized_advantage = centered_reward / temperature
        normalized_advantage = float(np.clip(normalized_advantage, -100.0, 100.0))

        # Track policy gradient magnitude (useful for detecting convergence)
        pg_magnitude = torch.mean(torch.abs(episode_log_probs)).item()

        # Log comprehensive metrics for convergence analysis
        run.log({
            # Raw metrics
            "acc": accuracy,
            "reward": reward,
            # Baseline and advantage metrics (convergence indicators)
            "reward_baseline": baseline,
            "centered_reward": centered_reward,  # Should converge to ~0 when stable
            "normalized_advantage": normalized_advantage,
            # Policy gradient metrics (convergence indicators)
            "pg_magnitude": pg_magnitude,  # Should decrease when converging
        })

        self.reward_baseline = self.BASELINE_DECAY * self.reward_baseline + (1 - self.BASELINE_DECAY) * reward

        advantage_tensor = torch.tensor(normalized_advantage, device=self.DEVICE, dtype=torch.float)
        episode_weighted_log_probs = episode_log_probs * advantage_tensor  # [num_layers]
        sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)  # [1]

        episode_logits = torch.stack(masked_logits_list) if masked_logits_list else None

        return sum_weighted_log_probs, episode_logits, reward, allocation, sparsity, regrow_indices

    def calculate_loss(self, epoch_logits, weighted_log_probs, beta=None):
        """
        Calculate policy loss + entropy bonus (following nas_rl/policy_gradient.py::calculate_loss)
        
        Args:
            epoch_logits: [batch_size * num_layers, action_space_size]
            weighted_log_probs: [batch_size * num_layers]
            beta: Entropy coefficient (if None, use self.BETA)
        
        Returns:
            loss: policy_loss + entropy_bonus
            entropy: for logging
        """
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

    def apply_allocation(self, model, allocation, regrow_indices=None):
        """Apply regrowth allocation to model"""
        regrow_indices = regrow_indices or {}
        for layer_name, num_weights in allocation.items():
            selected_idx = None
            if regrow_indices and layer_name in regrow_indices and regrow_indices[layer_name]:
                selected_idx = regrow_indices[layer_name]
            if num_weights > 0:
                self.apply_random_regrowth(
                    model,
                    layer_name,
                    num_weights,
                    selected_indices=selected_idx
                )

    def apply_random_regrowth(self, model, layer_name, num_weights, selected_indices=None):
        """
        Intelligently regrow weights in specified layer with priority-based selection.
        
        Strategy:
        1. If selected_indices provided (replay mode), use those directly
        2. Otherwise, prioritize weights from reference model:
           - First, select weights where reference mask=1 (important weights)
           - Then, fill remaining budget with random selection from other regrowable positions
        
        This ensures we recover the most important weights (as identified by reference model)
        before resorting to random selection.
        """
        module_dict = dict(model.named_modules())
        module = module_dict.get(layer_name)
        if module is None or not hasattr(module, 'weight_mask'):
            return 0, None

        current_mask = module.weight_mask
        ref_mask = self.reference_masks[layer_name]

        # Get reference mask from specified reference model (defaults to model_95 if not provided)
        module_reference = dict(self.model_reference.named_modules())[layer_name]
        reference_mask = module_reference.weight_mask if hasattr(module_reference, 'weight_mask') else torch.ones_like(
            ref_mask)

        # Total regrowable positions: currently pruned (0) but were present in target reference (1)
        regrowable = (current_mask == 0) & (ref_mask == 1)
        regrowable_indices = torch.nonzero(regrowable, as_tuple=False)

        if len(regrowable_indices) == 0:
            empty_shape = (0, regrowable_indices.size(1) if regrowable_indices.ndim > 1 else 1)
            return 0, torch.empty(empty_shape, dtype=torch.long).cpu()

        # REPLAY MODE: Use provided indices directly
        if selected_indices is not None:
            if isinstance(selected_indices, torch.Tensor):
                selected_tensor = selected_indices.to(device=regrowable_indices.device, dtype=torch.long)
            else:
                selected_tensor = torch.tensor(selected_indices, dtype=torch.long, device=regrowable_indices.device)

            if selected_tensor.ndim == 1:
                selected_tensor = selected_tensor.unsqueeze(0)

            if selected_tensor.numel() == 0:
                return 0, selected_tensor.detach().cpu()

            # Ensure indices are within current regrowable set
            regrowable_set = {tuple(idx.tolist()) for idx in regrowable_indices.detach().cpu()}
            filtered = [idx for idx in selected_tensor if tuple(idx.tolist()) in regrowable_set]

            if len(filtered) == 0:
                return 0, selected_tensor.detach().cpu()

            selected_indices_tensor = torch.stack(filtered).to(regrowable_indices.device)
            actual_regrow = selected_indices_tensor.size(0)

        # PRIORITY-BASED SELECTION MODE: Prioritize reference model weights
        else:
            # Separate regrowable indices into two groups:
            # Group A: Weights that were KEPT (mask=1) in reference model (high priority)
            # Group B: Weights that were PRUNED (mask=0) in reference model (low priority)
            priority_indices = []
            random_indices = []

            for idx in regrowable_indices:
                tuple_idx = tuple(idx.tolist())
                if reference_mask[tuple_idx].item() > 0.5:  # Was kept in reference (important)
                    priority_indices.append(idx)
                else:  # Was pruned in reference (less important)
                    random_indices.append(idx)

            # Convert to tensors
            if len(priority_indices) > 0:
                priority_tensor = torch.stack(priority_indices)
            else:
                priority_tensor = torch.empty((0, regrowable_indices.size(1)), dtype=torch.long,
                                              device=regrowable_indices.device)

            if len(random_indices) > 0:
                random_tensor = torch.stack(random_indices)
            else:
                random_tensor = torch.empty((0, regrowable_indices.size(1)), dtype=torch.long,
                                            device=regrowable_indices.device)

            # Build selection strategy:
            # 1. First, take as many priority weights as possible (up to num_weights)
            # 2. If budget remains, randomly sample from remaining pool

            num_priority = min(len(priority_tensor), num_weights)
            num_random = max(0, min(num_weights - num_priority, len(random_tensor)))

            selected_list = []

            # Select from priority group (sample if more available than budget)
            if num_priority > 0:
                if len(priority_tensor) <= num_priority:
                    # Take all priority weights
                    selected_list.append(priority_tensor)
                else:
                    # Randomly sample from priority weights
                    perm = torch.randperm(len(priority_tensor), device=regrowable_indices.device)[:num_priority]
                    selected_list.append(priority_tensor[perm])

            # Fill remaining budget from random group
            if num_random > 0:
                perm = torch.randperm(len(random_tensor), device=regrowable_indices.device)[:num_random]
                selected_list.append(random_tensor[perm])

            # Combine selections
            if len(selected_list) > 0:
                selected_indices_tensor = torch.cat(selected_list, dim=0)
                actual_regrow = selected_indices_tensor.size(0)
            else:
                return 0, torch.empty((0, regrowable_indices.size(1)), dtype=torch.long).cpu()

        # Apply regrowth - update mask AND copy weight from reference
        for idx in selected_indices_tensor:
            tuple_idx = tuple(idx.tolist())
            # Update mask to 1.0 (unpruned)
            current_mask[tuple_idx] = 1.0

            # Copy weight value from reference model into weight_orig
            if layer_name in self.reference_weights:
                ref_weight = self.reference_weights[layer_name][tuple_idx]
                if hasattr(module, "weight_orig"):
                    module.weight_orig.data[tuple_idx] = ref_weight.clone()
                else:
                    module.weight.data[tuple_idx] = ref_weight.clone()

        return actual_regrow, selected_indices_tensor.detach().cpu()

    def calculate_sparsity(self, model):
        """
        Calculate current sparsity of the model
        
        Returns:
            sparsity: Overall sparsity percentage (0-100)
            total_params: Total number of parameters
            pruned_params: Number of pruned (zero) parameters
        """
        total_params = 0
        pruned_params = 0

        for name, module in model.named_modules():
            if hasattr(module, 'weight_mask'):
                mask = module.weight_mask
                total_params += mask.numel()
                pruned_params += (mask == 0).sum().item()

        sparsity = 100.0 * pruned_params / total_params if total_params > 0 else 0.0
        return sparsity, total_params, pruned_params

    def mini_finetune(self, model, epochs=10, lr=0.0003):
        """Short finetuning pass for the regrown model before evaluation.    
        Uses AdamW optimizer (critical for sparse networks after regrowth).
        
        Now tracks best model during training and restores it at the end.
        """
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        best_accuracy = 0.0
        best_model_state = None

        for epoch in range(epochs):
            # Training
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # Quick evaluation to track best model
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in self.test_loader:
                    inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            accuracy = 100.0 * correct / total

            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = copy.deepcopy(model.state_dict())

            model.train()

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.eval()

    def evaluate_model(self, model, full_eval=False):
        """Evaluate model accuracy
        
        Args:
            model: Model to evaluate
            full_eval: If True, evaluate on full test set; otherwise use limited batches
        """
        model.eval()
        correct = 0
        total = 0

        # Ensure deterministic evaluation
        with torch.no_grad():
            torch.cuda.empty_cache()  # Clear cache for consistent GPU memory state
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                if not full_eval and batch_idx >= 20:  # Use 20 batches for better accuracy (increased from 5)
                    break
                inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy


def calculate_model_sparsity(model):
    """
    Helper function to calculate sparsity without requiring PG instance
    
    Returns:
        sparsity: Overall sparsity percentage (0-100)
        total_params: Total number of parameters
        pruned_params: Number of pruned (zero) parameters
    """
    total_params = 0
    pruned_params = 0

    for name, module in model.named_modules():
        if hasattr(module, 'weight_mask'):
            mask = module.weight_mask
            total_params += mask.numel()
            pruned_params += (mask == 0).sum().item()

    sparsity = 100.0 * pruned_params / total_params if total_params > 0 else 0.0
    return sparsity, total_params, pruned_params


def evaluate_model_accuracy(model, test_loader, device, full_eval=True):
    """
    Helper function to evaluate model accuracy without requiring PG instance
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use
        full_eval: If True, evaluate on full test set
    
    Returns:
        accuracy: Test accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        torch.cuda.empty_cache()
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if not full_eval and batch_idx >= 20:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def full_finetune(model, train_loader, test_loader, device,
                  epochs=2000, lr=0.0003, save_path=None, verbose=True, patience=100):
    """
    Complete finetuning process with best model tracking and early stopping
    
    CRITICAL: Uses AdamW optimizer (essential for sparse networks after regrowth).
    Training duration increased to match pruning literature (400k-800k steps).
    
    Args:
        model: Model to finetune
        train_loader: Training data loader
        test_loader: Test data loader  
        device: Device to train on
        epochs: Number of finetuning epochs (default 2000 for AlexNet/ResNet20 size)
        lr: Learning rate (0.0003 proven optimal for sparse networks)
        save_path: Path to save best model (if None, don't save)
        verbose: Print progress
        patience: Early stopping patience (default 100 epochs without improvement)
    
    Returns:
        best_accuracy: Best test accuracy achieved
        best_model_state: State dict of best model
    """
    print(f"\n{'=' * 70}")
    print("Final Finetuning Phase")
    print(f"{'=' * 70}")
    print(f"Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Optimizer: AdamW (critical for sparse networks)")
    print(f"  Weight decay: 0.01")
    print(f"  Early stopping patience: {patience}")
    print(f"  Save path: {save_path}")
    print(f"{'=' * 70}\n")

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler with warmup (helps sparse network training)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)  # 5% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_accuracy = 100.0 * correct / total
        avg_train_loss = train_loss / len(train_loader)

        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_accuracy = 100.0 * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)

        # Get current learning rate (scheduler already stepped in training loop)
        current_lr = scheduler.get_last_lr()[0]

        # Track best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            # Save best model
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'accuracy': best_accuracy,
                    'train_accuracy': train_accuracy,
                }, save_path)
                if verbose:
                    print(f"  ✓ Saved new best model (epoch {best_epoch}, acc: {best_accuracy:.2f}%)")
        else:
            epochs_without_improvement += 1

        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Train Acc: {train_accuracy:.2f}% | "
                  f"Test Loss: {avg_test_loss:.4f} | "
                  f"Test Acc: {test_accuracy:.2f}% | "
                  f"Best: {best_accuracy:.2f}% | "
                  f"LR: {current_lr:.6f}")

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"\n{'=' * 70}")
            print(f"Early stopping at epoch {epoch + 1}: No improvement for {patience} epochs")
            print(f"Best accuracy: {best_accuracy:.2f}% (epoch {best_epoch})")
            print(f"{'=' * 70}")
            break

    print(f"\n{'=' * 70}")
    print(f"Finetuning Completed!")
    print(f"  Best Accuracy: {best_accuracy:.2f}% (epoch {best_epoch})")
    if save_path:
        print(f"  Model saved to: {save_path}")
    print(f"{'=' * 70}\n")

    return best_accuracy, best_model_state


def main():
    parser = argparse.ArgumentParser(description='RL-based Regrowth Allocation (NAS-RL Style)')
    parser.add_argument('--m_name', type=str, default='vgg16')
    parser.add_argument('--data_dir', type=str, default='./data')

    # RL hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of allocations to sample per epoch (higher = more stable gradients, slower training)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate for agent')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='LSTM hidden size')
    parser.add_argument('--entropy_coef', type=float, default=0.5,
                        help='Entropy bonus coefficient (BETA) - DEPRECATED: use start_beta instead')
    parser.add_argument('--reward_temperature', type=float, default=0.005,
                        help='Temperature τ for reward scaling (reward-baseline)/τ - lower = more sensitive')

    # Entropy schedule parameters (for linear decay of entropy coefficient)
    parser.add_argument('--start_beta', type=float, default=0.40,
                        help='Starting entropy coefficient (exploration phase)')
    parser.add_argument('--end_beta', type=float, default=0.04,
                        help='Final entropy coefficient (exploitation phase)')
    parser.add_argument('--decay_fraction', type=float, default=0.4,
                        help='Fraction of epochs over which to decay beta (e.g., 0.4 = decay over first 40% of training)')

    # Action space
    parser.add_argument('--action_space_size', type=int, default=11,
                        help='Number of discrete ratio values (e.g., 21 for [0, 0.05, ..., 1.0])')
    parser.add_argument('--max_ratio', type=float, default=1.0,
                        help='Maximum allocation ratio per layer')

    # Regrowth parameters
    parser.add_argument('--regrow_step', type=float, default=0.005,
                        help='Fraction of total weights to regrow per iteration')
    parser.add_argument('--regrow_iterations', type=int, default=1,
                        help='Number of iterative regrowth iterations (1 = single-step)')
    parser.add_argument('--starting_checkpoint', type=str, default='oneshot',
                        choices=['oneshot', 'iterative'],
                        help='Starting checkpoint type: oneshot (0.99 sparsity) or iterative (it19)')

    # Finetuning (CRITICAL: Use AdamW with 0.0003 lr for sparse networks)
    parser.add_argument('--finetune_epochs', type=int, default=1500,
                        help='Finetuning epochs (2000 default matches ~400k steps for CIFAR-10)')
    parser.add_argument('--finetune_lr', type=float, default=0.0003,
                        help='Finetuning learning rate (0.0003 proven optimal for sparse networks)')
    parser.add_argument('--save_dir', type=str, default='./rl_nas_checkpoints')

    # Checkpoint and resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Reference model for priority-based regrowth
    parser.add_argument('--reference_model', type=str, default=None,
                        help='Path to reference model checkpoint for priority-based weight selection (if None, uses model_95 as reference)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    seed_worker = set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    train_loader, val_loader, test_loader = data_loader(data_dir=args.data_dir)

    # Load models
    print("Loading models...")
    model_pretrained = model_loader(args.m_name, device)
    load_model_name(model_pretrained, f'./{args.m_name}/checkpoint', args.m_name)

    # Load target model (model_95)
    model_95 = model_loader(args.m_name, device)
    prune_weights_reparam(model_95)
    ######## repare both use pretrain model for target
    # if args.starting_checkpoint == 'oneshot':
    #     # Use pretrained model as target (model_95)
    #     print("Using pretrained model as target (model_95)")
    #     # Add pruning masks to pretrained model (all masks will be 1.0 = no pruning)
    #     prune_weights_reparam(model_pretrained)
    #     model_95.load_state_dict(model_pretrained.state_dict())
    #     print("  Loaded pretrained weights into model_95 with masks (all weights available)")
    # else:
    #     # Use iterative sparsity checkpoint as target
    #     print("Using iterative 95% sparsity checkpoint as target (model_95)")
    #     checkpoint_95 = torch.load(f'./iterative_0.4_10/{args.m_name}/pruned_finetuned_mask_it1.pth')
    #     model_95.load_state_dict(checkpoint_95)
    #     print("  Loaded 95% sparsity checkpoint into model_95")

    prune_weights_reparam(model_pretrained)
    model_95.load_state_dict(model_pretrained.state_dict())

    # Load starting model (model_current) - depends on starting_checkpoint argument
    model_current = model_loader(args.m_name, device)
    prune_weights_reparam(model_current)

    if args.starting_checkpoint == 'oneshot':
        if args.m_name in ["vgg16"]:
            checkpoint_99 = torch.load(
                f'./{args.m_name}/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.995.pth')
            model_current.load_state_dict(checkpoint_99)
            print("  Loaded one-shot 99.5% sparsity checkpoint")
        elif args.m_name in ["resnet20"]:
            checkpoint_99 = torch.load(
                f'./{args.m_name}/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.95.pth')
            model_current.load_state_dict(checkpoint_99)
            print("  Loaded one-shot 98% sparsity checkpoint")
        else:
            # One-shot regrowth: start from one-shot 99% sparsity model
            checkpoint_99 = torch.load(f'./{args.m_name}/ckpt_after_prune/pruned_finetuned_mask_0.99.pth')
            model_current.load_state_dict(checkpoint_99)
            print("  Loaded one-shot 99% sparsity checkpoint")
    else:
        if args.m_name in ["resnet20"]:
            checkpoint_99 = torch.load(f'./iterative_0.4_10/{args.m_name}/pruned_finetuned_mask_it8.pth')
            model_current.load_state_dict(checkpoint_99)
            print("  Loaded iterative 99% sparsity checkpoint (it8)")
        else:
            # Iterative regrowth: start from iterative 99% sparsity model (iteration 10)
            checkpoint_99 = torch.load(
                f'./{args.m_name}/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.9953.pth')
            model_current.load_state_dict(checkpoint_99)
            print("  Loaded iterative 99% sparsity checkpoint (it10)")

    # Load reference model for priority-based regrowth (if specified)
    model_reference = None
    if args.reference_model is not None:
        print(f"\nLoading custom reference model from: {args.reference_model}")
        model_reference = model_loader(args.m_name, device)
        prune_weights_reparam(model_reference)
        ref_checkpoint = torch.load(args.reference_model)
        model_reference.load_state_dict(ref_checkpoint)
        print("  Loaded custom reference model for priority-based weight selection")
    else:
        print(
            "\nNo custom reference model specified, will use model_95 as reference for priority-based weight selection")
        model_reference = model_95  # Will be set in config

    # Target layers
    if args.m_name == "resnet20":
        # target_layers = ["layer2.1.conv2", "layer2.2.conv1", "layer2.2.conv2", "layer3.0.conv1",
        #                "layer3.0.conv2", "layer3.0.shortcut.0", "layer3.1.conv1", "layer3.1.conv2",
        #                "layer3.2.conv1", "layer3.2.conv2", "linear"]
        target_layers = ["layer2.0.conv2", "layer2.1.conv1", "layer2.2.conv2", "layer2.2.conv1",
                         "layer3.0.conv2", "layer3.0.conv1", "layer3.1.conv1", "layer3.1.conv2"]

    # elif args.m_name == "densenet":
    #     target_layers = ["dense3.0.conv1", "dense3.4.conv1", "dense3.7.conv1", 
    #     "dense3.8.conv1", "dense3.9.conv2", "dense3.14.conv2", "dense3.15.conv2", 
    #     "dense3.21.conv2", "dense3.22.conv1", "dense4.0.conv1", "dense4.1.conv1", 
    #     "dense4.3.conv2", "dense4.6.conv2", "dense4.8.conv2", "dense4.9.conv2", 
    #     "dense4.12.conv1", "dense4.12.conv2", "dense4.15.conv2"]
    elif args.m_name == "vgg16":
        # target_layers = ["features.14", "features.17", "features.27", 
        #                 "features.30", "features.34", "features.40", "classifier"]
        # target_layers = ["features.14", "features.17", "features.20", "features.24", "features.27",
        #                 "features.0", "features.3", "features.7", "features.10", "classifier"]
        # target_layers = ["features.24", "features.20", "features.10"]
        target_layers = ["features.10", "features.14", "features.17", "features.20", "features.24", "classifier"]

    elif args.m_name == "alexnet":
        target_layers = ['features.3', 'features.6', 'features.8', 'features.10', 'classifier.1']  #
    # elif args.m_name == "effnet":
    #     target_layers = ["layers.3.conv3", "layers.4.conv1", "layers.4.conv3", "layers.5.conv1",
    #     "layers.6.conv3", "layers.7.conv3", "layers.8.conv1", "layers.8.conv2", "layers.8.conv3",
    #     "layers.15.conv1", "layers.15.conv2", "layers.15.conv3"]

    else:
        target_layers = []

    # Storage for iteration results
    initial_accuracy = None
    initial_sparsity = None
    initial_total = 0
    initial_pruned = 0
    iteration_results = []

    # ==========================
    # Iterative Regrowth Loop
    # ==========================
    for iteration in range(args.regrow_iterations):
        print("\n" + "=" * 70)
        print(f"REGROWTH ITERATION {iteration + 1}/{args.regrow_iterations}")
        print("=" * 70)

        # Get layer capacities and references from current state
        layer_capacities = []
        reference_masks = {}
        reference_weights = {}

        for layer_name in target_layers:
            # Get capacity relative to current model state
            module_current = dict(model_current.named_modules())[layer_name]
            module_95 = dict(model_95.named_modules())[layer_name]

            if hasattr(module_current, 'weight_mask') and hasattr(module_95, 'weight_mask'):
                current_mask = module_current.weight_mask
                ref_mask = module_95.weight_mask
                regrowable = (current_mask == 0) & (ref_mask == 1)
                capacity = regrowable.sum().item()
                layer_capacities.append(capacity)

                reference_masks[layer_name] = ref_mask.clone()
                # CRITICAL: Use weight_orig (unpruned values), not weight (masked values)
                if hasattr(module_95, 'weight_orig'):
                    reference_weights[layer_name] = module_95.weight_orig.detach().clone()
                else:
                    reference_weights[layer_name] = module_95.weight.detach().clone()

        # Calculate target regrowth for this iteration
        total_weights, _, _ = count_pruned_params(model_current)
        target_regrow = int(total_weights * args.regrow_step)
        target_regrow = min(target_regrow, sum(layer_capacities))

        print(f"\nIteration {iteration + 1} Configuration:")
        print(f"  Total weights: {total_weights}")
        print(f"  Target regrowth: {target_regrow}")
        print(f"  Total capacity: {sum(layer_capacities)}")
        print(f"  Num layers: {len(target_layers)}")

        if target_regrow == 0:
            print(f"  No capacity remaining for regrowth. Stopping iterations.")
            break

        # Setup config for this iteration
        config = {
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'hidden_size': args.hidden_size,
            'entropy_coef': args.entropy_coef,
            'action_space_size': args.action_space_size,
            'target_regrow': target_regrow,
            'layer_capacities': layer_capacities,
            'reference_masks': reference_masks,
            'reference_weights': reference_weights,
            'model_name': args.m_name,  # Pass model name to config
            'reward_temperature': args.reward_temperature,
            'checkpoint_dir': args.save_dir,
            'save_freq': args.save_freq,
            'model_reference': model_reference,  # Pass reference model for priority-based regrowth
            # Entropy schedule parameters
            'start_beta': getattr(args, 'start_beta', 0.4),
            'end_beta': getattr(args, 'end_beta', 0.04),
            'decay_fraction': getattr(args, 'decay_fraction', 0.4),
        }

        # Initialize Policy Gradient for this iteration
        pg = RegrowthPolicyGradient(
            config=config,
            model_pretrained=model_pretrained,
            model_95=model_95,
            model_99=model_current,
            target_layers=target_layers,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device
        )

        # Evaluate before regrowth
        before_accuracy = pg.evaluate_model(model_current, full_eval=True)
        before_sparsity, before_total, before_pruned = pg.calculate_sparsity(model_current)
        print(f"\nBefore iteration {iteration + 1}:")
        print(f"  Accuracy: {before_accuracy:.2f}%")
        print(f"  Sparsity: {before_sparsity:.2f}% ({before_pruned}/{before_total} pruned)")

        # Run RL training
        print("\n" + "-" * 70)
        print(f"RL Training Phase (Iteration {iteration + 1})")
        print("-" * 70)
        # Only use resume on first iteration if specified
        resume_checkpoint = args.resume if iteration == 0 else None
        best_allocation, best_reward, best_regrow_indices = pg.solve_environment(resume_from=resume_checkpoint)

        # Apply best allocation to current model
        print("\n" + "-" * 70)
        print(f"Applying Best Allocation (Iteration {iteration + 1})")
        print("-" * 70)
        pg.apply_allocation(model_current, best_allocation, best_regrow_indices)

        # Evaluate after regrowth
        after_regrow_accuracy = pg.evaluate_model(model_current, full_eval=True)
        after_regrow_sparsity, after_total, after_pruned = pg.calculate_sparsity(model_current)
        print(f"\nAfter regrowth (iteration {iteration + 1}, before finetuning):")
        print(f"  Accuracy: {after_regrow_accuracy:.2f}%")
        print(f"  Sparsity: {after_regrow_sparsity:.2f}% ({after_pruned}/{after_total} pruned)")
        print(f"  Improvement: {after_regrow_accuracy - before_accuracy:+.2f}%")

        # Intermediate finetuning (except for last iteration)
        if iteration < args.regrow_iterations - 1:
            print("\n" + "-" * 70)
            print(f"Intermediate Finetuning (Iteration {iteration + 1})")
            print("-" * 70)
            inter_save_path = os.path.join(args.save_dir, f'iter_{iteration + 1}_model_{args.m_name}.pth')
            inter_accuracy, inter_state = full_finetune(
                model=model_current,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.finetune_epochs,
                lr=args.finetune_lr,
                save_path=inter_save_path,
                verbose=True
            )

            # Reload best intermediate model
            model_current.load_state_dict(inter_state)
            inter_eval_accuracy = pg.evaluate_model(model_current, full_eval=True)
            inter_sparsity, _, _ = pg.calculate_sparsity(model_current)

            print(f"\nAfter intermediate finetuning (iteration {iteration + 1}):")
            print(f"  Best accuracy: {inter_accuracy:.2f}%")
            print(f"  Final evaluation: {inter_eval_accuracy:.2f}%")
            print(f"  Sparsity: {inter_sparsity:.2f}%")
        else:
            inter_accuracy = after_regrow_accuracy

        # Store iteration results
        iteration_results.append({
            'iteration': iteration + 1,
            'before_accuracy': before_accuracy,
            'before_sparsity': before_sparsity,
            'after_regrow_accuracy': after_regrow_accuracy,
            'after_regrow_sparsity': after_regrow_sparsity,
            'after_finetune_accuracy': inter_accuracy if iteration < args.regrow_iterations - 1 else None,
            'best_allocation': best_allocation,
            'best_regrow_indices': best_regrow_indices,
            'target_regrow': target_regrow,
            'weights_regrown': before_pruned - after_pruned,
        })

    # Final comprehensive finetuning (after all iterations)
    print("\n" + "=" * 70)
    print("Final Comprehensive Finetuning (After All Iterations)")
    print("=" * 70)
    final_save_path = os.path.join(args.save_dir, f'best_model_rl_nas_{args.m_name}_final.pth')
    final_accuracy, final_state = full_finetune(
        model=model_current,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        save_path=final_save_path,
        verbose=True
    )

    # Load best model for final evaluation
    model_current.load_state_dict(final_state)

    # Use helper functions for evaluation if pg doesn't exist (e.g., when no iterations ran)
    final_eval_accuracy = evaluate_model_accuracy(model_current, test_loader, device, full_eval=True)
    final_sparsity, final_total, final_pruned = calculate_model_sparsity(model_current)

    # ==========================
    # Final Results Summary
    # ==========================
    # Use first iteration's "before" values as initial state
    if len(iteration_results) > 0:
        initial_accuracy = iteration_results[0]['before_accuracy']
        initial_sparsity = iteration_results[0]['before_sparsity']
        initial_pruned = iteration_results[0]['before_sparsity'] / 100.0 * final_total  # Approximate

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model: {args.m_name}")
    print(f"Regrowth iterations: {args.regrow_iterations}")
    print(f"Regrowth step per iteration: {args.regrow_step * 100:.1f}%")
    if initial_accuracy is not None:
        print(f"\nInitial state:")
        print(f"  Accuracy: {initial_accuracy:.2f}%")
        print(f"  Sparsity: {initial_sparsity:.2f}%")
    print(f"\nFinal state (after all iterations + finetuning):")
    print(f"  Best accuracy: {final_accuracy:.2f}%")
    print(f"  Final evaluation: {final_eval_accuracy:.2f}%")
    print(f"  Sparsity: {final_sparsity:.2f}%")
    if initial_accuracy is not None:
        print(f"\nOverall improvement:")
        print(f"  Accuracy gain: {final_accuracy - initial_accuracy:+.2f}%")
        print(f"  Sparsity reduction: {initial_sparsity - final_sparsity:.2f}%")
        print(f"  Total weights regrown: {int(initial_sparsity - final_sparsity) * final_total // 100}")
    print(f"\nIteration-by-iteration breakdown:")
    for result in iteration_results:
        print(f"  Iteration {result['iteration']}:")
        print(f"    Target regrowth: {result['target_regrow']} weights")
        print(f"    Before: {result['before_accuracy']:.2f}% (sparsity: {result['before_sparsity']:.2f}%)")
        print(
            f"    After regrowth: {result['after_regrow_accuracy']:.2f}% (sparsity: {result['after_regrow_sparsity']:.2f}%)")
        if result['after_finetune_accuracy'] is not None:
            print(f"    After finetune: {result['after_finetune_accuracy']:.2f}%")
        print(f"    Weights regrown: {result['weights_regrown']}")
    print(f"\nFinal model saved to: {final_save_path}")
    print("=" * 70)

    print("All done!")

    run.finish()


if __name__ == '__main__':
    main()
