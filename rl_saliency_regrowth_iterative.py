import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import copy
import os
import random
from pathlib import Path
from torch.distributions import Categorical
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from utils.model_loader import model_loader
from utils.data_loader import data_loader
from utils.analysis_utils import (
    load_model_name, prune_weights_reparam, count_pruned_params
)


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SaliencyComputer:
    """Compute gradient-based saliency scores (FairPrune formula)"""

    def __init__(self, model, criterion, device='cuda'):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.accumulated_grads = {}
        self.grad_count = 0

    def reset(self):
        self.accumulated_grads = {}
        self.grad_count = 0

    def compute(self, data_loader, target_layers, max_batches=None):
        """Compute saliency scores: S(θ) = (∂L/∂θ)² * θ²"""
        self.model.eval()
        self.reset()

        module_dict = dict(self.model.named_modules())

        for layer_name in target_layers:
            module = module_dict.get(layer_name)
            if module and hasattr(module, 'weight'):
                self.accumulated_grads[layer_name] = torch.zeros_like(
                    module.weight, device=self.device
                )

        batch_idx = 0
        for inputs, labels in tqdm(data_loader, desc="Computing saliency"):
            if max_batches and batch_idx >= max_batches:
                break

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.model.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)

            for param, grad in zip(self.model.parameters(), grads):
                if grad is None:
                    continue

                for name, p in self.model.named_parameters():
                    if p is param:
                        for layer_name in target_layers:
                            if name == f"{layer_name}.weight":
                                hessian_approx = grad.pow(2).detach()
                                param_squared = param.data.pow(2).detach()
                                self.accumulated_grads[layer_name] += hessian_approx * param_squared
                                break
                        break
            batch_idx += 1
            self.grad_count += 1

        saliency_dict = {}
        for layer_name in target_layers:
            if layer_name in self.accumulated_grads:
                saliency_dict[layer_name] = (
                        self.accumulated_grads[layer_name] / max(self.grad_count, 1)
                ).cpu()

        return saliency_dict


class RegrowthAgent(nn.Module):
    """LSTM controller for allocation decisions"""

    def __init__(self, action_dim, hidden_size, context_dim, device='cuda'):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.context_dim = context_dim

        self.lstm = nn.LSTMCell(action_dim + context_dim, hidden_size)
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
        return self.decoder(h_t)

    def init_hidden(self):
        return (
            torch.zeros(1, self.hidden_size, device=self.device),
            torch.zeros(1, self.hidden_size, device=self.device)
        )


class SaliencyRegrowth:
    """Saliency-based weight regrowth (RigL-style)"""

    @staticmethod
    @torch.no_grad()
    def apply(model, layer_name, saliency_tensor, num_weights,
              init_strategy='zero', device='cuda'):
        """Apply top-K saliency-based regrowth"""
        module_dict = dict(model.named_modules())
        module = module_dict.get(layer_name)

        if not module or not hasattr(module, 'weight_mask'):
            return 0, []

        mask = module.weight_mask
        saliency = saliency_tensor.to(device)
        pruned_pos = (mask == 0)

        if not pruned_pos.any():
            return 0, []

        saliency_masked = saliency.clone()
        saliency_masked[~pruned_pos] = -float('inf')

        flat_saliency = saliency_masked.flatten()
        k = min(num_weights, (flat_saliency > -float('inf')).sum().item())

        if k == 0:
            return 0, []

        _, top_k_indices = torch.topk(flat_saliency, k=k)

        regrown_indices = []
        weight_param = module.weight_orig if hasattr(module, 'weight_orig') else module.weight

        for flat_idx in top_k_indices:
            multi_idx = np.unravel_index(flat_idx.cpu().item(), saliency.shape)
            regrown_indices.append(multi_idx)
            mask[multi_idx] = 1.0

            if init_strategy == 'zero':
                weight_param.data[multi_idx] = 0.0
            elif init_strategy == 'kaiming':
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight_param)
                bound = np.sqrt(6.0 / fan_in)
                weight_param.data[multi_idx] = torch.empty(1).uniform_(-bound, bound).item()
            elif init_strategy == 'xavier':
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight_param)
                bound = np.sqrt(6.0 / (fan_in + fan_out))
                weight_param.data[multi_idx] = torch.empty(1).uniform_(-bound, bound).item()

        return k, regrown_indices


class PolicyGradient:
    """RL Policy Gradient with Saliency-Based Regrowth"""

    def __init__(self, config, model_pretrained, model_pruned, target_layers,
                 train_loader, test_loader, device):
        self.config = config
        self.device = device

        self.model_pretrained = model_pretrained.to(device)
        self.model_pruned = model_pruned.to(device)
        self.target_layers = target_layers
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.num_epochs = config['num_epochs']
        self.action_space = config['action_space_size']
        self.num_steps = len(target_layers)
        self.target_regrow = config['target_regrow']
        self.layer_capacities = config['layer_capacities']
        self.total_capacity = max(sum(self.layer_capacities), 1)

        # Compute saliency
        print("\nComputing initial saliency scores...")
        self.saliency_computer = SaliencyComputer(
            self.model_pretrained, nn.CrossEntropyLoss(), device
        )
        self.saliency_dict = self.saliency_computer.compute(
            train_loader, target_layers, config.get('saliency_max_batches', 50)
        )

        # RL agent
        self.agent = RegrowthAgent(
            action_dim=self.action_space,
            hidden_size=config['hidden_size'],
            context_dim=config.get('context_dim', 3),
            device=device
        ).to(device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=config['learning_rate'])
        self.reward_baseline = None
        self.baseline_decay = config.get('baseline_decay', 0.9)

        # Checkpointing
        self.checkpoint_dir = Path(config['checkpoint_dir']) / config['model_name'] / config['method']
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_reward = float('-inf')
        self.best_allocation = None
        self.best_regrow_indices = None

        # Wandb
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb.init(
                    project=config.get('wandb_project', 'regrowth'),
                    name=config.get('wandb_name', f"{config['model_name']}_regrowth"),
                    config=config
                )
            except ImportError:
                print("Warning: wandb not installed, disabling logging")
                self.use_wandb = False

    def get_entropy_coef(self, epoch):
        """Entropy coefficient with decay"""
        if not self.config.get('use_entropy_schedule', True):
            return self.config['entropy_coef']

        decay_epochs = self.num_epochs * self.config.get('decay_fraction', 0.4)
        start_beta = self.config.get('start_beta', 0.4)
        end_beta = self.config.get('end_beta', 0.04)

        if epoch < decay_epochs:
            return start_beta - (start_beta - end_beta) * (epoch / decay_epochs)
        return end_beta

    def create_model_copy(self):
        """Create copy of pruned model"""
        model = model_loader(self.config['model_name'], self.device)
        prune_weights_reparam(model)
        model.load_state_dict(self.model_pruned.state_dict())
        return model

    def train(self, resume_from=None):
        """Main RL training loop"""
        start_epoch = 0

        if resume_from and os.path.exists(resume_from):
            checkpoint = torch.load(resume_from)
            self.agent.load_state_dict(checkpoint['agent_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.best_reward = checkpoint['best_reward']
            self.best_allocation = checkpoint['best_allocation']
            self.best_regrow_indices = checkpoint['best_regrow_indices']
            if 'reward_baseline' in checkpoint:
                self.reward_baseline = checkpoint['reward_baseline']
            print(f"Resumed from epoch {start_epoch}, best reward: {self.best_reward:.4f}")

        pbar = tqdm(range(start_epoch, self.num_epochs), desc="RL Training")
        for epoch in pbar:
            metrics = self.run_episode()

            if metrics['reward'] > self.best_reward:
                self.best_reward = metrics['reward']
                self.best_allocation = metrics['allocation']
                self.best_regrow_indices = metrics['regrow_indices']
                self.save_best(epoch)

            beta = self.get_entropy_coef(epoch)
            loss, entropy = self.compute_loss(
                metrics['logits'], metrics['weighted_log_prob'], beta
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update progress bar
            pbar.set_postfix({
                'reward': f"{metrics['reward']:.4f}",
                'best': f"{self.best_reward:.4f}",
                'loss': f"{loss.item():.4f}"
            })

            # Wandb logging
            if self.use_wandb:
                self.wandb.log({
                    'epoch': epoch + 1,
                    'reward': metrics['reward'],
                    'best_reward': self.best_reward,
                    'loss': loss.item(),
                    'entropy': entropy.item(),
                    'beta': beta
                })

            if (epoch + 1) % self.config.get('save_freq', 10) == 0:
                self.save_checkpoint(epoch)

        if self.use_wandb:
            self.wandb.finish()

        return self.best_allocation, self.best_reward, self.best_regrow_indices

    def run_episode(self):
        """Run one RL episode with saliency-based regrowth"""
        self.agent.hidden = self.agent.init_hidden()

        prev_logits = torch.zeros(1, self.action_space, device=self.device)
        ratio_options = torch.linspace(0, 1, self.action_space, device=self.device)

        remaining_budget = int(self.target_regrow)
        log_probs = []
        masked_logits_list = []
        allocations = {}

        for idx, layer_name in enumerate(self.target_layers):
            capacity = int(self.layer_capacities[idx])
            context = torch.tensor([
                idx / max(self.num_steps - 1, 1),
                capacity / self.total_capacity,
                remaining_budget / self.target_regrow
            ], device=self.device).unsqueeze(0)

            logits = self.agent(prev_logits, context).squeeze(0)

            max_alloc = min(capacity, remaining_budget)
            counts = torch.round(ratio_options * max_alloc).long()
            feasible = counts <= remaining_budget
            if not feasible.any():
                feasible[0] = True

            masked_logits = torch.where(
                feasible, logits, torch.full_like(logits, -1e9)
            )

            dist = Categorical(F.softmax(masked_logits, dim=0))
            action = dist.sample()

            chosen_count = min(int(counts[action].item()), capacity, remaining_budget)
            remaining_budget = max(remaining_budget - chosen_count, 0)

            if chosen_count > 0:
                allocations[layer_name] = chosen_count

            log_probs.append(dist.log_prob(action))
            masked_logits_list.append(masked_logits)
            prev_logits = logits.unsqueeze(0)

        # Apply regrowth
        model = self.create_model_copy()
        regrow_indices = {}

        for layer_name, num_weights in allocations.items():
            saliency = self.saliency_dict.get(layer_name)
            if saliency is not None:
                _, indices = SaliencyRegrowth.apply(
                    model, layer_name, saliency, num_weights,
                    self.config.get('init_strategy', 'zero'), self.device
                )
                regrow_indices[layer_name] = indices

        # Finetune and evaluate
        self.mini_finetune(model, self.config.get('mini_finetune_epochs', 40))
        accuracy = self.evaluate(model, full=True)
        reward = accuracy / 100.0

        # Compute advantage
        if self.reward_baseline is None:
            self.reward_baseline = reward

        advantage = (reward - self.reward_baseline) / max(self.config.get('reward_temperature', 0.01), 1e-6)
        advantage = float(np.clip(advantage, -100, 100))

        self.reward_baseline = (
                self.baseline_decay * self.reward_baseline +
                (1 - self.baseline_decay) * reward
        )

        log_probs_tensor = torch.stack(log_probs)
        weighted_log_prob = (log_probs_tensor * advantage).sum().unsqueeze(0)

        return {
            'reward': reward,
            'allocation': allocations,
            'regrow_indices': regrow_indices,
            'weighted_log_prob': weighted_log_prob,
            'logits': torch.stack(masked_logits_list) if masked_logits_list else None
        }

    def compute_loss(self, logits, weighted_log_prob, beta):
        """Compute policy loss with entropy bonus"""
        policy_loss = -weighted_log_prob.mean()

        if logits is None or logits.numel() == 0:
            entropy = torch.tensor(0.0, device=self.device)
        else:
            p = F.softmax(logits, dim=1)
            log_p = F.log_softmax(logits, dim=1)
            entropy = -(p * log_p).sum(1).mean()

        return policy_loss - beta * entropy, entropy

    def mini_finetune(self, model, epochs=40, lr=0.0003):
        """Quick finetuning"""
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_state = None

        for _ in range(epochs):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()

            acc = self.evaluate(model, full=False, max_batches=20)
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())

        if best_state:
            model.load_state_dict(best_state)
        model.eval()

    def evaluate(self, model, full=False, max_batches=None):
        """Evaluate model accuracy"""
        model.eval()
        correct = total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                if not full and max_batches and batch_idx >= max_batches:
                    break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, pred = outputs.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()

        return 100.0 * correct / total

    def save_checkpoint(self, epoch):
        """Save training checkpoint"""
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_reward': self.best_reward,
            'best_allocation': self.best_allocation,
            'best_regrow_indices': self.best_regrow_indices,
            'reward_baseline': self.reward_baseline,
        }, path)

    def save_best(self, epoch):
        """Save best allocation"""
        path = self.checkpoint_dir / "best_allocation.pth"
        torch.save({
            'epoch': epoch,
            'reward': self.best_reward,
            'accuracy': self.best_reward * 100.0,
            'allocation': self.best_allocation,
            'regrow_indices': self.best_regrow_indices,
        }, path)


def full_finetune(model, train_loader, test_loader, device, config):
    """Final finetuning with early stopping"""
    print(f"\nStarting finetuning ({config['finetune_epochs']} epochs)...")

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['finetune_lr'],
        weight_decay=config.get('finetune_weight_decay', 0.01)
    )

    steps_per_epoch = len(train_loader)
    total_steps = config['finetune_epochs'] * steps_per_epoch
    warmup_steps = int(config.get('warmup_ratio', 0.05) * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_acc = 0.0
    best_state = None
    best_epoch = 0
    no_improve = 0
    patience = config.get('finetune_patience', 30)

    pbar = tqdm(range(config['finetune_epochs']), desc="Finetuning")
    for epoch in pbar:
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                _, pred = model(inputs).max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()

        acc = 100.0 * correct / total

        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            no_improve = 0
            if config.get('finetune_save_path'):
                torch.save(best_state, config['finetune_save_path'])
        else:
            no_improve += 1

        pbar.set_postfix({'acc': f'{acc:.2f}%', 'best': f'{best_acc:.2f}%'})

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(f"Best accuracy: {best_acc:.2f}% (epoch {best_epoch})")
    return best_acc, best_state


def iterative_regrowth(args, model_pretrained, model_pruned, target_layers,
                       train_loader, test_loader, device):
    """Perform iterative regrowth with multiple rounds"""

    # Determine regrowth schedule
    if args.regrow_schedule:
        regrow_steps = args.regrow_schedule
        print(f"\nIterative regrowth schedule: {regrow_steps}")
    else:
        regrow_steps = [args.regrow_step]
        print(f"\nSingle regrowth step: {args.regrow_step}")

    num_rounds = len(regrow_steps)

    # Track progress across rounds
    results = []
    current_model = model_pruned

    saliency_computer = SaliencyComputer(
        model_pretrained,
        nn.CrossEntropyLoss(),
        device
    )
    global_saliency_dict = saliency_computer.compute(
        train_loader,
        target_layers,
        args.saliency_max_batches
    )

    for round_idx, regrow_step in enumerate(regrow_steps):
        print(f"\n{'=' * 70}")
        print(f"REGROWTH ROUND {round_idx + 1}/{num_rounds}")
        print(f"Regrowth percentage: {regrow_step * 100:.2f}%")
        print(f"{'=' * 70}")

        # Calculate layer capacities for current model
        layer_capacities = []
        for layer_name in target_layers:
            module = dict(current_model.named_modules())[layer_name]
            if hasattr(module, 'weight_mask'):
                capacity = (module.weight_mask == 0).sum().item()
                layer_capacities.append(capacity)

        total_weights, _, _ = count_pruned_params(current_model)
        target_regrow = min(int(total_weights * regrow_step), sum(layer_capacities))

        print(f"  Total pruned weights: {total_weights}")
        print(f"  Target regrowth: {target_regrow}")
        print(f"  Available capacity: {sum(layer_capacities)}")

        # Setup config for this round
        config = {
            'model_name': args.m_name,
            'method': args.method,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'hidden_size': args.hidden_size,
            'entropy_coef': args.entropy_coef,
            'action_space_size': args.action_space_size,
            'target_regrow': target_regrow,
            'layer_capacities': layer_capacities,
            'init_strategy': args.init_strategy,
            'saliency_max_batches': args.saliency_max_batches,
            'reward_temperature': args.reward_temperature,
            'checkpoint_dir': args.save_dir,
            'save_freq': args.save_freq,
            'start_beta': args.start_beta,
            'end_beta': args.end_beta,
            'decay_fraction': args.decay_fraction,
            'mini_finetune_epochs': args.mini_finetune_epochs,
            'use_wandb': args.use_wandb,
            'wandb_project': args.wandb_project,
            'wandb_name': args.wandb_name or f"{args.m_name}_{args.method}_round{round_idx + 1}",
        }

        # Initialize Policy Gradient for this round
        pg = PolicyGradient(
            config=config,
            model_pretrained=model_pretrained,
            model_pruned=current_model,
            target_layers=target_layers,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device
        )

        pg.saliency_dict = global_saliency_dict

        # Evaluate before this round
        before_acc = pg.evaluate(current_model, full=True)
        print(f"\nBefore round {round_idx + 1}: {before_acc:.2f}%")

        # Run RL training
        best_allocation, best_reward, best_regrow_indices = pg.train(resume_from=args.resume)

        # Apply best allocation
        print(f"\nApplying best allocation for round {round_idx + 1}...")
        for layer_name, num_weights in best_allocation.items():
            if num_weights > 0:
                saliency = pg.saliency_dict.get(layer_name)
                if saliency is not None:
                    SaliencyRegrowth.apply(
                        current_model, layer_name, saliency, num_weights,
                        args.init_strategy, device
                    )

        # Evaluate after regrowth
        after_regrowth_acc = pg.evaluate(current_model, full=True)
        print(f"After regrowth: {after_regrowth_acc:.2f}% (+{after_regrowth_acc - before_acc:.2f}%)")

        # Mini finetuning for every round (including the last one)
        print(f"\nMini finetuning for round {round_idx + 1} ({args.mini_finetune_epochs} epochs)...")
        mini_config = {
            'finetune_epochs': args.mini_finetune_epochs,
            'finetune_lr': args.finetune_lr,
            'finetune_weight_decay': 0.01,
            'finetune_patience': 40,
            'warmup_ratio': 0.05,
            'finetune_save_path': str(
                Path(args.save_dir) / args.m_name / args.method / f'round_{round_idx + 1}_model.pth'
            ),
        }
        mini_acc, _ = full_finetune(current_model, train_loader, test_loader, device, mini_config)
        after_mini_acc = mini_acc

        # Save round results
        round_result = {
            'round': round_idx + 1,
            'regrow_step': regrow_step,
            'before_acc': before_acc,
            'after_regrowth_acc': after_regrowth_acc,
            'after_mini_finetune_acc': after_mini_acc,
            'improvement': after_mini_acc - before_acc,
            'allocation': best_allocation,
        }
        results.append(round_result)

        # Save intermediate model
        save_path = Path(args.save_dir) / args.m_name / args.method / f'model_after_round_{round_idx + 1}.pth'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(current_model.state_dict(), save_path)
        print(f"Saved model after round {round_idx + 1}: {save_path}")

        print(f"\nRound {round_idx + 1} Summary:")
        print(f"  Before: {before_acc:.2f}%")
        print(f"  After regrowth: {after_regrowth_acc:.2f}%")
        print(f"  After mini finetune: {after_mini_acc:.2f}%")
        print(f"  Improvement: +{after_mini_acc - before_acc:.2f}%")

    return current_model, results


def parse_args():
    parser = argparse.ArgumentParser(description='RL Saliency-Based Regrowth')

    # Model and data
    parser.add_argument('--m_name', type=str, default='vgg16',
                        choices=['vgg16', 'resnet20', 'alexnet'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--method', type=str, default='oneshot',
                        choices=['oneshot', 'iterative'],
                        help='Pruning method: oneshot or iterative')
    parser.add_argument('--target_layers', type=str, nargs='+',
                        default=["features.10", "features.14", "features.17", "features.20", "features.24"],
                        help='Target layers for regrowth')
    parser.add_argument('--sparsity', type=float, default=0.995,
                        help='Sparsity level of pruned model')

    # RL hyperparameters
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of RL training epochs per round')
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
    parser.add_argument('--regrow_step', type=float, default=0.005,
                        help='Single regrowth step (used if --regrow_schedule not provided)')
    parser.add_argument('--regrow_schedule', type=float, nargs='+', default=None,
                        help='Iterative regrowth schedule (e.g., 0.005 0.01 0.015)')
    parser.add_argument('--init_strategy', type=str, default='zero',
                        choices=['zero', 'kaiming', 'xavier', 'magnitude'])
    parser.add_argument('--saliency_max_batches', type=int, default=50)

    # Finetuning
    parser.add_argument('--finetune_epochs', type=int, default=1500,
                        help='Final full finetuning epochs (only used if --skip_final_finetune is False)')
    parser.add_argument('--finetune_lr', type=float, default=0.0003)
    parser.add_argument('--mini_finetune_epochs', type=int, default=40,
                        help='Mini finetuning epochs between regrowth rounds')
    parser.add_argument('--skip_final_finetune', action='store_true',
                        help='Skip final full finetuning (only use mini finetuning)')
    parser.add_argument('--save_dir', type=str, default='./rl_saliency_checkpoints')

    # Checkpoint
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    # Wandb
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='regrowth')
    parser.add_argument('--wandb_name', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

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

    # Load pruned checkpoint
    if args.method == 'oneshot':
        checkpoint_99 = torch.load(
            f'./{args.m_name}/ckpt_after_prune_oneshot/pruned_oneshot_mask_{args.sparsity}.pth')
    elif args.method == 'iterative':
        checkpoint_99 = torch.load(
            f'./{args.m_name}/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_{args.sparsity}.pth')
    else:
        raise ValueError(f"Unknown method: {args.method}")

    target_layers = args.target_layers
    model_99.load_state_dict(checkpoint_99)

    # Initial evaluation
    print("\nEvaluating initial model...")
    model_99.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_99(inputs)
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
    initial_acc = 100.0 * correct / total

    print(f"Initial model accuracy: {initial_acc:.2f}%")

    # Perform iterative regrowth
    final_model, round_results = iterative_regrowth(
        args, model_pretrained, model_99, target_layers,
        train_loader, test_loader, device
    )

    # Optional final full finetuning
    if not args.skip_final_finetune:
        print("\n" + "=" * 70)
        print("FINAL FULL FINETUNING")
        print("=" * 70)

        final_config = {
            'finetune_epochs': args.finetune_epochs,
            'finetune_lr': args.finetune_lr,
            'finetune_weight_decay': 0.01,
            'finetune_patience': 50,
            'warmup_ratio': 0.05,
            'finetune_save_path': str(Path(args.save_dir) / args.m_name / args.method / 'final_model.pth'),
        }

        final_acc, final_state = full_finetune(final_model, train_loader, test_loader, device, final_config)
    else:
        print("\n" + "=" * 70)
        print("Skipping final full finetuning (using last round's result)")
        print("=" * 70)
        final_acc = round_results[-1]['after_mini_finetune_acc']
        # Save final model
        final_path = Path(args.save_dir) / args.m_name / args.method / 'final_model.pth'
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_model.state_dict(), final_path)
        print(f"Saved final model: {final_path}")

    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("COMPREHENSIVE RESULTS")
    print("=" * 70)
    print(f"Initial accuracy: {initial_acc:.2f}%")
    print()

    for result in round_results:
        print(f"Round {result['round']}:")
        print(f"  Regrowth step: {result['regrow_step'] * 100:.2f}%")
        print(f"  Before: {result['before_acc']:.2f}%")
        print(f"  After regrowth: {result['after_regrowth_acc']:.2f}%")
        print(f"  After mini finetune ({args.mini_finetune_epochs} epochs): {result['after_mini_finetune_acc']:.2f}%")
        print(f"  Round improvement: +{result['improvement']:.2f}%")
        print()

    print(f"Final accuracy: {final_acc:.2f}%")
    print(f"Total improvement: {final_acc - initial_acc:+.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
