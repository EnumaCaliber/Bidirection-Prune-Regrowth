"""
极简版单文件测试 - 稀疏度 + 准确率
"""
import torch
import torch.nn.functional as F
from utils.model_loader import model_loader
from utils.data_loader import data_loader
import argparse
from pathlib import Path
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim
from utils.analysis_utils import (prune_weights_reparam)

parser = argparse.ArgumentParser()
parser.add_argument('--m_name', type=str, default='vgg16')
parser.add_argument('--sparsity', default=0.98)
parser.add_argument('--model_path', type=str,
                    default='./rl_saliency_checkpoints/vgg16/oneshot/0.99/baseline_exceeded_epoch26_acc0.9067.pth')  # 改为 model_path
args = parser.parse_args()


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

            best_epoch = epoch + 1
            epochs_without_improvement = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Test Acc: {test_accuracy:.2f}% | Best: {best_accuracy:.2f}%")
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'./rl_saliency_checkpoints/vgg16/oneshot/0.99fullfinetune/epoch{epoch}.pth')


        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(f"\nBest accuracy: {best_accuracy:.2f}% (epoch {best_epoch})")
    return best_accuracy, best_model_state


# Main
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader, _, test_loader = data_loader(data_dir='./data')

model_99 = model_loader(args.m_name, device)
prune_weights_reparam(model_99)
checkpoint_99 = torch.load(args.model_path,weights_only=False).get('model_state_dict')
model_99.load_state_dict(checkpoint_99)
# model_99.load_state_dict(checkpoint_99)

# for name, module in model_99.named_modules():
#     if hasattr(module, 'weight_mask'):
#         sparsity = (module.weight_mask == 0).float().mean().item()
#         print(f"{name}: mask存在, sparsity={sparsity:.2%}")




final_acc = full_finetune(
    model=model_99,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    epochs=400,
    lr=0.0003,
    save_path=f"./rl_saliency_checkpoints/vgg16/oneshot/0.99fullfinetune/final_model_sparsity.pth",
    patience=50,
)
print(f"\n{'=' * 60}")
print(f"Accuracy after finetune:   {final_acc}%")
print(f"{'=' * 60}\n")
