import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from transformers import get_cosine_schedule_with_warmup
from utils.data_loader import data_loader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    default='./structured_rl_ckpts/effnet/structured_oneshot/sp90-86/best_model_rwd-3.12pp.pth')
parser.add_argument('--save_dir', type=str,
                    default='./structured_rl_ckpts/effnet/structured_oneshot/fullfinetune_sp90-86')
parser.add_argument('--sparsity', type=str, default='0.89')
parser.add_argument('--epochs',   type=int, default=400)
parser.add_argument('--lr',       type=float, default=3e-4)
parser.add_argument('--patience', type=int, default=50)
args = parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def full_finetune(model, train_loader, test_loader, device,
                  epochs=400, lr=3e-4, save_dir=None,
                  sparsity='0.952', patience=50):
    print(f"\n{'=' * 70}")
    print("Full Finetune")
    print(f"{'=' * 70}\n")

    import os
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    steps_per_epoch = len(train_loader)
    total_steps     = epochs * steps_per_epoch
    warmup_steps    = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_acc, best_epoch, no_improve = 0.0, 0, 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(model(x), y).backward()
            optimizer.step()
            scheduler.step()

        # ── Evaluate ──────────────────────────────────────────────────────────
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                _, pred = model(x).max(1)
                total   += y.size(0)
                correct += pred.eq(y).sum().item()
        acc = 100.0 * correct / total

        # ── Best ──────────────────────────────────────────────────────────────
        if acc > best_acc:
            best_acc, best_epoch, no_improve = acc, epoch + 1, 0
            torch.save(model.state_dict(),
                       f'{save_dir}/final_model_{sparsity}.pth')
        else:
            no_improve += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Acc={acc:.2f}%  Best={best_acc:.2f}%  NoImp={no_improve}")

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(),
                       f'{save_dir}/epoch{epoch+1}.pth')

        # ── Early stop ────────────────────────────────────────────────────────
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(f"\nBest: {best_acc:.2f}%  (epoch {best_epoch})")
    return best_acc


# ── Main ──────────────────────────────────────────────────────────────────────
set_seed()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader, _, test_loader = data_loader(data_dir='./data')

# RL ckpt 存的是整模型，直接 load
model = torch.load(args.model_path, map_location=device, weights_only=False)
model = model.to(device)

final_acc = full_finetune(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    epochs=args.epochs,
    lr=args.lr,
    save_dir=args.save_dir,
    sparsity=args.sparsity,
    patience=args.patience,
)

print(f"\n{'=' * 60}")
print(f"Final Acc: {final_acc}%")
print(f"{'=' * 60}")