import torch
from utils.model_loader import model_loader

MODELS = {
    "vgg16": {
        "pretrained": "vgg16/checkpoint/pretrain_vgg16_ckpt.pth"
    }
}


def load_pretrained(ckpt_path, device, model_name):
    model = model_loader(model_name, device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    return model.eval()


def count_params(model):
    total = 0
    for name, param in model.named_parameters():
        if 'bias' in name:  # 跳过bias
            continue
        num = param.numel()
        print(f"{name}: {num:,}")
        total += num
    print(f"\nTotal (without bias): {total:,}")
    return total


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 统计pretrained模型
    model = load_pretrained(MODELS["vgg16"]["pretrained"], device, "vgg16")
    count_params(model)
