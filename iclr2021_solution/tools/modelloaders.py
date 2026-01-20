import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torchvision.models as tmodels
from functools import partial
from iclr2021_solution.tools.models import *
from iclr2021_solution.tools.pruners import prune_weights_reparam

def model_and_opt_loader(model_string,DEVICE):
    if DEVICE == None:
        raise ValueError('No cuda device!')
    if model_string == 'vgg16':
        model = VGG16().to(DEVICE)
        amount = 0.4
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 40000, # 40000 for iterative, 400000 for one-shot
            "scheduler": None
        }
    elif model_string == 'resnet20':
        model = ResNet20().to(DEVICE)
        amount = 0.985
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 400000, # 40000 for iterative, 400000 for one-shot
            "scheduler": None
        }
    # elif model_string == 'resnet32':
    #     model = ResNet32().to(DEVICE)
    #     amount = 0.94
    #     batch_size = 100
    #     opt_pre = {
    #         "optimizer": partial(optim.AdamW,lr=0.0003),
    #         "steps": 50000,
    #         "scheduler": None
    #     }
    #     opt_post = {
    #         "optimizer": partial(optim.AdamW,lr=0.0003),
    #         "steps": 40000,
    #         "scheduler": None
    #     }
    # elif model_string == 'resnet44':
    #     model = ResNet44().to(DEVICE)
    #     amount = 0.95
    #     batch_size = 100
    #     opt_pre = {
    #         "optimizer": partial(optim.AdamW,lr=0.0003),
    #         "steps": 50000,
    #         "scheduler": None
    #     }
    #     opt_post = {
    #         "optimizer": partial(optim.AdamW,lr=0.0003),
    #         "steps": 40000,
    #         "scheduler": None
    #     }
    # elif model_string == 'resnet56':
    #     model = ResNet56().to(DEVICE)
    #     amount = 0.95
    #     batch_size = 100
    #     opt_pre = {
    #         "optimizer": partial(optim.AdamW,lr=0.0003),
    #         "steps": 50000,
    #         "scheduler": None
    #     }
    #     opt_post = {
    #         "optimizer": partial(optim.AdamW,lr=0.0003),
    #         "steps": 40000,
    #         "scheduler": None
    #     }
    # elif model_string == 'resnet18':
    #     model = ResNet18().to(DEVICE)
    #     amount = 0.4
    #     batch_size = 100
    #     opt_pre = {
    #         "optimizer": partial(optim.AdamW,lr=0.0003),
    #         "steps": 50000,
    #         "scheduler": None
    #     }
    #     opt_post = {
    #         "optimizer": partial(optim.AdamW,lr=0.0003),
    #         "steps": 40000,
    #         "scheduler": None
    #     }
    elif model_string == 'densenet':
        model = DenseNet121().to(DEVICE)
        amount = 0.4
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 80000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 60000, # 60000 for iterative, 600000 for one-shot
            "scheduler": None
        }
    elif model_string == 'effnet':
        model = EfficientNetB0().to(DEVICE)
        amount = 0.98
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 40000, # 40000 for iterative, 400000 for one-shot
            "scheduler": None
        }
    elif model_string == 'alexnet':
        model = AlexNet_CIFAR10().to(DEVICE)
        amount = 0.95
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 400000, # 40000 for iterative, 400000 for one-shot
            "scheduler": None
        }

    elif model_string == 'alexnet_imagenet':
        model = AlexNet_ImageNet().to(DEVICE)
        amount = 0.95
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 400000, # 40000 for iterative, 400000 for one-shot
            "scheduler": None
        }
    # elif model_string == 'googlenet':
    #     model = GoogleNet().to(DEVICE)
    #     amount = 0.98  # Moderate sparsity for Inception architecture
    #     batch_size = 100
    #     opt_pre = {
    #         "optimizer": partial(optim.AdamW,lr=0.0003),
    #         "steps": 50000,
    #         "scheduler": None
    #     }
    #     opt_post = {
    #         "optimizer": partial(optim.AdamW,lr=0.0003),
    #         "steps": 40000,
    #         "scheduler": None
    #     }
    else:
        raise ValueError('Unknown model')
    prune_weights_reparam(model)
    return model,amount,batch_size,opt_pre,opt_post