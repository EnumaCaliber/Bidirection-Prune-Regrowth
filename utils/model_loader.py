from models import *

def model_loader(model_string, DEVICE):
    if DEVICE == None:
        raise ValueError('No cuda device!')
    if model_string == 'vgg16':
        model = VGG('VGG16').to(DEVICE)

    elif model_string == 'resnet20':
        model = ResNet20().to(DEVICE)

    elif model_string == 'resnet32':
        model = ResNet32().to(DEVICE)

    elif model_string == 'resnet44':
        model = ResNet44().to(DEVICE)

    elif model_string == 'resnet56':
        model = ResNet56().to(DEVICE)

    elif model_string == 'resnet18':
        model = ResNet18().to(DEVICE)

    elif model_string == 'densenet':
        model = DenseNet121().to(DEVICE)

    elif model_string == 'effnet':
        model = EfficientNetB0().to(DEVICE)

    elif model_string == 'alexnet':
        model = AlexNet().to(DEVICE)

    elif model_string == 'googlenet':
        model = GoogLeNet().to(DEVICE)

    else:
        raise ValueError('Unknown model')
    # prune_weights_reparam(model)
    return model