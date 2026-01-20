from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

def data_loader(data_dir, val_split=0.2):
    
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform_test, download=True)

    # Split train dataset into train and validation sets
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=15)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=15)
    test_loader = DataLoader(test_dataset, batch_size=100, num_workers=15)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, val_loader, test_loader