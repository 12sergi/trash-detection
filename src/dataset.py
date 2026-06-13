from torchvision import datasets, transforms

def get_datasets():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        "data/train",
        transform=train_transform
    )

    valid_dataset = datasets.ImageFolder(
        "data/valid",
        transform=valid_transform
    )

    return train_dataset, valid_dataset
