from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

def get_data(with_synthetic, train_dir, synthetic_dir, test_dir, batch_size_train, batch_size_test):
    
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # original data
    real_dataset = datasets.ImageFolder(train_dir, transform=transform)

    # synthetic data
    synthetic_dataset = datasets.ImageFolder(synthetic_dir, transform=transform)

    # combined datasets if needed
    if with_synthetic:
        final_dataset = ConcatDataset([real_dataset, synthetic_dataset])
    else:
        final_dataset = ConcatDataset([real_dataset])

    train_loader = DataLoader(
        final_dataset,
        batch_size = batch_size_train,
        shuffle = True
    )

    test_loader = DataLoader(
        datasets.ImageFolder(test_dir, transform=transform),
        batch_size=batch_size_test, shuffle=False
    )

    return train_loader, test_loader