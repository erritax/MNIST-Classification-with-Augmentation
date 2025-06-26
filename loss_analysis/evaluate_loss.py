import json
import numpy as np
import torch
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision import datasets, transforms

from models.model import Net
from train import train
from test import test
from utils.load_data import get_data

# get subset of data based on desired size
def get_subset(dataset, n_per_class):
    targets = np.array([label for _, label in dataset.samples])
    indices = []
    for digit in range(10):
        digit_indices = np.where(targets == digit)[0]
        selected = np.random.choice(digit_indices, n_per_class, replace=False)
        indices.extend(selected)
    return Subset(dataset, indices)

# runner
def run_evaluation(dataset_sizes, use_synthetic):

    train_dir = 'real_data/training'
    synthetic_dir = 'synthetic_data'
    test_dir = 'real_data/testing'

    epochs = 3
    batch_size_train = 50
    batch_size_test = 1000

    results = {}
    loss_history = {}

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    real_dataset = datasets.ImageFolder(train_dir, transform = transform)

    device = torch.device('cpu')

    _, test_loader = get_data(
        with_synthetic = False,
        train_dir = train_dir,
        synthetic_dir = synthetic_dir,
        test_dir = test_dir,
        batch_size_train = batch_size_train,
        batch_size_test = batch_size_test
    )
    
    # for each dataset size to be analyzed
    for size in dataset_sizes:
        n_per_class = size // 10
        subset = get_subset(real_dataset, n_per_class)

        if use_synthetic:
            synthetic_dataset = datasets.ImageFolder(synthetic_dir, transform = transform)
            train_dataset = ConcatDataset([subset, synthetic_dataset])
        else:
            train_dataset = subset

        train_loader = DataLoader(train_dataset, batch_size = batch_size_train, shuffle = True)

        model = Net().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)

        print(f"\nTraining with {size} real images {'+ synthetic' if use_synthetic else ''}")

        # track losses
        losses = []

        for epoch in range(1, epochs + 1):
            avg_loss = train(model, device, train_loader, optimizer, epoch)
            acc = test(model, device, test_loader)
            losses.append(avg_loss)

        results[size] = acc
        loss_history[size] = losses

    return results, loss_history

# save results in json in results dir
def save_loss(loss_results, filename):
    with open(f'results/{filename}.json', 'w') as f:
        json.dump(loss_results, f, indent = 4)