import torch
import torch.optim as optim

from models.model import Net
from utils.load_data import get_data
from train import train
from test import test

# hyperparameters
n_epochs = 3
batch_size_train = 50
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
with_synthetic = False

# load data
train_loader, test_loader = get_data(
    with_synthetic = with_synthetic,
    train_dir = 'real_data/training',
    synthetic_dir = "synthetic_data",
    test_dir = 'real_data/testing',
    batch_size_train = batch_size_train,
    batch_size_test = batch_size_test
)

# initialize model
device = torch.device("cpu")
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)

# train and test
for num in range(1, n_epochs + 1):
    train(model, device, train_loader, optimizer, num)
    test(model, device, test_loader)