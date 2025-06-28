import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# class labels
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# transform images
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# plot for real data
real_dataset = datasets.ImageFolder('real_data/training', transform = transform)
real_loader = DataLoader(real_dataset, batch_size = 10, shuffle = True)

real_images, real_labels = next(iter(real_loader))

fig, ax = plt.subplots(1, 10, figsize = (16, 4))
fig.suptitle('Real Data Samples', fontsize = 16, y = 0.85)
for i in range(10):
    ax[i].imshow(real_images[i].squeeze(), cmap = 'gray')
    ax[i].set_title(f'{classes[real_labels[i]]}')
    ax[i].axis('off')
plt.show()

# plot for synthetic data
synthetic_dataset = datasets.ImageFolder('synthetic_data', transform = transform)
synthetic_loader = DataLoader(synthetic_dataset, batch_size = 10, shuffle = True)

synthetic_images, synthetic_labels = next(iter(synthetic_loader))

fig, ax = plt.subplots(1, 10, figsize = (16, 4))
fig.suptitle('Synthetic Data Samples', fontsize = 16, y = 0.85)
for i in range(10):
    ax[i].imshow(synthetic_images[i].squeeze(), cmap = 'gray')
    ax[i].set_title(f'{classes[synthetic_labels[i]]}')
    ax[i].axis('off')
plt.show()