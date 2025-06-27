# Exploring Data Quality and Augmentation for MNIST Classification

The performance of any machine learning model depends on the data it's trained on: its size, diversity, accuracy, uniqueness, etc. Despite huge advancements in artificial intelligence in all industries, access to high quality datasets has become an increasing challenge. For instance, financial and healthcare documents often contain highly sensitive and non-public information that companies are then unable to use without breaching federal laws. This project explored the effect of dataset size on image classification. It also performed data augmentation using LLMs and explored the effectiveness of the ML model whilst trained on both the real and synthetic data to explore discrepencies.

## Methods
### Dataset
MNIST (Modified National Institute of Standards and Technology) is a large dataset of 70,000 handwritten digits (0-9) widely used for training image processing and classification models due to its robust size and variety. This dataset was chosen for this experiment to allow for the exploration of a wide range of dataset sizes to have a model be tested on. 

To learn more about the data augmentation process with LLMs, [visit this](https://github.com/erritax/MNIST-Data-Augmentation?tab=readme-ov-file).

### The Model
A simple convolutional neural network (CNN) was programmed using PyTorch. Set hyperparameters included:
- number of epochs = 3
- training batch size = 50
- testing batch size = 1000
- learning rate = 0.01
- momentum = 0.5

## Results

![image](https://github.com/user-attachments/assets/9320b9e0-ae60-48d5-b8f4-9a95c0af16ed)

![image](https://github.com/user-attachments/assets/47bebe23-0955-49f6-8ab0-c3e0775a19e7)
