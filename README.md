# Exploring Data Quality and Augmentation for MNIST Image Classification

The performance of any machine learning model depends on the data it's trained on: its size, diversity, accuracy, and completeness. Despite significant advancements in artificial intelligence in all industries, access to high-quality data has increasingly become a challenge. For instance, financial and healthcare documents often contain highly sensitive and non-public information that private tech companies are then unable to access without breaching federal laws. This project explored the effect of dataset completeness on image classification. It also performed data augmentation using LLMs and investigated the effectiveness of a simple image classification model whilst synthetic data is or isn't added to the real training dataset.

## Methods
### Dataset
MNIST (Modified National Institute of Standards and Technology) is a large dataset of 70,000 handwritten digits (0-9) widely used for training image processing and classification models due to its robust size and variety. This dataset was chosen for this experiment to allow for the exploration of a wide range of dataset sizes to have a model be tested on. 

To learn more about the data augmentation process with LLMs, [visit this](https://github.com/erritax/MNIST-Data-Augmentation?tab=readme-ov-file).

![image](https://github.com/user-attachments/assets/9dca66eb-7db3-445c-b64c-0a8a93c1af9c)
![image](https://github.com/user-attachments/assets/6299acc6-9968-42db-9d1a-76ced6dc0f02)


### The Model
A simple convolutional neural network (CNN) was programmed using PyTorch. The model consisted of two layers, which were then flattened. Set hyperparameters included:
- number of epochs = 3
- training batch size = 50
- testing batch size = 1000
- learning rate = 0.01
- momentum = 0.5

## Results
The classification model was trained on real MNIST images as well as with synthetically generated images if indicated. The model was tested with the testing dataset in MNIST.

The following figure depicts the results of various dataset sizes and accuracy scores when tested with and without synthetic data. Without fail, when synthetic data was added to the dataset, accuracy was improved.

![image](https://github.com/user-attachments/assets/9320b9e0-ae60-48d5-b8f4-9a95c0af16ed)

An interesting metric when exploring the effects of the synthetic data is losses. Shown below, there's no clear indication that the addition of synthetic data decreased average losses, meaning that the testing results often deviate more from the ground truth.

![image](https://github.com/user-attachments/assets/47bebe23-0955-49f6-8ab0-c3e0775a19e7)

A notable reason for this explanation could be the quality of the synthetic data generated. It was noted that the natural language prompt fed to the LLM for data augmentation was simple, and only ten example images were provided. One of the most prevalent issues was that because a singular grid of digits were provided rather than 10 separate images, despite indicating in the prompt, the image generated included multiple digits on one image. Examples of poorly generated images are shown below:

Ground Truth: 0   
![image](https://github.com/user-attachments/assets/330b5a74-bb42-4672-af58-834db92230f2)
![image](https://github.com/user-attachments/assets/c2cd8e50-b72b-4a84-a7c0-c2c89f675f08)

Ground Truth: 1   
![image](https://github.com/user-attachments/assets/dbd3c212-cc2c-490c-b98f-f6854c71e079)
![image](https://github.com/user-attachments/assets/9c8bbbe4-f6e1-40f8-8d8a-f5dc3bedb8f8)

Ground Truth: 2   
![image](https://github.com/user-attachments/assets/544bde84-7d34-4183-84c0-8d368864f9c8)
![image](https://github.com/user-attachments/assets/f1a13a83-43a9-4b29-8198-6b005cd7b84e)

Ground Truth: 3   
![image](https://github.com/user-attachments/assets/2e237e9e-8e45-4547-ad39-383351e6e090)
![image](https://github.com/user-attachments/assets/ca455122-9e35-46d7-933c-b7aca1fa4ff2)

Ground Truth: 4   
![image](https://github.com/user-attachments/assets/82470927-26ea-4295-b6c3-16b4d3ab97aa)
![image](https://github.com/user-attachments/assets/0365b4cb-28e6-487f-bcc5-5563ec079dc5)

Ground Truth: 5   
![image](https://github.com/user-attachments/assets/8b5a50cc-4ed3-4348-8014-7c01fc75bb47)
![image](https://github.com/user-attachments/assets/972d973a-524f-4bef-90c1-f2e67022d49e)

Ground Truth: 6   
![image](https://github.com/user-attachments/assets/4f1728cb-2002-4796-a8e4-8c3137f23d61)
![image](https://github.com/user-attachments/assets/1237a7ac-c2ce-4d23-8442-fe13e0854e64)

Ground Truth: 7   
![image](https://github.com/user-attachments/assets/41005092-4908-403e-820a-ac5aa1a7efa2)
![image](https://github.com/user-attachments/assets/9d954078-1e72-424c-b044-cd133bc7ad3d)

Ground Truth: 8   
![image](https://github.com/user-attachments/assets/d6d95962-6348-4013-9bc6-839d23b157fe)
![image](https://github.com/user-attachments/assets/ac7c2fd1-31bb-473d-9976-e55c0a9d2713)

Ground Truth: 9   
![image](https://github.com/user-attachments/assets/d419ae51-a9eb-40fb-8493-ca59e38113c2)
![image](https://github.com/user-attachments/assets/4bac8dad-9453-4f16-a59e-23ae687bf8a3)
