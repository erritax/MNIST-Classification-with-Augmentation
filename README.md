# Exploring Data Quality and Augmentation for MNIST Image Classification

The performance of any machine learning model depends on the data it's trained on: its size, diversity, accuracy, and completeness. Despite significant advancements in artificial intelligence in all industries, access to high-quality data has increasingly become a challenge. For instance, financial and healthcare documents often contain highly sensitive and non-public information that private tech companies are then unable to access without breaching federal laws. This project explored the effect of dataset completeness on image classification. It also performed data augmentation using LLMs and investigated the effectiveness of a simple image classification model whilst synthetic data is or isn't added to the real training dataset.

## Methods
### Dataset
MNIST (Modified National Institute of Standards and Technology) is a large dataset of 70,000 handwritten digits (0-9) widely used for training image processing and classification models due to its robust size and variety. This dataset was chosen for this experiment to allow for the exploration of a wide range of dataset sizes to have a model be tested on. 

To learn more about the data augmentation process with LLMs, [visit this](https://github.com/erritax/MNIST-Data-Augmentation?tab=readme-ov-file).

![image](https://github.com/user-attachments/assets/31d15e9e-c90c-491e-9480-4c36bb2bf3cc)
![image](https://github.com/user-attachments/assets/d7eb6e1c-c7c5-4084-8e43-0822252d2ce0)


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
![image](https://github.com/user-attachments/assets/5df814a3-0040-48d8-9622-eed81bf1acdd)
![image](https://github.com/user-attachments/assets/330b5a74-bb42-4672-af58-834db92230f2)
![image](https://github.com/user-attachments/assets/c2cd8e50-b72b-4a84-a7c0-c2c89f675f08)
![image](https://github.com/user-attachments/assets/7e4aec38-ae91-4b29-ba2c-44b9b195e67b)
![0_043](https://github.com/user-attachments/assets/606eb829-3cb8-4dc6-9228-2865f9ded606)
![0_053](https://github.com/user-attachments/assets/7439ae81-fdc9-4d74-958d-523ecd9e4e00)

Ground Truth: 1   
![image](https://github.com/user-attachments/assets/dbd3c212-cc2c-490c-b98f-f6854c71e079)
![image](https://github.com/user-attachments/assets/9c8bbbe4-f6e1-40f8-8d8a-f5dc3bedb8f8)
![1_062](https://github.com/user-attachments/assets/b6511fad-83e5-40ab-891e-c5f0c2146b5d)
![1_070](https://github.com/user-attachments/assets/8cc8a973-284c-461f-9f15-3446aa48d3c0)
![1_090](https://github.com/user-attachments/assets/764295a0-2f40-47ab-ba17-2ebeda8d4ba6)
![1_095](https://github.com/user-attachments/assets/6e2016f9-50c5-4cb5-93bc-801e4dfa4f35)

Ground Truth: 2   
![image](https://github.com/user-attachments/assets/544bde84-7d34-4183-84c0-8d368864f9c8)
![image](https://github.com/user-attachments/assets/f1a13a83-43a9-4b29-8198-6b005cd7b84e)
![2_079](https://github.com/user-attachments/assets/bf7b72e7-0560-4d3c-8945-b216aa006da3)
![2_066](https://github.com/user-attachments/assets/8c48fcf8-753b-4fdb-8bf5-2656d95126c2)
![2_061](https://github.com/user-attachments/assets/02dbc5d4-a590-4bd4-8d41-bba3032a7944)
![2_010](https://github.com/user-attachments/assets/6b908260-9ce8-4d69-9d28-a44ecd34d66a)

Ground Truth: 3   
![image](https://github.com/user-attachments/assets/2e237e9e-8e45-4547-ad39-383351e6e090)
![image](https://github.com/user-attachments/assets/ca455122-9e35-46d7-933c-b7aca1fa4ff2)
![3_073](https://github.com/user-attachments/assets/42970357-16ff-4194-8962-436504722f27)
![3_038](https://github.com/user-attachments/assets/f8952863-4b3d-45fc-967d-e42e09fa590c)
![3_025](https://github.com/user-attachments/assets/9df2a8c5-69ca-4eac-8d38-47a6a0e43790)
![3_017](https://github.com/user-attachments/assets/5a615672-320a-4f2b-acde-06fef04abcdb)


Ground Truth: 4   
![4_060](https://github.com/user-attachments/assets/04fe2dae-dc08-4380-a854-89e5a043558b)
![4_061](https://github.com/user-attachments/assets/7c64d825-d383-4124-8233-4b59ba3b6468)
![4_050](https://github.com/user-attachments/assets/26887d38-a1dd-4eef-82ee-8420e1e74c04)
![4_036](https://github.com/user-attachments/assets/529a710d-c82b-4854-baa6-73247ab1b035)
![image](https://github.com/user-attachments/assets/82470927-26ea-4295-b6c3-16b4d3ab97aa)
![image](https://github.com/user-attachments/assets/0365b4cb-28e6-487f-bcc5-5563ec079dc5)

Ground Truth: 5   
![5_085](https://github.com/user-attachments/assets/429b00fb-b02d-4500-8138-0fced869abef)
![5_099](https://github.com/user-attachments/assets/db1cfb6e-2314-45d7-935d-be94f367ac38)
![5_040](https://github.com/user-attachments/assets/995c9b0e-4166-4121-96e5-80dfad22bb3e)
![5_026](https://github.com/user-attachments/assets/8f83aedc-7c04-4a04-bb8e-f46ea82dee9f)
![image](https://github.com/user-attachments/assets/8b5a50cc-4ed3-4348-8014-7c01fc75bb47)
![image](https://github.com/user-attachments/assets/972d973a-524f-4bef-90c1-f2e67022d49e)

Ground Truth: 6   
![6_098](https://github.com/user-attachments/assets/e9548153-1043-4e10-bfb5-cc6fe4d1a3b5)
![6_023](https://github.com/user-attachments/assets/96d8fd5a-015e-4725-9ab3-a53d10049d44)
![6_052](https://github.com/user-attachments/assets/1cd852aa-13a5-4402-a46e-81ea928dada3)
![6_014](https://github.com/user-attachments/assets/df43edf4-a1b7-4685-820f-943015d8f3ab)
![image](https://github.com/user-attachments/assets/4f1728cb-2002-4796-a8e4-8c3137f23d61)
![image](https://github.com/user-attachments/assets/1237a7ac-c2ce-4d23-8442-fe13e0854e64)

Ground Truth: 7   
![7_094](https://github.com/user-attachments/assets/6e493989-4fe5-4097-9d80-649a21b334ea)
![7_057](https://github.com/user-attachments/assets/b590bf05-10d3-42eb-87af-b71656527411)
![7_028](https://github.com/user-attachments/assets/787b5f0e-a2bb-4fb1-9675-c9ae148d606e)
![7_000](https://github.com/user-attachments/assets/ce042205-0e53-4b48-af6e-d89ae2ac148b)
![image](https://github.com/user-attachments/assets/41005092-4908-403e-820a-ac5aa1a7efa2)
![image](https://github.com/user-attachments/assets/9d954078-1e72-424c-b044-cd133bc7ad3d)

Ground Truth: 8   
![8_072](https://github.com/user-attachments/assets/95e974fb-de77-470f-9b63-f4d087ace96c)
![8_035](https://github.com/user-attachments/assets/3b5d3ccb-d3d2-40ba-a8e5-497c42b61452)
![8_020](https://github.com/user-attachments/assets/e467b392-85c1-4a7b-9952-5fe9eec1c4b4)
![8_002](https://github.com/user-attachments/assets/d7d859e3-17d5-46a2-b19a-cf92533cd129)
![image](https://github.com/user-attachments/assets/d6d95962-6348-4013-9bc6-839d23b157fe)
![image](https://github.com/user-attachments/assets/ac7c2fd1-31bb-473d-9976-e55c0a9d2713)

Ground Truth: 9   
![9_048](https://github.com/user-attachments/assets/06b2df27-6449-4124-b2bd-7cc5560224f0)
![9_090](https://github.com/user-attachments/assets/87b13e54-1da3-48d1-aaff-78610b5cf847)
![9_098](https://github.com/user-attachments/assets/919a28e2-4099-4c6c-afc7-f4bd21f92565)
![9_031](https://github.com/user-attachments/assets/5112f361-f701-4eb6-855f-0e93c510d82e)
![image](https://github.com/user-attachments/assets/d419ae51-a9eb-40fb-8493-ca59e38113c2)
![image](https://github.com/user-attachments/assets/4bac8dad-9453-4f16-a59e-23ae687bf8a3)
