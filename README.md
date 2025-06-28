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

A notable reason for this explanation could be the quality of the synthetic data generated. It was noted that the natural language prompt fed to the LLM for data augmentation was simple, and only ten example images were provided. One of the most prevalent issues was that because a singular grid of digits were provided rather than 10 separate images, despite indicating in the prompt, the image generated included multiple digits on one image. 

## Examples of Noisy Data

Ground Truth: 0   
![0_000](https://github.com/user-attachments/assets/91a619fb-c6c5-4d34-a661-ab3e88cd9bb4)
![0_022](https://github.com/user-attachments/assets/704cd9fd-1355-4600-b4c9-f730fbef2f86)
![0_091](https://github.com/user-attachments/assets/45ea550c-9484-40b7-8d49-08462b0d52a2)
![0_026](https://github.com/user-attachments/assets/ad703249-ea5d-4ca6-ab9b-30b549976b24)
![0_043](https://github.com/user-attachments/assets/606eb829-3cb8-4dc6-9228-2865f9ded606)
![0_053](https://github.com/user-attachments/assets/7439ae81-fdc9-4d74-958d-523ecd9e4e00)

Ground Truth: 1   
![1_024](https://github.com/user-attachments/assets/c3bdb69b-1793-41fa-8514-edb82713eace)
![1_049](https://github.com/user-attachments/assets/ed08495d-2dad-4fd8-bd32-40523c0f911f)
![1_062](https://github.com/user-attachments/assets/b6511fad-83e5-40ab-891e-c5f0c2146b5d)
![1_070](https://github.com/user-attachments/assets/8cc8a973-284c-461f-9f15-3446aa48d3c0)
![1_090](https://github.com/user-attachments/assets/764295a0-2f40-47ab-ba17-2ebeda8d4ba6)
![1_011](https://github.com/user-attachments/assets/1a99040b-445e-41b5-93ad-24a0fb644c92)

Ground Truth: 2   
![2_010](https://github.com/user-attachments/assets/0cc39ea8-7525-46a3-b286-14ebaafca65b)
![2_087](https://github.com/user-attachments/assets/65106111-6701-4b02-8677-8447f8667a55)
![2_079](https://github.com/user-attachments/assets/bf7b72e7-0560-4d3c-8945-b216aa006da3)
![2_066](https://github.com/user-attachments/assets/8c48fcf8-753b-4fdb-8bf5-2656d95126c2)
![2_061](https://github.com/user-attachments/assets/02dbc5d4-a590-4bd4-8d41-bba3032a7944)
![2_010](https://github.com/user-attachments/assets/6b908260-9ce8-4d69-9d28-a44ecd34d66a)

Ground Truth: 3   
![3_029](https://github.com/user-attachments/assets/3a60eb1c-15df-47fc-a7e0-f187eed64d99)
![3_000](https://github.com/user-attachments/assets/0f6a0257-6701-40e8-9152-a02a739b2d6c)
![3_073](https://github.com/user-attachments/assets/42970357-16ff-4194-8962-436504722f27)
![3_038](https://github.com/user-attachments/assets/f8952863-4b3d-45fc-967d-e42e09fa590c)
![3_025](https://github.com/user-attachments/assets/9df2a8c5-69ca-4eac-8d38-47a6a0e43790)
![3_017](https://github.com/user-attachments/assets/5a615672-320a-4f2b-acde-06fef04abcdb)


Ground Truth: 4   
![4_060](https://github.com/user-attachments/assets/04fe2dae-dc08-4380-a854-89e5a043558b)
![4_061](https://github.com/user-attachments/assets/7c64d825-d383-4124-8233-4b59ba3b6468)
![4_050](https://github.com/user-attachments/assets/26887d38-a1dd-4eef-82ee-8420e1e74c04)
![4_036](https://github.com/user-attachments/assets/529a710d-c82b-4854-baa6-73247ab1b035)
![4_049](https://github.com/user-attachments/assets/e4f6b656-967c-4f0e-8b01-35b91ddaca3a)
![4_092](https://github.com/user-attachments/assets/5a8e0479-892d-4303-b671-fa104226ed14)

Ground Truth: 5   
![5_085](https://github.com/user-attachments/assets/429b00fb-b02d-4500-8138-0fced869abef)
![5_099](https://github.com/user-attachments/assets/db1cfb6e-2314-45d7-935d-be94f367ac38)
![5_040](https://github.com/user-attachments/assets/995c9b0e-4166-4121-96e5-80dfad22bb3e)
![5_026](https://github.com/user-attachments/assets/8f83aedc-7c04-4a04-bb8e-f46ea82dee9f)
![5_080](https://github.com/user-attachments/assets/66d1e869-6822-4041-a43a-088e63dae0ba)
![5_024](https://github.com/user-attachments/assets/bf5c942b-4713-4891-a6c2-b867553dce1b)

Ground Truth: 6   
![6_098](https://github.com/user-attachments/assets/e9548153-1043-4e10-bfb5-cc6fe4d1a3b5)
![6_023](https://github.com/user-attachments/assets/96d8fd5a-015e-4725-9ab3-a53d10049d44)
![6_052](https://github.com/user-attachments/assets/1cd852aa-13a5-4402-a46e-81ea928dada3)
![6_014](https://github.com/user-attachments/assets/df43edf4-a1b7-4685-820f-943015d8f3ab)
![6_079](https://github.com/user-attachments/assets/03c88a49-f2b4-4170-836e-ef05618dbb07)
![6_058](https://github.com/user-attachments/assets/ace49f06-a8b5-4246-b812-0f2e95ae5eae)

Ground Truth: 7   
![7_094](https://github.com/user-attachments/assets/6e493989-4fe5-4097-9d80-649a21b334ea)
![7_057](https://github.com/user-attachments/assets/b590bf05-10d3-42eb-87af-b71656527411)
![7_028](https://github.com/user-attachments/assets/787b5f0e-a2bb-4fb1-9675-c9ae148d606e)
![7_000](https://github.com/user-attachments/assets/ce042205-0e53-4b48-af6e-d89ae2ac148b)
![7_004](https://github.com/user-attachments/assets/88e37680-230f-49b5-bb32-a15020a8d955)
![7_022](https://github.com/user-attachments/assets/cfa2b734-766e-47b5-b1b6-7b8f4af7f20d)

Ground Truth: 8   
![8_072](https://github.com/user-attachments/assets/95e974fb-de77-470f-9b63-f4d087ace96c)
![8_035](https://github.com/user-attachments/assets/3b5d3ccb-d3d2-40ba-a8e5-497c42b61452)
![8_020](https://github.com/user-attachments/assets/e467b392-85c1-4a7b-9952-5fe9eec1c4b4)
![8_002](https://github.com/user-attachments/assets/d7d859e3-17d5-46a2-b19a-cf92533cd129)
![8_015](https://github.com/user-attachments/assets/05d8e188-ac8c-40c7-a17c-8e9ebcea4f5b)
![8_081](https://github.com/user-attachments/assets/e089611a-e53e-4650-9999-11b029cc6e2b)

Ground Truth: 9   
![9_048](https://github.com/user-attachments/assets/06b2df27-6449-4124-b2bd-7cc5560224f0)
![9_090](https://github.com/user-attachments/assets/87b13e54-1da3-48d1-aaff-78610b5cf847)
![9_098](https://github.com/user-attachments/assets/919a28e2-4099-4c6c-afc7-f4bd21f92565)
![9_031](https://github.com/user-attachments/assets/5112f361-f701-4eb6-855f-0e93c510d82e)
![9_086](https://github.com/user-attachments/assets/95dc717c-3464-4aaa-9bcd-4a8f36d732c9)
