# MNIST Data Augmentation using Gemini 2.0

MNIST (Modified National Institute of Standards and Technology) is a large dataset of 70,000 handwritten digits (0-9). It's widely used for training image processing and classification models due to its robust size and variety. However, there are still limitations. For instance, all digits are 28 x 28 pixels and greyscale. This raises concerns about its implications due to the lack of real-world noise in the dataset.

LLMs have become increasingly robust over the years, capable of generating outputs in seconds. This program demonstrates the basics of creating an API that takes in randomly selected images from MNIST and prompts *Gemini 2.0 Flash Preview Image Generation* to produce an additional image to add to the dataset, requesting geometric, elastic, and color space transformations. This simple application demonstrates the strength of utilizing LLMs to augment datasets as the retrieval of large, diverse, and high-quality datasets becomes increasingly difficult and costly.

Images from the MNIST dataset were downloaded from [a public repository](https://github.com/myleott/mnist_png). To view more about image generation with Gemini [visit here](https://ai.google.dev/gemini-api/docs/image-generation).

# Example:



*Input:*

![image](https://github.com/user-attachments/assets/b7dacd38-18be-4759-9513-e2b26d17a0de)

*Output:*

![image](https://github.com/user-attachments/assets/fe3a5e0e-15ba-4713-bd52-494381c7aa2c)



*Input:*

![image](https://github.com/user-attachments/assets/de916c69-0090-4d1d-adbf-1a882335af25)


*Output:*

![image](https://github.com/user-attachments/assets/ae251d81-0a21-4afc-ac35-cbf014553832)



*Input:*

![image](https://github.com/user-attachments/assets/4375b0dd-5262-4fe0-9768-7ff91202f12e)


*Output:*

![image](https://github.com/user-attachments/assets/dbc3e58a-c44f-415c-b889-8e2d4e51b756)



*Input:*

![image](https://github.com/user-attachments/assets/96a7f6f1-bbb2-472f-9efc-ba5bf3b6a2c7)


*Output:*

![image](https://github.com/user-attachments/assets/74240b1a-acb0-4055-b066-311beab9fb2e)



*Input:*

![image](https://github.com/user-attachments/assets/6686aa40-532b-45d1-9fa5-e71c0140256a)


*Output:*

![image](https://github.com/user-attachments/assets/77e39d22-8732-4610-b927-5f442a8b850e)



*Input:*

![image](https://github.com/user-attachments/assets/24509a4e-357c-4f52-89e8-7e8b62c1a829)


*Output:*

![image](https://github.com/user-attachments/assets/58f17930-e73e-400a-9efb-957becdafee4)



*Input:*

<<<<<<< HEAD
![image](https://github.com/user-attachments/assets/d8a65d6b-6d5c-4849-8b4b-daf6a7f83386)
=======
![image](https://github.com/user-attachments/assets/63aa5185-cc2c-41e4-97e3-8dd7a8279bbb)
>>>>>>> dec388c (edit test)


*Output:*

<<<<<<< HEAD
![image](https://github.com/user-attachments/assets/a7e6c0b5-56cb-4efc-8b61-4ac0658db293)
=======
![image](https://github.com/user-attachments/assets/4f9d6819-78e1-4bfb-9498-d7b09ced04b1)
>>>>>>> dec388c (edit test)



*Input:*

![image](https://github.com/user-attachments/assets/e47f1ceb-2678-45e6-b31d-721957b0aa4f)


*Output:*

![image](https://github.com/user-attachments/assets/c235dda9-d62b-4e67-a8d2-fdf0e1ea4be2)



*Input:*

![image](https://github.com/user-attachments/assets/78b391df-2b8b-4cbe-9f53-2c33c1a7858e)


*Output:*

![image](https://github.com/user-attachments/assets/2ff6a716-25ca-4897-91ec-b380a25ca6d3)



*Input:*

![image](https://github.com/user-attachments/assets/6c5145a7-099f-4d17-9247-7d2b61a6c559)


*Output:*

![image](https://github.com/user-attachments/assets/b22a33fa-9ac9-41eb-ae92-c82527e015b4)


