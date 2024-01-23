# Fashion items classification
Ojima Adaji (ID: s5630014)

## Abstract
Over the years, much research has been done in the fashion industry to automate fashion processes. Different systems, classifiers, and variations of feature extraction algorithms have been developed for fashion product classification tasks. The rise of deep learning techniques over time has replaced traditional classification techniques. In this project, two deep neural network methods for image classification of fashion items were used with a focus on their accuracy rates.

## Detailed Description

Traditional machine learning algorithms manually extracted features from images for classification purposes, however, there were limitations in handling large datasets which led to the rise of deep learning algorithms (Xu et al. 2022). Table 1 gives a summary of traditional classification methods and techniques.


<img width="467" alt="Traditional techniques" src="https://github.com/Iampegassi/s5630014_software_Engineering/assets/149077212/3813a269-096b-4090-b952-09acbb10712a">




In recent years, CNN has been adapted as a better alternative to the traditional classification techniques. This is because they have been improved models with new network structures that use multi-convolutional layers for fashion classification with good accuracy (Nocentini et al. 2022).  Due to its advantages, two neural network models will be trained for a fashion classification task using the same datasets. The differences in their performance will be shown below. 


### Datasets

The fashion-mnist dataset was used which consists of 60,000 training images and 10,000 testing images. Every image in the dataset belongs to one of the 10 classes:

0- T-shirt/Top

1- Trouser

2- Pullover

3- Dress

4- Coat

5- Sandal

6- Shirt

7- Sneaker

8- Bag

9- Ankle boot

Each image in the dataset is a 28x28 pixel grayscale image. Each row is a separate image, column 1 is the class label. The other columns are pixel numbers which are 784 in total. Each value is the darkness of the pixel (1 to 255). [Get Here](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

### Model architecture
