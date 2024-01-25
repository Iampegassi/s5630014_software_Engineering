# Fashion items classification
Ojima Adaji (ID: s5630014)

## Abstract
Over the years, much research has been done in the fashion industry to automate fashion processes. Different systems, classifiers, and variations of feature extraction algorithms have been developed for fashion product classification tasks. The rise of deep learning techniques over time has replaced traditional classification techniques. In this project, two deep neural network methods for image classification of fashion items were used with a focus on their accuracy rates.

## Introduction

Traditional machine learning algorithms manually extracted features from images for classification purposes, however, there were limitations in handling large datasets which led to the rise of deep learning algorithms (Xu et al. 2022). Table 1 gives a summary of traditional classification methods and techniques.


<img width="467" alt="Traditional techniques" src="https://github.com/Iampegassi/s5630014_software_Engineering/assets/149077212/3813a269-096b-4090-b952-09acbb10712a">



In recent years, CNN has been adapted as a better alternative to the traditional classification techniques. This is because they have been improved models with new network structures that use multi-convolutional layers for fashion classification with good accuracy (Nocentini et al. 2022).  Due to its advantages, two neural network models will be trained for a fashion classification task using the same datasets. The differences in their performance will be shown below. 

## Literature Review

Fashion classification can be challenging due to the unique properties and attributes of an individual product. However, the emergence of CNN a deep learning technique has been used to achieve good results in an image classification task. Three different CNN architectures for image classification using the fashion-mnist dataset were proposed by Bhatnagar et al. (2017). Results show an accuracy of 92.54% using a two-layered CNN with batch normalization and skip connections.

A CNN-SVM model and CNN-softmax model were used on a fashion-mnist dataset by Agarap (2019). An accuracy of 90.72% was achieved using the CNN-SVM model while the CNN-softmax model had a test accuracy of 91.86%.


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

![Simple Neural Network (1)](https://github.com/Iampegassi/s5630014_software_Engineering/assets/149077212/3b525249-4fee-445b-b057-349623857818)

#### 1. Neural Network Model with Pytorch
Neural network architecture is designed to enable fast model training with low implementation complexity (McDonnell and Vladusich 2015).

A basic PyTorch neural network model was created, featuring two fully connected layers separated by a ReLU activation. The final layer utilizes a softmax activation, suitable for multiclass classification. The 'NeuralNetwork' class encapsulates the network architecture, with the constructor ('init') initializing layers, and the 'forward' method describing the forward pass.

For the training loop, using a batch size of 16, the dataset was split into 70% for training and 30% for validation. A multi-layer perceptron by Pytorch was used. It incorporates early stopping to prevent overfitting, checkpoints are saved periodically and utilizes Tensorboard for training visualisation. Input reshaping is performed for each batch, and loss and accuracy metrics are logged using Tensorboard ‘Summary Writer’.

The 'val_mlp' function evaluates the model on the evaluation datasets and loss and accuracy metrics are computed.

Results show that during training, the validation accuracy reached a maximum of approximately 46% after 10 epochs, and the corresponding validation loss was around 2.26. However, the accuracy did not improve further. The training process was completed, and the best model state was saved.


![CNN (3)](https://github.com/Iampegassi/s5630014_software_Engineering/assets/149077212/fc18ecea-0fbf-4ca5-af87-835c3f428858)

#### 2. Convolutional neural network model with PyTorch

This PyTorch code defines a concise Convolutional Neural Network (CNN) class with convolutional and fully connected layers. It initializes an instance of the CNN class and sets up a neural network using another class, NeuralNetwork. The constructor (init method) includes a num_classes parameter for classification output. Three convolutional layers (conv1, conv2, conv3) with ReLU activations and two fully connected layers (fc1, fc2) with ReLU and softmax activations are specified. The 'forward' method defines the forward pass



