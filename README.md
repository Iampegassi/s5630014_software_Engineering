# Fashion items classification
Ojima Adaji (ID: s5630014)

## Abstract
Over the years, much research has been done in the fashion industry to automate fashion processes. Different systems, classifiers, and variations of feature extraction algorithms have been developed for fashion product classification tasks. The rise of deep learning techniques over time has replaced traditional classification techniques. In this project, two deep neural network methods for image classification of fashion items were used with a focus on their accuracy rates.

## 1. Introduction

Traditional machine learning algorithms manually extracted features from images for classification purposes, however, there were limitations in handling large datasets which led to the rise of deep learning algorithms (Xu et al. 2022). Table 1 gives a summary of traditional classification methods and techniques.


<img width="467" alt="Traditional techniques" src="https://github.com/Iampegassi/s5630014_software_Engineering/assets/149077212/3813a269-096b-4090-b952-09acbb10712a">



In recent years, CNN has been adapted as a better alternative to the traditional classification techniques. This is because they have been improved models with new network structures that use multi-convolutional layers for fashion classification with good accuracy (Nocentini et al. 2022).  Due to its advantages, two neural network models will be trained for a fashion classification task using the same datasets. The differences in their performance will be shown below. 

## 2. Literature Review

Fashion image classification can be challenging due to the unique properties and attributes of an individual product. However, the emergence of CNN a deep learning technique has been used to achieve good results in an image classification task. 


Three different CNN architectures for image classification using the fashion-mnist dataset were proposed by Bhatnagar et al. (2017). Results show an accuracy of 92.54% using a two-layered CNN with batch normalization and skip connections.

A CNN-SVM model and CNN-softmax model were used on a fashion-mnist dataset by Agarap (2019). An accuracy of 90.72% was achieved using the CNN-SVM model while the CNN-softmax model had a test accuracy of 91.86%.

Different forms of CNN were applied for image classification using the fashion-mnist dataset by Xhaferra et al. (2022).To resolve the issue of model overfitting, CNN-C1 and CNN-C2 were compared to determine the model with the best result. Results show that CNN-C2 is best with an accuracy of 93.11% compared to CNN-C1 with an accuracy of 88.95%.

## 3. Datasets

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

The datasets were further preprocessed and visualized. This is explained in the sub-heads below.

###  3.1   FashionMnistDataset Class
A PyTorch dataset class 'FashionMnistDataset' was defined. It is a CSV file containing fashion image data.
+ This class inherits from PyTorch's Dataset class.
+ The __init__ method takes a file path as input, reads the data from a CSV file located at that path using Pandas, and stores it in the self.dataset attribute.
+ The __len__ method returns the number of samples in the dataset.
+ The __getitem__ method is used to retrieve an item (image and label) at a given index from the dataset.

###  3.2  Auxillary functions
This includes some functions and they are explained below.

#### (a)  Process_image Function:
+ Takes image file name as input.
+ Opens image using Python Imaging Library (PIL).
+ Applies transformation (transform) to the image.
+ Utilizes the defined DATA_PREFIX variable.

#### (b)  Get_image_tensor Function:
+ Takes Pandas Series (pd_series) and threshold as input.
+ Extracts pixel values from the series.
+ Converts values into a PyTorch FloatTensor.
+ Reshape the tensor into a 1x28x28 format.

#### (c)  Custom Collate function

The custom_collate_fn function processes a batch of data, loads images and labels and prepares them for training. It does this by:
+ Creating an empty tensor (`image_batch_tensor`) to store the loaded images.
+ Iterates through the batch and loads each image using the `get_image_tensor` function with a specified threshold.
+ Concatenates the individual image tensors into a single batch tensor.
+ Creates a tensor (`label_batch_tensor`) containing the corresponding labels.

#### (d)  Data loading function

The function is responsible for efficiently loading and splitting the fashion image dataset into training and validation sets. It utilizes PyTorch's DataLoader for batch processing and does this by:

+ Instantiating the FashionMNISTDataset using the provided data_path.
+ Use fractions for training and validation splits (0.7, 0.3)
+ Use a batch_size of 16 in the dataloader.
+ incorporates the custom collate function (custom_collate_fn) for efficient data loading.

###  3.3  Data visualization function
This creates visual representations of the fashion image data at various stages and is further discussed with the sub-heads below.

####  (a)  Image grid and batch display functions:
image_grid: This function takes a batch of images and arranges them into a grid for visualization. It reshapes the images and concatenates them to create a grid.
show_batch: This uses the image_grid function to display a batch of images. It sets up the figure and displays the images in a grid format.

#### (b)  Display fashion items by label function:

This function displays a fashion item from the dataset based on the specified label. It takes a DataFrame, filters it based on the label, and visualizes the first image found for that label.

#### (c)  Data augmentation :
This section provides an example of data augmentation using PyTorch's transforms module. It includes code for horizontal and vertical flips, random rotations, and demonstrates how to apply these transformations to a batch of images.

## 4.  Model architecture

![Simple Neural Network (1)](https://github.com/Iampegassi/s5630014_software_Engineering/assets/149077212/3b525249-4fee-445b-b057-349623857818)

### 1. Neural Network Model with Pytorch
Neural network architecture is designed to enable fast model training with low implementation complexity (McDonnell and Vladusich 2015).

A basic PyTorch neural network model was created with a class named NeuralNetwork inherited from 'nn.Module'.

The `__init__` method, acting as the class constructor, initializes the neural network layers and activation functions when an object of the class is created.
+  `self.fc1 = nn.Linear(input_size, hidden_size)` creates a fully connected layer with `input_size` neurons and `hidden_size` output neurons.
+  `self.relu = nn.ReLU()` applies the Rectified Linear Unit (ReLU) activation function to the output.
+  `self.fc2 = nn.Linear(hidden_size, num_classes)` creates another fully connected layer with `hidden_size` input neurons and `num_classes` output neurons.
+  `self.softmax = nn.Softmax(dim=1)` then applies the softmax activation function along the second dimension, commonly used for multi-class classification.

The training loop uses a batch size of 16, with a 70-30 split for training and validation. Running for 10 epochs, it employs PyTorch's multi-layer perceptron, utilizing `train_mlp` and `val_mlp` functions for training and validation. The code logs loss, accuracy, and implements early stopping based on validation accuracy. Checkpoints are saved at a specified frequency, and input reshaping is performed for each batch. Loss and accuracy metrics are recorded using Tensorboard's 'Summary Writer'.

Results show that during training, the validation accuracy reached a maximum of 57% after 10 epochs, and the corresponding validation loss was 1.89%. However, the accuracy did not improve further so the best model using this network was not saved.


![CNN (3)](https://github.com/Iampegassi/s5630014_software_Engineering/assets/149077212/fc18ecea-0fbf-4ca5-af87-835c3f428858)

### 2. Convolutional neural network model with PyTorch

This PyTorch code defines a Convolutional Neural Network (CNN) using PyTorch's neural network module (`nn.Module`). Let's break down the key components:

#### Initialization (`__init__` method):
+  `self.conv1`, `self.conv2`, and `self.conv3` define three convolutional layers with increasing output channels, applying ReLU activation after each convolution, and followed by max-pooling.
+  `self.fc1` and `self.fc2` define two fully connected layers, applying ReLU activation after the first fully connected layer.
+ `self.softmax` is the softmax activation used for the final output layer.


#### Forward Pass (`forward` method):
+ The forward method specifies how input data should pass through the network.
+ Convolutional layers (`self.conv1`, `self.conv2`, `self.conv3`) are followed by ReLU activation and max-pooling.
+ The output is then flattened and passed through fully connected layers (`self.fc1`, `self.fc2`) with ReLU activation.
+ The final output is obtained using the softmax activation.

This CNN architecture is suitable for image classification tasks, and it includes convolutional layers for feature extraction and fully connected layers for classification. The ReLU activation functions introduce non-linearity, and the softmax activation at the output provides probabilities for different classes.



