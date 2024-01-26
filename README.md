# Fashion items classification
Ojima Adaji (ID: s5630014)

## Abstract
Over the years, much research has been done in the fashion industry to automate fashion processes. Different systems, classifiers, and variations of feature extraction algorithms have been developed for fashion product classification tasks. The rise of deep learning techniques over time has replaced traditional classification techniques. In this project, two deep neural network methods for image classification of fashion items were compared regarding their accuracy rates.

## 1. Introduction

Traditional machine learning algorithms manually extracted features from images for classification purposes, however, there were limitations in handling large datasets which led to the rise of deep learning algorithms (Xu et al. 2022). Table 1 gives a summary of traditional classification methods and techniques.


<img width="467" alt="Traditional techniques" src="https://github.com/Iampegassi/s5630014_software_Engineering/assets/149077212/3813a269-096b-4090-b952-09acbb10712a">



In recent years, CNN has been adapted as a better alternative to the traditional classification techniques. This is because they have been improved models with new network structures that use multi-convolutional layers for fashion classification with good accuracy (Nocentini et al. 2022).  Due to its advantages, two neural network models will be trained for a fashion classification task using the same datasets. A comparison of their performance will be discussed below. 

## 2. Literature Review

Fashion image classification can be challenging due to the unique properties and attributes of an individual product. However, the emergence of CNN a deep learning technique, has demonstrated notable success in addressing the challenges of fashion image classification tasks. Three different CNN architectures for image classification using the fashion-mnist dataset were proposed by Bhatnagar et al. (2017). Results show an accuracy of 92.54% using a two-layered CNN with batch normalization and skip connections. A CNN-SVM model and CNN-softmax model were used on a fashion-mnist dataset by Agarap (2019). An accuracy of 90.72% was achieved using the CNN-SVM model while the CNN-softmax model had a test accuracy of 91.86%. Different forms of CNN were applied for image classification using the fashion-mnist dataset by Xhaferra et al. (2022).To resolve the issue of model overfitting, CNN-C1 and CNN-C2 were compared to determine the model with the best result. Results show that CNN-C2 is best with an accuracy of 93.11% compared to CNN-C1 with an accuracy of 88.95%. However, there are limitations to using CNN on the Fashion MNIST dataset because of its susceptibility to overfitting, challenges in capturing intricate details, and the computational intensity of training.

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

The training loop adopts a batch size of 16, with a 70-30 split between training and validation data. For 10 epochs, it leverages PyTorch's multi-layer perceptron, employing the `train_mlp` and `val_mlp` functions for training and validation. The code logs loss and accuracy metrics and incorporates early stopping based on validation accuracy. Checkpoints are saved at a defined frequency, and input reshaping is conducted for each batch. Tensorboard's 'Summary Writer' is utilized to record loss and accuracy metrics.

Results show that during training, the validation accuracy reached a maximum of 57% after 10 epochs, and the corresponding validation loss was 1.89%. However, the accuracy did not improve further so the best model using this network was not saved.

![NEURAL-NETWORK_GRAPH](https://github.com/Iampegassi/s5630014_software_Engineering/assets/149077212/7049e8c7-369f-4ac4-a9ea-f1033ef09559)


![CNN (3)](https://github.com/Iampegassi/s5630014_software_Engineering/assets/149077212/fc18ecea-0fbf-4ca5-af87-835c3f428858)

### 2. Convolutional neural network model with PyTorch

This PyTorch code defines a Convolutional Neural Network (CNN) using PyTorch's neural network module (`nn.Module`). Let's break down the key components:

#### Initialization (`__init__` method):
+  `self.conv1`, `self.conv2`, and `self.conv3` define three convolutional layers with increasing output channels, applying ReLU activation after each convolution, and followed by max-pooling.
+  `self.fc1` and `self.fc2` define two fully connected layers, applying ReLU activation after the first fully connected layer.
+ `self.softmax` is the softmax activation used for the final output layer.

#### Max-Pooling Layers (nn.MaxPool2d):
+ Max-pooling is a downsampling operation that reduces the spatial dimensions of the input.
+ It operates on 2D input tensors (feature maps) and extracts the maximum value from each local region.
+ The kernel_size parameter specifies the size of the pooling window, and it is commonly set to 2 in this code.
+ Max-pooling helps in reducing the spatial resolution, making the network more computationally efficient and promoting translational invariance.

#### Forward Pass (`forward` method):
+ The forward method specifies how input data should pass through the network.
+ Convolutional layers (`self.conv1`, `self.conv2`, `self.conv3`) are followed by ReLU activation and max-pooling.
+ The output is then flattened and passed through fully connected layers (`self.fc1`, `self.fc2`) with ReLU activation.
+ The final output is obtained using the softmax activation.

The training loop iterates for 10 epochs, incorporating periodic checkpoint saving and logging of loss and accuracy metrics for both training and validation. The process halts if early stopping criteria are satisfied. Tensorboard is employed for visualization, and the model's state is saved either after training or if early stopping is triggered.

The model evolves over 10 epochs, starting with a training loss of 1.83 (63.68% accuracy) and a validation loss of 1.77 (69.31% accuracy). As training progresses, losses decrease, and accuracies rise. Ultimately, the model attains a training loss of 1.71 (74.65% accuracy) and a validation loss of 1.72 (73.49% accuracy). The best model was saved.

![CNN_MODEL_GRAPH](https://github.com/Iampegassi/s5630014_software_Engineering/assets/149077212/b23ff801-21d2-407e-a087-b68a203124d5)

## 5. Results 
The testing loop output shows the assessment of the pre-trained Convolutional Neural Network (CNN) model using the `test_model` function. This function computes and displays the test accuracy and loss on fashion-mnist test datasets. After loading a saved checkpoint and extracting the model's state dictionary, a new instance of the CNN model is initialized. The model is then moved to the (DEVICE). The testing function is executed, producing the following metrics: Test Accuracy of 73.82% and Test Loss of 1.7219%.

A unit test class, `TestFashionClassification`, was created to evaluate the pre-trained CNN model and the key points include:

#### Setup (`setUp` method):
+ Loads the pre-trained CNN model from a checkpoint.
+ Initializes the model, device, test dataset, and data loader.
+ Sets the criterion for loss calculation.

#### Model Prediction Test (`test_model_prediction` method):
+ Verifies that model predictions fall within the expected range (0 to 9).

#### Model Accuracy Test (`test_model_accuracy` method):
+ Validates the accuracy calculation for model predictions, ensuring it lies between 0 and 1.

#### Data Loader Test (`test_data_loader` method):
+ Checks the shape of inputs and labels in the data loader.

#### Model Training Test (`test_model_training` method):
+ Tests the model training process for a few epochs using a simple optimizer.

#### Model Save and Load Test (`test_model_save_load` method):
+ Saves and loads the model state, ensuring the loaded model is an instance of the CNN class.

#### Model Forward Pass Test (`test_model_forward_pass` method):
+ Validates the shape of the output from a forward pass.

#### Early Stopping Test (`test_early_stopping` method):
+ Sets up an early stopper and simulates a training loop, verifying the correct epoch counter.

Finally, the code runs seven tests with no errors or failures, indicating successful execution.


## Conclusion
In conclusion, results from comparing the performance of the simple neural network and the CNN model on the fashion-mnist dataset show that the CNN model accuracy in the training and testing loop outperforms the simple neural network. However,  experimenting with deeper architectures, adjusting hyperparameters and exploring regularization techniques can further enhance the  CNN model. Implementing transfer learning using pre-trained models for relevant tasks could also be considered.  

## References
[1] Agarap, A. F., 2019. An Architecture Combining Convolutional Neural Network (CNN) and Support Vector Machine (SVM) for Image Classification [online]. arXiv.org. Available from: https://arxiv.org/abs/1712.03541v2 [Accessed 25 Jan 2024].

[2] Bhatnagar, S., Ghosal, D. and Kolekar, M. H., 2017. Classification of fashion article images using convolutional neural networks. 2017 Fourth International Conference on Image Information Processing (ICIIP).

[3] kaggle, n.d. Fashion MNIST [online]. www.kaggle.com. Available from: https://www.kaggle.com/datasets/zalando-research/fashionmnist.

[4] McDonnell, M. D. and Vladusich, T., 2015. Enhanced image classification with a fast-learning shallow convolutional neural network. 2015 International Joint Conference on Neural Networks (IJCNN).

[5] Nocentini, O., Kim, J., Bashir, M. Z. and Cavallo, F., 2022. Image Classification Using Multiple Convolutional Neural Networks on the Fashion-MNIST Dataset. Sensors, 22 (23), 9544.

[6] Xhaferra, E., Cina, E. and Toti, L., 2022. Classification of Standard FASHION MNIST Dataset Using Deep Learning Based CNN Algorithms [online]. IEEE Xplore. Available from: https://ieeexplore.ieee.org/document/9932737 [Accessed 25 Jan 2024].

[7] Xu, J., Wei, Y., Wang, A., Zhao, H. and Lefloch, D., 2022. Analysis of Clothing Image Classification Models: A Comparison Study between Traditional Machine Learning and Deep Learning Models. Fibres & Textiles in Eastern Europe, 30 (5), 66–78.
