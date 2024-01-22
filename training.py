import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
import warnings
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Suppress  messages
warnings.simplefilter(action='ignore', category=FutureWarning)



train_csv_path=f"fashion-mnist_train.csv"
class FashionMNISTDataset(Dataset):
    def __init__(self, filepath: str):
        super().__init__()
        # Load data from CSV filepath defined earlier into a Pandas dataframe
        self.dataset = pd.read_csv(filepath)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.iloc[index]

def process_image (image):
    img = Image.open(f"{DATA_PREFIX}/" + image)
    return transform(img)


# item is referred to as pd_series because of the get_image_tensor function
def get_image_tensor(pd_series, threshold):
    img_tensor = torch.FloatTensor(pd_series.iloc[1:])
    img_tensor = img_tensor.reshape(1, 28, 28)
    #img_tensor = torch.where(img_tensor < threshold, 0, img_tensor)
    return img_tensor


def custom_collate_fn(batch):
    # Define a tensor of the same size as our image batch to store loaded images into
    image_batch_tensor = torch.FloatTensor(len(batch), 1, 28, 28)
    # Define empty lists to hold items we encounter
    image_tensors = []
    labels = []
    for item in batch:
        # load a single image
        image_tensor = get_image_tensor(item, threshold=80)
        # put image into a list
        image_tensors.append(image_tensor.unsqueeze(0))
        # put the same image's label into another list
        labels.append(item.iloc[0])


    # Concatenate the list of individual tensors (image_tensors) into a single Pytorch tensor (image_batch_tensor)
    torch.cat(image_tensors, out=image_batch_tensor)
    label_batch_tensor = torch.LongTensor(labels)
    # Use the label list to create a torch tensor of ints
    return image_batch_tensor, label_batch_tensor



def load_data(data_path, batch_sz=100, train_val_split=[0.7, 0.3]):
    assert sum(train_val_split) == 1, "Train and val fractions should sum to 1!"  # Always a good idea to use static asserts when processing arguments that are passed in by a user!
    # Instantiate our previously defined dataset
    dataset = FashionMNISTDataset(data_path)
    # split dataset into train and val
    tr_va = []
    for frac in train_val_split:
        actual_count = frac * len(dataset)
        actual_count = round(actual_count)
        tr_va.append(actual_count)

    train_split, val_split = random_split(dataset, tr_va)

    # Use Pytorch DataLoader to load each split into memory. It's important to pass in our custom collate function, so it knows how to interpret the
    # data and load it. num_workers tells the DataLoader how many CPU threads to use so that data can be loaded in parallel, which is faster

    # Get CPU count
    n_cpus = mp.cpu_count() # returns number of CPU cores on this machine
    train_dl = DataLoader(
        train_split,
        batch_size=batch_sz,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=n_cpus
    )

    val_dl = DataLoader(
        val_split,
        batch_size=batch_sz,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=n_cpus
    )

    return train_dl, val_dl



class EarlyStopper:
    def __init__(self, patience=3, tolerance=0):
       self.patience = patience # How many epochs in a row the model is allowed to underperform
       self.tolerance = tolerance # How much leeway the model has (i.e. how close it can get to underperforming before it is counted as such)
       self.epoch_counter = 0 # Keeping track of how many epochs in a row were failed
       self.max_val_acc = np.NINF # Keeping track of best metric so far

    def should_stop(self, val_acc):
        # print(f"current val max : {self.max_val_acc} , val acc : {val_acc}" )
        if val_acc > self.max_val_acc:
            self.max_val_acc = val_acc
            self.epoch_counter = 0
        elif val_acc < (self.max_val_acc - self.tolerance):
            self.epoch_counter += 1
            if self.epoch_counter >= self.patience:
                return True
        return False


class NeuralNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNetwork, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, num_classes)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    out = self.softmax(out)
    return out



from torch.optim.lr_scheduler import ExponentialLR
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dl, val_dl = load_data(f"fashion-mnist_train.csv", batch_sz=16)
n_input = 28*28
n_hidden = 128
n_classes = 10
epoch = 10


model = NeuralNetwork(n_input, n_hidden, n_classes)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=0.001, momentum =0.9)

gamma = 0.8 #gama is multiplied by the lr every epoch, decreasing the learning rate every time
schd = ExponentialLR(optimiser,gamma)

print(f"usig device: {DEVICE}")

from torch.utils.tensorboard import SummaryWriter

import os


# Saves a model to file, and names it after the current epoch
def save_checkpoint(model, epoch, save_dir):
    filename = f"checkpoint_{epoch}.pth"
    save_path = f"{save_dir}/{filename}"
    torch.save({'state_dict': model.state_dict()}, save_path)

# Assuming EarlyStopper and save_checkpoint are defined elsewhere
writer = SummaryWriter()
early_stopper = EarlyStopper()


# Assuming net, criterion, optimiser, train_dl, val_dl, writer, EarlyStopper, and save_checkpoint are defined elsewhere
# ...


# Specify the training CSV path
train_csv_path = "fashion-mnist_train.csv"

#training loop
def train_mlp(net, criterion, optimiser, train_dl, writer, epoch, stopper, checkpoint_frequency):
    running_loss = 0.0
    running_acc = 0.0
    net.train()

    for i, data in enumerate(train_dl, 0):
        inputs, labels = data
        batch_size = labels.shape[0]

        optimiser.zero_grad()
        inputs = inputs.reshape(batch_size, 28*28).to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        preds = torch.argmax(outputs, dim=1)
        loss = criterion(outputs, labels)
        correct = int(torch.eq(preds, labels).sum())

        running_acc += correct / batch_size
        running_loss += loss.item()
        

    epoch_loss = running_loss / len(train_dl)
    epoch_acc = running_acc / len(train_dl)
    return epoch_loss, epoch_acc

def val_mlp(net, criterion, val_dl):
    running_loss = 0.0
    running_acc = 0.0
    net.eval()

    for i, data in enumerate(val_dl, 0):
        with torch.no_grad():
            inputs, labels = data
            batch_size = labels.shape[0]
            labels = labels.to(DEVICE)

            inputs = inputs.reshape(batch_size, 28*28).to(DEVICE)
            outputs = net(inputs)
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)

            correct = int(torch.eq(preds, labels).sum())

            running_acc += correct / batch_size
            running_loss += loss.item()

    epoch_loss = running_loss / len(val_dl)
    epoch_acc = running_acc / len(val_dl)

    return epoch_loss, epoch_acc

# Set your desired checkpoint frequency
checkpoint_frequency = 1

# for epoch in range(10):
#     train_loss, train_acc = train_mlp(model, criterion, optimiser, train_dl, writer, epoch, early_stopper, checkpoint_frequency)
#     val_loss, val_acc = val_mlp(model, criterion, val_dl)

#     writer.add_scalar('Loss/train', train_loss, epoch)
#     writer.add_scalar('Loss/val', val_loss, epoch)

#     writer.add_scalar('Accuracy/train', train_acc, epoch)
#     writer.add_scalar('Accuracy/val', val_acc, epoch)

#     # Log loss and accuracy metrics using the writer so we can see them in Tensorboard
#     # Check whether we need to save the model to a checkpoint file
#     if epoch % checkpoint_frequency == 0:
#         save_checkpoint(model, epoch, f"saved_models")  # Adjust the path

#     # Check whether we should stop the training based on the maximum epochs
#     if early_stopper.should_stop(val_acc):
#         print(f"\nStopping after {early_stopper.epoch_counter} epochs.")
#         # If stopping, save the model's state
#         save_checkpoint(model, epoch, f"saved_models")  # Adjust the path
#         break

#     print(f"Epoch {epoch}, train_loss: {train_loss}, train_acc: {train_acc}, val_loss: {val_loss}, val_acc: {val_acc}")

# print("Finished Training")


# exit()
# Define the CNN Module
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Convolutional layers
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)

        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_input = 28*28
n_hidden = 128
n_classes = 10
cnn_model = CNN()
cnn_model = cnn_model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(cnn_model.parameters(), lr=0.0003, momentum =0.9)

gamma = 0.9 #gama is multiplied by the lr every epoch, decreasing the learning rate every time
schd = ExponentialLR(optimiser,gamma)
epoch = 10


print(f"usig device: {DEVICE}")


# Initialize EarlyStopper with a maximum of 3 epochs
early_stopper = EarlyStopper()

# Set your desired checkpoint frequency
checkpoint_frequency = 1

# Initialize Tensorboard writer
writer = SummaryWriter()

# Training loop
def train_cnn(net, criterion, optimiser, train_dl, writer, epoch, early_stopper, checkpoint_frequency):
    running_loss = 0.0
    running_acc = 0.0
    net.train()

    for i, data in enumerate(train_dl, 0):
        inputs, labels = data
        labels = labels.to(DEVICE)
        batch_sz = inputs.shape[0]

        optimiser.zero_grad()
        # Assuming you want to reshape inputs if needed
        inputs = inputs.reshape(batch_sz, 1, 28, 28).to(DEVICE)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        preds = torch.argmax(outputs, dim=1)
        correct = int(torch.eq(preds, labels).sum())

        running_acc += correct / len(labels)
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dl)
    epoch_acc = running_acc / len(train_dl)

    return epoch_loss, epoch_acc

# Validation loop
def val_cnn(net, criterion, val_dl):
    running_loss = 0.0
    running_acc = 0.0
    net.eval()

    with torch.no_grad():
        for i, data in enumerate(val_dl, 0):
            inputs, labels = data
            labels =labels.to(DEVICE)
            batch_sz = inputs.shape[0]

            # Assuming you want to reshape inputs if needed
            inputs = inputs.reshape(batch_sz, 1, 28, 28).to(DEVICE)

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            correct = int(torch.eq(preds, labels).sum())

            running_acc += correct / len(labels)
            running_loss += loss.item()

    epoch_loss = running_loss / len(val_dl)
    epoch_acc = running_acc / len(val_dl)

    return epoch_loss, epoch_acc

 # Set your desired checkpoint frequency
checkpoint_frequency = 1

for epoch in range(10):
    
    train_loss, train_acc = train_cnn(cnn_model, criterion, optimiser, train_dl, writer, epoch, early_stopper, checkpoint_frequency)
    val_loss, val_acc = val_cnn(cnn_model, criterion, val_dl)
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)

    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    # Log loss and accuracy metrics using the writer so we can see them in Tensorboard
    # Check whether we need to save the model to a checkpoint file
    if epoch % checkpoint_frequency == 0:
        save_checkpoint(cnn_model, epoch, f"saved_models")

    # Check whether we should stop the training based on the maximum epochs
    if early_stopper.should_stop(val_acc):
        print(f"\nStopping after {early_stopper.epoch_counter} epochs.")
        # If stopping, save the model's state
        save_checkpoint(cnn_model, epoch, f"saved_models")
        break
    print(f"Epoch {epoch}, train_loss: {train_loss}, train_acc: {train_acc}, val_loss: {val_loss}, val_acc: {val_acc}")

print("Finished Training")

# Close Tensorboard writer
writer.close()
