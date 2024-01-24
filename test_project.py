import unittest
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from your_utility_functions import calculate_accuracy, EarlyStopper
import multiprocessing as mp
import os


class TestFashionClassification(unittest.TestCase):
    def setUp(self):
        # Load the trained model from the checkpoint
        checkpoint = torch.load("saved_models/cnn_checkpoint_9.pth")
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        self.model = CNN(num_classes=10)  # Assuming 10 classes for Fashion MNIST
        self.model.load_state_dict(state_dict)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Load the test dataset using the provided CSV path
        test_csv_path = "fashion-mnist_test.csv"  # Replace with the actual path
        test_dataset = FashionMNISTDataset(test_csv_path)
        n_cpus = mp.cpu_count()
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=n_cpus
        )

        self.criterion = torch.nn.CrossEntropyLoss()

    def test_model_prediction(self):
        for inputs, labels in self.test_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            predicted_class = self.model(inputs).argmax(dim=1)
            self.assertTrue(torch.all(predicted_class >= 0) and torch.all(predicted_class < 10))


        # Define a simple version of calculate_accuracy for testing purposes
    def calculate_accuracy(predictions, labels):
        correct = (predictions == labels).sum().item()
        total = len(labels)
        accuracy = correct / total
        return accuracy

    def test_model_accuracy(self):
        for inputs, labels in self.test_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            predictions = self.model(inputs).argmax(dim=1)
            accuracy = calculate_accuracy(predictions, labels)
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)

    def test_data_loader(self):
        for inputs, labels in self.test_dataloader:
            self.assertEqual(inputs.shape, torch.Size([16, 1, 28, 28]))
            self.assertEqual(labels.shape, torch.Size([16]))

    
    def test_model_training(self):
        optimizer = SGD(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(3):
            for inputs, labels in self.test_dataloader:
                optimizer.zero_grad()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def test_model_save_load(self):
        torch.save(self.model.state_dict(), 'fashionmnist_test_model.pth')
        loaded_model = CNN(num_classes=10)
        loaded_model.load_state_dict(torch.load('fashionmnist_test_model.pth'))
        self.assertTrue(isinstance(loaded_model, CNN))

    def test_model_forward_pass(self):
        inputs = torch.randn(16, 1, 28, 28).to(self.device)
        outputs = self.model(inputs)
        self.assertEqual(outputs.shape, torch.Size([16, 10]))

    def test_early_stopping(self):
        # Set up early stopper
        early_stopper = EarlyStopper(patience=3)

        # Training loop with early stopping
        for epoch in range(10):
            # Simulating training process
            val_acc = 0.75 - epoch * 0.1  # Decreasing validation accuracy
            if early_stopper.should_stop(val_acc):
                break

        self.assertEqual(early_stopper.epoch_counter, 3)  # Patience is 3

# Run the tests
unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestFashionClassification))
