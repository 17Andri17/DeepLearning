import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tqdm import tqdm



def load_data(batch_size=128):
    cinic_directory = 'C:/Users/zp13279/Desktop/Repos_other/DeepLearning/kod/data'
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    torch.manual_seed(42)

    cinic_train = DataLoader(
        torchvision.datasets.ImageFolder(cinic_directory + '/train',
        transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)])),
        batch_size=batch_size, shuffle=True
    )

    cinic_test = DataLoader(
        torchvision.datasets.ImageFolder(cinic_directory + '/valid',
        transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)])),
        batch_size=batch_size, shuffle=False
    )
    return {
        'train': cinic_train,
        'test': cinic_test
    }

def load_data_few_shot(batch_size=128, few_shot_samples_per_class=None):
    cinic_directory = 'C:/Users/zp13279/Desktop/Repos_other/DeepLearning/kod/data'
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    torch.manual_seed(42)

    train_dataset = torchvision.datasets.ImageFolder(
        cinic_directory + '/train',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std)
        ])
    )
    
    if few_shot_samples_per_class is not None:
        # Create a few-shot dataset
        few_shot_data = []
        class_count = defaultdict(int)
        
        for img, label in train_dataset:
            if class_count[label] < few_shot_samples_per_class:
                few_shot_data.append((img, label))
                class_count[label] += 1
                
            # Stop if we have enough samples for all classes
            if all(count >= few_shot_samples_per_class for count in class_count.values()):
                break

        train_dataset = few_shot_data

    cinic_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    cinic_test = DataLoader(
        torchvision.datasets.ImageFolder(cinic_directory + '/valid',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std)
        ])),
        batch_size=batch_size, shuffle=False
    )
    
    return {
        'train': cinic_train,
        'test': cinic_test
    }





class VGGStyleCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(VGGStyleCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Redukcja rozmiaru 32x32 -> 16x16

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Redukcja rozmiaru 16x16 -> 8x8

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Redukcja rozmiaru 8x8 -> 4x4

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Redukcja rozmiaru 4x4 -> 2x2
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 10)  # CINIC-10 ma 10 klas
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    