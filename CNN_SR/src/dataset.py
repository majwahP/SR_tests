import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from PIL import Image

"""
Class to create a dataset from CIFAR10 for training of a SR model. The class loads the dataset, and creates a LR- 
dataset by downsapling followed by upsampling. 

"""

class CIFAR10_SR(Dataset):
    def __init__(self, root="./data", train=True, scale_factor=4):
        super().__init__()
        
        # Define HR (high-resolution) transformations
        self.hr_transform = transforms.Compose([
            transforms.ToTensor()  # Convert image to tensor (C, H, W)
        ])

        # Define LR (low-resolution) transformations (downsampling)
        self.lr_transform = transforms.Compose([
            transforms.Resize((16, 16), interpolation=Image.BICUBIC),  # Downscale
            transforms.Resize((32, 32), interpolation=Image.BICUBIC), # Upscale back
            transforms.ToTensor()  # Convert to tensor
        ])
        
        # Load CIFAR-10 dataset
        self.data = datasets.CIFAR10(root=root, train=train, download=True)
        self.images = self.data.data  # Numpy array (50000, 32, 32, 3)
        self.labels = np.array(self.data.targets)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = Image.fromarray(img)  # Convert numpy to PIL Image

        hr_img = self.hr_transform(img)  # Apply HR transform
        lr_img = self.lr_transform(img)  # Apply LR transform

        return lr_img, hr_img  # Return both LR and HR versions

"""
return touple of training and test dataset of HR and LR images, use a sudset of total, 50 000 train and 10 000 test 
resuce the training time
"""
def get_dataloaders(batch_size=32, num_workers=2, train_size=50000, test_size=10000):
    # Load the full dataset
    train_set = CIFAR10_SR(train=True)
    test_set = CIFAR10_SR(train=False)

    # Select a random subset of the dataset
    train_indices = np.random.choice(len(train_set), train_size, replace=False)
    test_indices = np.random.choice(len(test_set), test_size, replace=False)

    train_set = Subset(train_set, train_indices)
    test_set = Subset(test_set, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

