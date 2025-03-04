import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
from PIL import Image

"""
Class to create a dataset from CIFAR10 for training of a SR model. The class loads the dataset, and creates a LR- 
dataset by downsapling followed by upsampling. 

"""

#directorys for using CIFAR10 64x64
import os
data_dir = "/scratch/mwahlin/datasets/cifar10-64/train"
test_dir = "/scratch/mwahlin/datasets/cifar10-64/test"


class CIFAR10_SR(Dataset):
    def __init__(self, root="./data", train=True):
        super().__init__()
        
        # Select correct dataset path when using CIFAR10 64x64
        self.dataset_folder = data_dir if train else test_dir

        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Dataset folder {self.dataset_folder} does not exist! Check the path.")

        # Define HR (high-resolution) transformations
        self.hr_transform = transforms.Compose([
            transforms.ToTensor()  # Convert image to tensor (C, H, W)
        ])

        # Define LR (low-resolution) transformations (downsampling)
        self.lr_transform = transforms.Compose([
            #CIFAR10 32x32
            #transforms.Resize((16, 16), interpolation=Image.BICUBIC),  # Downscale
            #transforms.Resize((32, 32), interpolation=Image.BICUBIC), # Upscale back
            #CIFAR10 64x64
            transforms.Resize((32, 32), interpolation=Image.BICUBIC),  # Downscale
            transforms.Resize((64, 64), interpolation=Image.BICUBIC),  # Upscale back to 64x64
            transforms.ToTensor()  # Convert to tensor
        ])
        
        # Load CIFAR-10 32x32 dataset
        #self.data = datasets.CIFAR10(root=root, train=train, download=True)
        #self.images = self.data.data  # Numpy array (50000, 32, 32, 3)
        #self.labels = np.array(self.data.targets)

        # Load CIFAR10 64x64 dataset
        self.data = datasets.ImageFolder(root=self.dataset_folder, transform=None)

        self.images = [img_path for img_path, _ in self.data.samples]
        self.labels = [label for _, label in self.data.samples]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        #CIFAR 32x32
        #img = Image.fromarray(img_path)  # Convert numpy to PIL Image

        #CIFAR10 64x64
        img = Image.open(img_path).convert("RGB")

        hr_img = self.hr_transform(img)  # Apply HR transform
        lr_img = self.lr_transform(img)  # Apply LR transform

        return lr_img, hr_img  # Return both LR and HR versions

"""
return touple of training and test dataset of HR and LR images, use a sudset of total, 50 000 train and 10 000 test 
resuce the training time
"""
def get_dataloaders(batch_size=32, num_workers=2, train_size=40000, val_size=10000, test_size=10000):
    # Load the full dataset CIFAR10 32x32
    #full_train_set = CIFAR10_SR(train=True)
    #test_set = CIFAR10_SR(train=False)

    full_train_set = CIFAR10_SR(root=data_dir, train=True)
    test_set = CIFAR10_SR(root=test_dir, train=False)

    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)  # No shuffle for validation
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

