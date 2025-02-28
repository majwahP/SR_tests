import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloaders  # Import dataset loader
import argparse  # For command-line arguments
import matplotlib.pyplot as plt
import numpy as np

"""
Implementation of Super Resolution using CNN, SRCNN according to C,dong et.al (https://ieeexplore.ieee.org/document/7115171)
Training done with downsapling of CIFAR10 dataset
run by;
>>'cd CNN_SR'
>>'python ml_model_SR_CNN.py --train'
>>'python ml_model_SR_CNN.py --eval'
"""

# SRCNN Model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)  # f1 = 9
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)  # f2 = 1
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)  # f3 = 5
        self.relu = nn.ReLU()
        #self._initialize_weights() #init weights from gaussian distribution and weghts as zero

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)  # No activation, regression task
        return x
    
    #def _initialize_weights(self):
    #    """Initialize weights as per SRCNN paper (Gaussian mean=0, std=0.001) and biases as 0."""
    #    for m in self.modules():
    #        if isinstance(m, nn.Conv2d):
    #            nn.init.normal_(m.weight, mean=0, std=0.001)  # Gaussian Distribution
    #            if m.bias is not None:
    #                nn.init.constant_(m.bias, 0)  # Bias = 0

# Define PSNR Calculation for evaluation
def psnr(sr_img, hr_img):
    """Compute PSNR (Peak Signal-to-Noise Ratio) between SR and HR images."""
    mse = torch.mean((sr_img - hr_img) ** 2)
    if mse == 0:
        return float('inf')  # Perfect reconstruction
    max_pixel = 1.0  # Since images are normalized between [0,1]
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

#function to plot result
def tensor_to_image(tensor):
    """Convert a PyTorch tensor (C, H, W) to a NumPy image (H, W, C)."""
    image = tensor.permute(1, 2, 0).cpu().numpy()  # Convert (C, H, W) → (H, W, C)
    image = np.clip(image, 0, 1)  # Ensure values are in range [0,1]
    return image

def plot_sample_images(model, test_loader, device, save_path="evaluation_results.png"):
    """Saves a few Super-Resolved (SR) images vs. High-Resolution (HR) images instead of displaying them."""
    model.eval()
    with torch.no_grad():
        lr_images, hr_images = next(iter(test_loader))  # Get one batch
        lr_images, hr_images = lr_images.to(device), hr_images.to(device)

        sr_images = model(lr_images)  # Generate Super-Resolved images

        fig, axes = plt.subplots(3, len(lr_images[:5]), figsize=(15, 6))  # Display 5 images

        for i in range(len(lr_images[:5])):  # Only show the first 5 images
            axes[0, i].imshow(tensor_to_image(lr_images[i]))  # Low-Resolution
            axes[0, i].axis("off")
            axes[0, i].set_title("Low-Res")

            axes[1, i].imshow(tensor_to_image(sr_images[i]))  # Super-Resolved
            axes[1, i].axis("off")
            axes[1, i].set_title("Super-Res")

            axes[2, i].imshow(tensor_to_image(hr_images[i]))  # High-Resolution (Ground Truth)
            axes[2, i].axis("off")
            axes[2, i].set_title("High-Res")

        plt.savefig(save_path)  # Saves the plot as an image file
        plt.close()  # Prevents display issues

    print(f"Evaluation images saved to {save_path}")




# Train the SRCNN Model
def train_model(model, train_loader, val_loader, device, num_epochs=5):
    """Trains the SRCNN model."""
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.Adam([
    #{'params': model.conv1.parameters(), 'lr': 1e-4},  # First layer
    #{'params': model.conv2.parameters(), 'lr': 1e-4},  # Second layer
    #{'params': model.conv3.parameters(), 'lr': 1e-5}   # Last layer
#])

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for lr_images, hr_images in train_loader:
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            
            optimizer.zero_grad()
            outputs = model(lr_images)
            loss = criterion(outputs, hr_images)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()


        # Compute validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr_images, hr_images in val_loader:
                lr_images, hr_images = lr_images.to(device), hr_images.to(device)
                outputs = model(lr_images)
                val_loss += criterion(outputs, hr_images).item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

    print("Training complete!")
    torch.save(model.state_dict(), "srcnn_model.pth")  # Save trained model
    print("Model saved as srcnn_model.pth")

# Evaluate the SRCNN Model
def evaluate_model(model, test_loader, device):
    """Evaluates the trained SRCNN model using PSNR."""
    model.to(device)
    model.load_state_dict(torch.load("srcnn_model.pth", map_location=device))  # Load trained model
    model.eval()
    
    total_psnr = 0
    num_images = 0

    with torch.no_grad():
        for lr_images, hr_images in test_loader:
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            
            sr_images = model(lr_images)  # Generate Super-Resolved image
            image_psnr = psnr(sr_images, hr_images)
            total_psnr += image_psnr.item()
            num_images += 1

    avg_psnr = total_psnr / num_images
    print(f"Average PSNR on Test Set: {avg_psnr:.2f} dB")

    plot_sample_images(model, test_loader, device)

# Main function to allow both training and evaluation in one file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Evaluate SRCNN Model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    args = parser.parse_args()

    # Load dataset
    train_loader,val_loader, test_loader = get_dataloaders(batch_size=32)

    # Initialize model
    device = torch.device("cpu")  # Use CPU
    model = SRCNN()

    # Train and/or evaluate
    if args.train:
        train_model(model, train_loader, val_loader, device, num_epochs=7)

    if args.eval:
        evaluate_model(model, test_loader, device)

    if not args.train and not args.eval:
        print("Please specify --train, --eval, or both.")
