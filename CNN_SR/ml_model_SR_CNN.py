import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from src.dataset import get_dataloaders
import matplotlib.pyplot as plt
import numpy as np

# Parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEPTH = 10
WIDTH_FACTOR = 2

# Initialize WandB
wandb.init(
    project="Super-Resolution",
    name=f"SRCNN-epochs:{NUM_EPOCHS}-Depth:{DEPTH}-Width_fac:{WIDTH_FACTOR}-resudual",
    mode="online", #to not save each run locally
    config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "optimizer": "Adam",
        "architecture": "SRCNN",
        "dataset" : "CIFAR10-32",
        "depth": DEPTH,
        "width factor": WIDTH_FACTOR,
        "conv1_filters": 64,
        "conv2_filters": 32,
    }
)


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        base_filters1 = 64 * WIDTH_FACTOR  # Increase width (default: 64 → 128 if width_factor=2)
        base_filters2 = 32 * WIDTH_FACTOR

        self.conv1 = nn.Conv2d(3, base_filters1, kernel_size=9, padding=4)  

        #additional layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(DEPTH - 3):  # Subtract 3 to keep original structure
            self.hidden_layers.append(nn.Conv2d(base_filters1, base_filters1, kernel_size=3, padding=1))
        self.conv2 = nn.Conv2d(base_filters1, base_filters2, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(base_filters2, 3, kernel_size=5, padding=2)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)  
        return x


class SRCNN2(nn.Module):
    def __init__(self, width_start=128, width_end=16):  
        """
        width_start: Number of filters in the first layer (default 128)
        width_end: Number of filters in the second last layer before output (default 16)
        """
        super(SRCNN2, self).__init__()

        # Define filter sizes decreasing continuously over 5 layers
        base_filters = torch.linspace(width_start, width_end, steps=5).int().tolist()
        
        self.relu = nn.ReLU()

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, base_filters[0], kernel_size=9, padding=4)

        # Intermediate layers with decreasing filters
        self.conv2 = nn.Conv2d(base_filters[0], base_filters[1], kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(base_filters[1], base_filters[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(base_filters[2], base_filters[3], kernel_size=3, padding=1)

        # Output layer (fixed at 3 channels for RGB)
        self.conv_final = nn.Conv2d(base_filters[3], 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv_final(x)  # No activation in the final layer
        return x

class ResidualSRCNN(nn.Module):
    def __init__(self):
        super(ResidualSRCNN, self).__init__()

        base_filters1 = 64 * WIDTH_FACTOR  # Wider first layer
        base_filters2 = 32 * WIDTH_FACTOR  # Wider second layer

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, base_filters1, kernel_size=9, padding=4)

        # Residual layers (Extra convolutional layers)
        self.hidden_layers = nn.ModuleList()
        for _ in range(DEPTH - 3): 
            self.hidden_layers.append(nn.Conv2d(base_filters1, base_filters1, kernel_size=3, padding=1))

        self.conv2 = nn.Conv2d(base_filters1, base_filters2, kernel_size=1, padding=0)

        # Output layer (fixed at 3 channels for RGB)
        self.conv3 = nn.Conv2d(base_filters2, 3, kernel_size=5, padding=2)

    def forward(self, x):
        identity = x  # Save the original input

        x = self.relu(self.conv1(x))

        # Apply residual layers
        for layer in self.hidden_layers:
            residual = x  # Save the input before transformation
            x = self.relu(layer(x))
            x = x + residual  # Add residual connection

        x = self.relu(self.conv2(x))
        x = self.conv3(x)  # No activation in the final layer

        x += identity  #  Add residual connection from the input to the final output

        return x


# PSNR Calculation
def psnr(sr_img, hr_img):
    mse = torch.mean((sr_img - hr_img) ** 2)
    if mse == 0:
        return float('inf')  
    max_pixel = 1.0  
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))


def tensor_to_image(tensor):
    image = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(image, 0, 1)



def plot_sample_images(model, test_loader, device, save_path="evaluation_results.png"):
    model.eval()
    with torch.no_grad():
        lr_images, hr_images = next(iter(test_loader))  # Get one batch
        lr_images, hr_images = lr_images.to(device), hr_images.to(device)
        sr_images = model(lr_images)  # Generate Super-Resolved images

        fig, axes = plt.subplots(3, len(lr_images[:5]), figsize=(15, 6))  # Display 5 images

        for i in range(len(lr_images[:5])):  
            axes[0, i].imshow(tensor_to_image(lr_images[i]))  # Low-Resolution
            axes[0, i].axis("off")
            axes[0, i].set_title("Low-Res")

            axes[1, i].imshow(tensor_to_image(sr_images[i]))  # Super-Resolved
            axes[1, i].axis("off")
            axes[1, i].set_title("Super-Res")

            axes[2, i].imshow(tensor_to_image(hr_images[i]))  # High-Resolution (Ground Truth)
            axes[2, i].axis("off")
            axes[2, i].set_title("High-Res")

        plt.savefig(save_path)  
        plt.close()  

        # Log images to WandB
        wandb_images = []
        for i in range(min(5, len(lr_images))):  
            wandb_images.append(wandb.Image(tensor_to_image(lr_images[i]), caption="Low-Res"))
            wandb_images.append(wandb.Image(tensor_to_image(sr_images[i]), caption="Super-Res"))
            wandb_images.append(wandb.Image(tensor_to_image(hr_images[i]), caption="High-Res"))

        wandb.log({"Super-Resolution Samples": wandb_images})

    print(f"Evaluation images saved to {save_path}")


def train_model(model, train_loader, val_loader, device, num_epochs):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

        # Compute Validation Loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr_images, hr_images in val_loader:
                lr_images, hr_images = lr_images.to(device), hr_images.to(device)
                outputs = model(lr_images)
                val_loss += criterion(outputs, hr_images).item()

        #log
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
        })

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), "srcnn_model.pth")
    print("Model saved as srcnn_model.pth")


def evaluate_model(model, test_loader, device):
    model.to(device)
    model.load_state_dict(torch.load("srcnn_model.pth", map_location=device))
    model.eval()

    total_psnr = 0
    num_images = 0

    with torch.no_grad():
        for lr_images, hr_images in test_loader:
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            sr_images = model(lr_images)
            image_psnr = psnr(sr_images, hr_images)
            total_psnr += image_psnr.item()
            num_images += 1

    avg_psnr = total_psnr / num_images
    print(f"Average PSNR on Test Set: {avg_psnr:.2f} dB")

    wandb.log({"Test PSNR": avg_psnr})

    plot_sample_images(model, test_loader, device)


#main
model = ResidualSRCNN()

train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

train_model(model, train_loader, val_loader, DEVICE, NUM_EPOCHS)

evaluate_model(model, test_loader, DEVICE)

wandb.finish()
