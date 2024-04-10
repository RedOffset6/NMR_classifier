import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(3, 32, 5)  # Increase the kernel size for larger input
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)  # Increase the kernel size
        self.conv3 = nn.Conv2d(64, 64, 5)  # Increase the kernel size
        # Calculate the linear layer input size based on the dimensions of the output from the last convolutional layer
        self.fc1 = nn.Linear(64 * 42 * 42, 64)  # Adjust the linear layer input size
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        #print(f"The starting shape of the nn is: {x.shape}")
        x = F.relu(self.conv1(x))  # N, 32, 196, 196
        #print(f"The current shape is : {x.shape}")
        x = self.pool(x)  # N, 32, 98, 98
        #print(f"The current shape is : {x.shape}")
        x = F.relu(self.conv2(x))  # N, 64, 94, 94
        #print(f"The current shape is : {x.shape}")
        x = self.pool(x)  # N, 64, 47, 47
        #print(f"The current shape is : {x.shape}")
        x = F.relu(self.conv3(x))  # N, 64, 43, 43
        #print(f"The current shape is : {x.shape}")
        x = torch.flatten(x, 1)  # N, 64 * 43 * 43
        #print(f"The current shape is : {x.shape}")
        x = F.relu(self.fc1(x))  # N, 64
        #print(f"The current shape is : {x.shape}")
        x = self.fc2(x)  # N, 10
        #print(f"The final shape is : {x.shape}")
        return x
    

# Path to my pre trained model
PATH = "models/alkene18262cnn.pth"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
checkpoint = torch.load(PATH, map_location=device)  # Load model to the device
loaded_model = ConvNet().to(device)  # Move model to the device
loaded_model.load_state_dict(checkpoint)
loaded_model.eval()


# Load and preprocess the image
image_path = "images/1858.png"  # Path to the image
image = Image.open(image_path).convert("RGB")
data_transforms = transforms.Compose([
    transforms.Resize((196, 196)),  # Resize to match the input size expected by the model
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalises the image
])

#transforms the image
input_image = data_transforms(image).unsqueeze(0).to(device)

# I used a batch size of 10 when the mdoel was trained so I create an input batch which includes the image x10
batch_size = 10
batch_images = input_image.repeat(batch_size, 1, 1, 1)

#has a go at making the class activation map 
with SmoothGradCAMpp(loaded_model) as cam_extractor:
  # Preprocess your data and feed it to the model
  out = loaded_model(batch_images.unsqueeze(0))
  # Retrieve the CAM by passing the class index and the model output
  activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

# Visualize the raw CAM
plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()