import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

# Set OpenCV to use a non-threaded backend
cv2.setNumThreads(0)

# Define your ConvNet class
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
        x = F.relu(self.conv1(x))  # N, 32, 196, 196
        x = self.pool(x)  # N, 32, 98, 98
        x = F.relu(self.conv2(x))  # N, 64, 94, 94
        x = self.pool(x)  # N, 64, 47, 47
        x = F.relu(self.conv3(x))  # N, 64, 43, 43
        x = torch.flatten(x, 1)  # N, 64 * 43 * 43
        x = F.relu(self.fc1(x))  # N, 64
        x = self.fc2(x)  # N, 10
        return x

# Path to the pre-trained model
PATH = "models/COMBINED/alkene18262cnn.pth"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the pre-trained model
checkpoint = torch.load(PATH, map_location=device)  # Load model to the device
loaded_model = ConvNet().to(device)  # Move model to the device
loaded_model.load_state_dict(checkpoint)
loaded_model.eval()

# Get the last convolutional layer
final_conv = loaded_model.conv3.to(device)  # Move convolutional layer to the device

# Define hook to get feature maps
feature_map = None
def hook_fn(module, input, output):
    global feature_map
    feature_map = output

# Register hook to the final convolutional layer
hook = final_conv.register_forward_hook(hook_fn)

# Load and preprocess the image
image_path = "sorted_image_sets/COMBINED/alkene_training_data/alkenes/1856.png"  # Path to your image
image = Image.open(image_path).convert("RGB")
data_transforms = transforms.Compose([
    transforms.Resize((196, 196)),  # Resize to match the input size expected by the model
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Use the same mean and std as used during training
])
input_image = data_transforms(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Forward pass
logits = loaded_model(input_image)

# Remove the hook to avoid memory leaks
hook.remove()


# Get the predicted class index
predicted_class = torch.argmax(logits, dim=1).item()

# Compute gradients of the predicted class score with respect to the feature maps
loaded_model.zero_grad()


#####
#     OLD version starts
#####

# Compute gradients of the output with respect to the feature maps
logits[:, predicted_class].backward(retain_graph=True)

# Get the gradients from the hook
gradients = final_conv.weight.grad

# Compute CAM
cam = torch.mean(gradients, dim=[2, 3], keepdim=True)
cam = F.relu(cam)

# Resize CAM to match the input image size
cam = F.interpolate(cam, size=(196, 196), mode='bilinear', align_corners=False)
cam = cam[0, 0]

# Normalize CAM and move it to CPU memory
cam = (cam - cam.min()) / (cam.max() - cam.min())
cam = cam.cpu().numpy()  # Move to CPU memory and convert to NumPy array

# Handle NaN values, if any
cam = np.nan_to_num(cam)

# Convert to heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

# Resize the heatmap to match the size of the original image
heatmap_resized = cv2.resize(heatmap, (image.width, image.height))

# Overlay heatmap on the original image
overlayed_img = cv2.addWeighted(np.array(image), 0.5, heatmap_resized, 0.5, 0)

# Convert the original image to grayscale
gray_original = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
gray_original = cv2.cvtColor(gray_original, cv2.COLOR_GRAY2RGB)



# Save the grayscale original image
original_save_path = "cam_images/original_image.jpg"
cv2.imwrite(original_save_path, gray_original)
print(f"Grayscale original image saved at: {original_save_path}")

# Save the heatmap
heatmap_save_path = "cam_images/heatmap.jpg"
cv2.imwrite(heatmap_save_path, heatmap_resized)
print(f"Heatmap saved at: {heatmap_save_path}")

print(cam.min(), cam.max())


#########
# OLD VERSION ENDS
#########
# # Forward pass
# logits = loaded_model(input_image)

# # Compute gradients of the output with respect to the feature maps
# logits[:, predicted_class].backward(retain_graph=True)

# # Check if gradients are properly computed
# if feature_map.grad is None:
#     raise RuntimeError("Gradients are not properly computed. Check the backward pass.")

# # Get the gradients from the hook
# gradients = feature_map.grad.cpu().numpy()  # Move gradients to CPU memory and convert to NumPy array

# # Compute CAM
# cam = np.mean(gradients, axis=(2, 3), keepdims=True)
# cam = np.maximum(cam, 0)  # ReLU operation

# # Resize CAM to match the input image size
# cam = cv2.resize(cam[0, 0], (image.width, image.height))

# # Normalize CAM
# cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

# # Convert to heatmap
# heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

# # Overlay heatmap on the original image
# overlayed_img = cv2.addWeighted(np.array(image), 0.5, heatmap, 0.5, 0)

# # Convert the original image to grayscale
# gray_original = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
# gray_original = cv2.cvtColor(gray_original, cv2.COLOR_GRAY2RGB)

# # Save the grayscale original image
# original_save_path = "cam_images/original_image.jpg"
# cv2.imwrite(original_save_path, gray_original)
# print(f"Grayscale original image saved at: {original_save_path}")

# # Save the heatmap
# heatmap_save_path = "cam_images/heatmap.jpg"
# cv2.imwrite(heatmap_save_path, heatmap)
# print(f"Heatmap saved at: {heatmap_save_path}")

# print(cam.min(), cam.max())