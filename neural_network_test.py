####################################
#                                  #
#       WHAT THIS FILE DOES        #
#                                  #
####################################

#this file aims to create a convolutional neural network which can determine between the mock cosy images I have gernerates
#in the mk 1 version I aim to determine between ~20000 molecules. ~10000 alkenes and ~10000 non alkenes
#these alkenes are stored in the following datastructure

# alkene_training_data/
# ├── alkenes/
# │   ├── 1111.png
# │   ├── 2222.png
# │   └── ...
# └── non_alkenes/
#     ├── 3333.png
#     ├── 4444.png
#     └── ...



#importing packages
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets
import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import matplotlib as plt
import sys
import pandas as pd

#importing some of my functions
from index_generation import read_pickle
from index_generation import molecule_plotter
from index_generation import img_read_and_plot
import pickle as pkl

# trait = sys.argv[1]
# subset_size = int(sys.argv[2])
# experiment_type = sys.argv[3]
# PATH = sys.argv[1]
# print(f"PATH = {PATH}")

############################################################################
#                                                                          #
#       READS A DICTIONARY FROM DISK WHICH STORERES SMILE STRINGS          #
#                    WITH QM9DB INDECES AS THEIR KEYS                      #
#                                                                          #
############################################################################

# index_smile_dict = read_pickle("index_smile_dict.pkl")

# molecule_plotter(index_smile_dict["1530"])

# alkene_list = read_pickle("alkene_list.pkl")

############################################################################
#                                                                          #
#       READS THE IMAGES INTO PYTORCH TENSORS AND SETS THE DEVICE TO       #
#                          BE THE GRAPHICS CARD                            #
#                                                                          #
############################################################################

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset, SubsetRandomSampler

import openpyxl

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_epochs = 10
batch_size = 10
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import confusion_matrix

######################################################################################################

#TRYING 200X200 NETWORK


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

#PATH = f"./models/{experiment_type}/{trait}{subset_size}cnn.pth"

loss_df = pd.DataFrame()
confusion_matrices = {}
accuracy_df = pd.DataFrame()


#PATH = "models/COMBINED/alkene/[0.01]alkene150cnn.pth" 
experiment_type_list  = os.listdir("models/high_epoch")
for experiment_type in experiment_type_list:
    functional_groups = os.listdir(f"models/high_epoch/{experiment_type}")
    for functional_group in functional_groups:
        models = os.listdir(f"models/high_epoch/{experiment_type}/{functional_group}")
        for model in models:
            PATH = f"models/high_epoch/{experiment_type}/{functional_group}/{model}"

            with open(f'data_loaders/{experiment_type}/test_loaders/{functional_group}_test_loader.pkl', 'rb') as f:
                test_loader = pkl.load(f)

            print(f"the length of the test loader is {len(test_loader)}")


            checkpoint = torch.load(PATH)
            running_losses = checkpoint["running_losses"]

            print(f"printing model running losses: {running_losses}")

            loaded_model = ConvNet()
            loaded_model.load_state_dict(checkpoint["model_state_dict"])
            loaded_model.to(device)
            loaded_model.eval()

            count = 0 

            print(test_loader)

            print("Batch size:", test_loader.batch_size)
            #print("Shuffle:", test_loader.shuffle)
            print("Sampler:", test_loader.sampler)

            with torch.no_grad():
                n_correct = 0
                n_correct2 = 0
                n_samples = len(test_loader.dataset)
                all_labels = []
                all_predictions = []
                
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = loaded_model(images)
                    


                    #max returns value, index
                    _, predicted = torch.max(outputs, 1)
                    n_correct += (predicted == labels).sum().item()

                        # Append true and predicted labels to the lists
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

                    # print(f"Prediction = {predicted}")
                    # print(f"Labels = {labels}")
                    count = count + len(predicted)

            # Calculate the confusion matrix
            cm = confusion_matrix(all_labels, all_predictions)

            # Define class labels
            classes = ['Negative', 'Positive']  # Update with your class labels

            # Print the confusion matrix
            print
            print("Confusion Matrix:")
            print(classes)
            print(cm)

            # Print the class labels
            print("Class Labels:")
            print(classes)


            acc = 100.0 * n_correct / count

            print(f"there were {n_correct} correct attempts")

            print(f"There were {count} samples")
            print(f"there were {n_samples} images in the test dataset")
            print(PATH)
            print(f"The accuracy of the model is {acc}%")
            
            loss_df[f"{model},{experiment_type}"] = running_losses
            accuracy_df.loc["accuracy", f"{model},{experiment_type}"] = acc
            confusion_matrices[f"{model},{experiment_type}"] = cm
            print("helloooo")
            print(f"{model}, {acc}")#
            print("hellooo")

#### new code
#             incorrect_count = 0
#             for images, labels in test_loader:
#                 images = images.to(device)
#                 labels = labels.to(device)
#                 outputs = loaded_model(images)
#                 _, predicted = torch.max(outputs, 1)
                
#                 # Iterate through the batch
#                 for i in range(len(predicted)):
#                     if predicted[i] != labels[i]:
#                         # Get the filename of the misclassified image
#                         filename = test_loader.dataset.samples[i][0]
#                         print(f"Incorrectly classified image: {filename}")
#                         incorrect_count += 1
                        
#                         # Stop after printing 10 misclassified images
#                         if incorrect_count == 10:
#                             break
    
# ###new code
#     # Stop after printing 10 misclassified images
#     if incorrect_count == 10:
#         break

loss_df.to_excel('high_epoch_loss_curves.xlsx', index=False)
# accuracy_df.to_excel('accuracy.xlsx', index=False)

# #saving confusion matrices
# pickle_filename = "confusion_matrices.pkl"
# with open(pickle_filename, 'wb') as f:
#     pkl.dump(confusion_matrices, f)





