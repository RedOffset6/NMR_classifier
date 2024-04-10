#import staments
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image
import shutil

#importing some of my functions
from index_generation import read_pickle
from index_generation import molecule_plotter
from index_generation import img_read_and_plot
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


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

#importing some of my functions
from index_generation import read_pickle
from index_generation import molecule_plotter
from index_generation import img_read_and_plot

############################################################################
#                                                                          #
#     PLOTTING A FEW MOLECULES WHICH I CAN USE AS EXAMPLES FOR THE DEMO    #
#                                                                          #
############################################################################

index_smile_dict = read_pickle("index_smile_dict.pkl")

#molecule 11000 is an ether without a ketone alkene or amide
molecule_plotter(index_smile_dict["11000"], "individual_molecule_classifier/11000.png")

#an amide with a ketone
molecule_plotter(index_smile_dict["11700"], "individual_molecule_classifier/11700.png")

#ketone
molecule = "21090"
molecule_plotter(index_smile_dict[molecule], f"individual_molecule_classifier/{molecule}.png")

#alkene with an ether
molecule = "1921"
molecule_plotter(index_smile_dict[molecule], f"individual_molecule_classifier/{molecule}.png")

############################################################################
#                                                                          #
#                         DEFINING THE MODEL CLASS                         #
#                                                                          #
############################################################################

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

############################################################################
#                                                                          #
#                       INITILISING GRAPHICS CARD                          #
#                                                                          #
############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################################################
#                                                                          #
#                    GENERATING INSTANCES OF THE CLASS FOR EACH            #
#                               FUNCITIONAL GROUP                          #
#                                                                          #
############################################################################

#
#LOADING AMIDE MODEL
#

subset_size = "16464"

PATH = f"./amide{subset_size}cnn.pth"

amide_model = ConvNet()
amide_model.load_state_dict((torch.load(PATH)))
amide_model.to(device)
amide_model.eval()

############################################################################
#                                                                          #
#                 DEFINING A FUNCTION WHICH CAN FIND THE BATCH             #
#                               OF A MOLECULE                              #
#                                                                          #
############################################################################

#A function which is parsed a qm9 index and retuns the batch number

def batch_finder(index):
    #gets a list of the batches
    batch_list = os.listdir("images")

    #loops through all of the batches in the batch directory
    for batch in batch_list:
        #generates a list of all of the png files in the current batch
        png_list = os.listdir(f"images/{batch}")

        #checks to see if this image is in the batch
        if f"{index}.png" in png_list:
            return batch

        #If the image isnt found in any of the folders prints an error message

        print(f"Error an image file corresponding to {index}.png was not found in any of the batch folders")

############################################################################
#                                                                          #
#                            LOADING THE IMAGES                            #
#                           OF THE TEST MOELCULES                          #
#                                                                          #
############################################################################


#
#A function which is parsed the index of an image and returns a transformed tensor which can be loaded into the neural network
#

#I did this in a messed up way. i couldnt find a good way to read a single image into a pytorch tensor but could read a folder
#to rectify this I created a hack folder which will only every contain one image

def image_loader(index):
    #getting the batch
    batch = batch_finder(index)

    #creates a file path with the batch and image name
    image_path = f"images/{batch}/{index}.png"
    
    #deletes the contents of the hack folder
    file_list = os.listdir("hack_folder/hack_class")
    for file in file_list:
        os.remove(f"hack_folder/hack_class/{file}")

    #copys the image to the hack folder
    shutil.copy2(image_path, "hack_folder/hack_class")

    data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((196, 196))
    ])

    normalised_image = ImageFolder(root=f'hack_folder', transform=data_transforms)
    #return transformed_image

    image_loader = DataLoader(normalised_image, batch_size=10)

    return image_loader
#print(image_loader("11700"))

# loaded_image = image_loader("11700")



# # Get the first batch
# dataiter = iter(loaded_image)
# images, _ = next(dataiter)

# # Access the shape of the first image
# image_shape = images[0].shape
# print("HELLO\n\n\n\n")
# print("Shape of the first image:", image_shape)
# print("HELLO\n\n\n\n\n")


# with torch.no_grad():
#     for images, labels in loaded_image:
#         images = images.to(device)
#         #labels = labels.to(device)
#         outputs = amide_model(images)
        
#         #max returns value, index
#         _, predicted = torch.max(outputs, 1)

# print (f"The value stored in _ is as follows: {_}")
# print(f"The value stored in predicted is as follows {predicted}")

#A function which takes a qm9 index and looks for a certain trait
#Then returns a prediction of whether or not that trait is present

#IMPORTANT!!!
#returning zero means that the trait has been detected
#this because the model was trained to identify the molecule as being part of one of two classes with the first class being 
#molecules which have the desired trait and the second class being molecules which do not

#  For example in the case of alkenes the classes are as follows
#
#  class 0: alkenes
#  class 1: non_alkene


def load_and_test(index, trait):
    #loads the image
    loaded_image = image_loader(index)

    #loads the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #defines a dictionary which holds the best model for each trait
    training_size_dict = {'amide': '16464',
                          'alkene': '18262', 
                          'ketone': '17706'}

    #creates the model path 
    subset_size = "16464"
    PATH = f"./{trait}{training_size_dict[trait]}cnn.pth"

    #initilises model
    loaded_model = ConvNet()
    loaded_model.load_state_dict((torch.load(PATH)))
    loaded_model.to(device)
    loaded_model.eval()
    
    #generates a prediction
    with torch.no_grad():
        for images, labels in loaded_image:
            images = images.to(device)
            #labels = labels.to(device)
            outputs = loaded_model(images)
            
            #max returns value, index
            _, predicted = torch.max(outputs, 1)
    
    return predicted

print(load_and_test("1", "alkene"))


#runs tests a varaity of functional groups for a given molecule
def batch_test(index):
    print(f"\nRESULTS FOR MOLECULE {index}\n")
    if load_and_test(index, "alkene"):
        print("ALKENE : NEGATIVE")
    else:
        print("ALKENE : POSITIVE")

    if load_and_test(index, "amide"):
        print("AMIDE : NEGATIVE")
    else:
        print("AMIDE : POSITIVE")

    if load_and_test(index, "ketone"):
        print("KETONE : NEGATIVE")
    else:
        print("KETONE : POSITIVE")

batch_test("11700")

batch_test("1921")


# batch_test("11000")

# #why wasnt this found in the batch folders???????
# batch_test("2098")

# batch_test("1090")