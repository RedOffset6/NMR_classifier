####################################
#                                  #
#       WHAT THIS FILE DOES        #
#                                  #
####################################

#To train a pytorch neural network it needs the data to be preclassified and stored in seperated directories

# an example of this is shown below:

# data/
# ├── cat/
# │   ├── cat_image1.jpg
# │   ├── cat_image2.jpg
# │   └── ...
# ├── dog/
# │   ├── dog_image1.jpg
# │   ├── dog_image2.jpg
# │   └── ...
# ├── bird/
# │   ├── bird_image1.jpg
# │   ├── bird_image2.jpg
# │   └── ...
# └── ...

#this file starts by reading in lists of indeces from the .pkl data and then finds the images in the images filestructure
#it then saves the images in a new file system which matches the scheme shown above
#it also randomly samples the images to make sure that each class has roughly the same amount of data
#e.g we wouldnt want 1000 alkenes and 100000 non alkenes as the model would probably just overtrain

####################################
#                                  #
#             PROBLEMS             #
#                                  #
####################################

#None yet some to come ;)


############################
#                          #
#         IMPORTS          #
#                          #
############################

from index_generation import read_pickle
import random
import os
import shutil
import sys

####################################
#                                  #
#         READING IN DATA          #
#                                  #
####################################

trait = sys.argv[1]
experiment_type = sys.argv[2]

#reads in the alkene and non alkene lists
positives = read_pickle(f"molecule_lists/{trait}_list.pkl")
non_positives = read_pickle(f"molecule_lists/non_{trait}_list.pkl")

######################################################
#                                                    #
#         SELECTING A SUBSET OF NON-ALKENES          #
#                                                    #
######################################################

#Here I select a subset of non alkenes so the alkene and non alkene filestructures are equally balenced

#seeding the random sampler
random.seed(42)
sample_size = len(positives)

#sampling the non alkenes to pick the same number of non alkenes as alkenes
sampled_non_positives = random.sample(non_positives, sample_size)

######################################################
#                                                    #
#         making index -> batch dictionaries         #
#             for alkenes and non alkenes            #
#                                                    #
######################################################

#in this section i want to go from having a list of all of the alkene indexes to having a dictionary which contains the index and a key
#and the batch that this index is stored in as the data
#this will allow me to easily access each png file when I copy them later
#otherwise I would have to do some kind of annoying search

#i acchieve this by looping through every image and batch and then checking if this index is in the alkene list
# if it is I store this image and its batch in the dictionary 

#creates dictionaries for the alkenes and non alkenes
positive_batch_dict = {}
non_positive_batch_dict = {}

#gets a list of the batches
batch_list = os.listdir(f"../image_generation/{experiment_type}_images")

#loops through all of the batches in the batch directory
try:
    for batch in batch_list:
        #generates a list of all of the png files in the current batch
        png_list = os.listdir(f"../image_generation/{experiment_type}_images/{batch}")
        print(f"PRINTING PNG LIST {png_list}")
        #does some string slicing to create a new list which has stripped the indexes out of the .png file names
        #cuts off the HSQC handel from the file name
        if experiment_type == "HSQC" or experiment_type == "COSY" or experiment_type == "HMBC":
            indeces = [entry.split('.')[0][:-4] for entry in png_list]
        elif experiment_type == "COMBINED":
           indeces = [entry.split('.')[0][:-8] for entry in png_list]            
        else:
            indeces = [entry.split('.')[0] for entry in png_list]
        print(f"Printing indeces {indeces}")
        #loops through each of these indeces
        for index in indeces:
            #checks if the index is an alkene and if so adds the batch to the dictionary
            if index in positives:
                positive_batch_dict[index] = batch
            #same as above for non alkenes
            if index in sampled_non_positives:
                non_positive_batch_dict[index] = batch
except:
    print(f"Image {index} was not in the {experiment_type} dataset")

######################################################
#                                                    #
#         SAVING IMAGES TO THE NEW DIRECTORY         #
#                                                    #
######################################################

# now I can loop though the alkene and non alkene datasets and copy accross the images to the new filestructure
#i decided to try to use hard linking rather than copying to avoid using extra storage space and processing power
#time will tell whether this was a terrible error or not

#A function which copies files found in a list of indeces into a given ouutput file path
def file_transferer(index_list, index_batch_dict, class_name, output_filepath_stem = f"sorted_image_sets/{experiment_type}/{trait}_training_data"):
    print(f"index batch dict {index_batch_dict}")
    for index in index_list:
        batch = index_batch_dict[index]
        
        #constructs source and destinatin filepaths
        if experiment_type == "HSQC" or experiment_type == "COMBINED" or experiment_type == "COSY" or experiment_type == "HMBC":
            source_filepath = f"../image_generation/{experiment_type}_images/{batch}/{index}{experiment_type}.png"
        else:
            source_filepath = f"../image_generation/{experiment_type}_images/{batch}/{index}.png"
        destination_dir = f"{output_filepath_stem}/{class_name}"
        
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)


        destination_file = os.path.join(destination_dir, f'{index}.png')
        
        #creates a hard link with shutil2
        print(f"trying to copy from {source_filepath} to {destination_file}")



        shutil.copy2(source_filepath, destination_file)

#transfers files for alkenes and non alkenes

#actually calling the functions and moving the files


file_transferer(positives, positive_batch_dict, f"{trait}s")
file_transferer(sampled_non_positives, non_positive_batch_dict, f"non_{trait}s")