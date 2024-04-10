####################################
#                                  #
#       WHAT THIS FILE DOES        #
#                                  #
####################################

#The main aim of this file is to create a dictionary which stores the smile string of each molecule in the qm9bg dataset with its qm9bg index as the key

#The dictionary should look like this:

# +----------------------------------------+
# |         index_smile_dict.pkl           |
# |    +---------+       +---------+       |
# |    |   1111  |  ->   |  C=CCCC |       |
# |    +---------+       +---------+       |
# |    |   1112  |  ->   |  CCCOCC |       |
# |    +---------+       +---------+       |
# |    |   1113  |  ->   |  CCC=CC |       |
# |    +---------+       +---------+       |
# |    |   ....  |  ->   |  .....  |       |
# |    +---------+       +---------+       |
# +----------------------------------------+

#This is then saved as index_smile_dict.pkl so it can be easily read by other files

#This dictionary was nessecary because each cosy images is saved as index.png e.g 11123.png
#To classify these images you need to access their smile string 
#This means that in pre training the index_smile_dict needs to be read in and the smile strings accessed to see the structure of each molecule

#This file also contains a range of helper function which are used in other python files

#These include molecule_plotter which takes in a smile string and saves a .png of the molecular structure and read and save to pickle which read and write from pickle files

####################################
#                                  #
#             PROBLEMS             #
#                                  #
####################################

#This file is unable to process nitro containing compounds and a try and except has been used to filter out the failed molecules
#Some molecules are saves as NaN if they are messed up for some reason
#These problems should both be fixed to better cover the chemical space


from rdkit import Chem
import pandas as pd
import numpy as np
from rdkit.Chem import PandasTools
import glob
import os
import pickle

def get_smile_string(input_path):
    #some molecules failed with an error which stated
    try:
        #Create an SDF MolSupplier object
        sdf_supplier = Chem.SDMolSupplier(input_path)

        # Iterate over molecules in the SDF file
        for mol in sdf_supplier:
            if mol is not None:
                # Do something with the molecule (e.g., print or process it)
                mol.GetPropsAsDict()

        ##use an rdkit atribute to get smile uistrings
        smile_string = Chem.MolToSmiles(mol)
        return smile_string
    except:
        return "NaN"

def get_index(input_path):
    #splits the path to get the filename
    split_file_path = input_path.split("/")

    #extracts the index from the file name
    molecule_index = split_file_path[5][6:-13]
    return molecule_index

def index_smile_dict_generator(qm9bg_filepath = "../data/datasets/qm9bg"):
    #gets a list of all of the names of all the batch files in the dataset
    batch_list = os.listdir(qm9bg_filepath) 

    #the final file is a nmrlog file which we dont need so I remove that here
    batch_list = batch_list [:-1]

    #genertates a list which will hold each index and smile string
    index_and_smile = {}
    
    #iterates through every batch in the set
    for batch in batch_list:
        #constructs a file path for this specific batch
        batch_file_path = f"{qm9bg_filepath}/{batch}"
        #generates a list of all of the sdf filenames in the batch
        sdf_list = os.listdir(batch_file_path)

        #loops through every sdf file in the sdf path 
        for sdf_file_name in sdf_list:
            sdf_path = f"{batch_file_path}/{sdf_file_name}"

            #gets the index and generates a smile string
            index = get_index(sdf_path)
            smile_string = get_smile_string(sdf_path)

            index_and_smile[index] = smile_string
            
    return index_and_smile  

#a function which saves a dictionary to a pickle file
def save_to_pickle(data, filename = "index_smile_dict.pkl"):
    with open(filename, "wb") as pickle_file:
        pickle.dump(data, pickle_file)

#a function which reads a dictionary from a .pkl file
def read_pickle(filename):
    with open(filename, 'rb') as pickle_file:
        loaded_data = pickle.load(pickle_file)
        return loaded_data

def main():
    #generates a dictionary which contains each smile string indexed by the qm9bg index
    index_smile_dict = index_smile_dict_generator()
    
    #saves the dictionary to a .pkl file
    save_to_pickle(index_smile_dict)

def molecule_plotter(smile_string, filename = "molecule.png"):
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import AllChem

    # Convert SMILES to a molecule object
    mol = Chem.MolFromSmiles(smile_string)

    # Check if the conversion was successful
    if mol is None:
        print("Invalid SMILES string")
    else:
        # Generate 2D coordinates
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)

        # Plot the molecule
        img = Draw.MolToImage(mol, size=(300, 300))
        img.save(filename)

        ###img.close()    ??????

from PIL import Image
def img_read_and_plot(image_path):

    img = Image.open(image_path)

    # Get image dimensions
    width, height = img.size
    print(f"Image dimensions: {width} x {height}")

    # Convert the image to grayscale
    img_gray = img.convert("L")

    # Save the modified image
    img_gray.save("grayscale_image_1530.png")

    img.close()

    




