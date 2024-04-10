####################################
#                                  #
#       WHAT THIS FILE DOES        #
#                                  #
####################################

#The main purpose of this file is to generate a mock cosy image from the NMR data stored in the qm9bg data
#It is passed an input path (the path to a sdf file) and an output path (a directory to save the .png in)
#Currently the .png stores a mock cosy which represents every H-H coupling in a correlational sprectrum
#The intensity of the point is dependent on the coupling constant


####################################
#                                  #
#             PROBLEMS             #
#                                  #
####################################

#This file is pretty messed up and has the following major problems

#1)
#it throws errors for a huge fraction of molecules (probably mostly nitro resonance structures)

#2)
#It also seems to contain some type of terrible memory leak
#when the function is called it appears to use up a new space in ram each time
#to get 70,000 molecule images saved 128GB of ram on blue crystal was needed
#this is insane as the image is saved each time the function is called so there should be no reason
#to use any more ram than what is needed to store one image

#3)
#The function is super nested and should probably be flattened into several smaller functions 
#This would improve readability

#4)
#The looping and image saving are both probably very inefficient and mean that heaps of resorces are needed
#to run the file

#5)
#The string slicing was rushed really badly when I wrote it and is unreadable and non portable
#I also dont really understand what I did so there are almost certanly terrible unforseen consequences
#It is also probably terribly inefficient adding to the undesirable walltime

#6)  FIXED -> Now takes slices ../data/123.sdf   into [".."" , "data" , "123.sdf"] and returns list[-1] to get the sdf name
#The program fails when imputted with paths of different lengths when it slices to get the sdf name
#

from rdkit import Chem
import pandas as pd
import numpy as np
#from rdkit.Chem import PandasTools

import matplotlib.pyplot as plt
#hides unhelpful error messages
import warnings
warnings.filterwarnings("ignore")

    


#a function which reads an sdf file and generates a normalised cosy spectrum which is saved as a .png
def generate(input_path, output_path):
    sdf_file_path = input_path
    try:
        
        #splits the path to get the filename
        sdf_name = input_path.split("/")

        #extracts the index from the file name
        molecule_index = sdf_name[-1][6:-13]
        #
        #commented out this line to try to fix the memory leak
        #

        # #Create an SDF MolSupplier object
        # sdf_supplier = Chem.SDMolSupplier(sdf_file_path)

        
        ####################################################################################################
        #                                                                                                  #
        #     CHECK OUT THIS!!!!!!!!!!!                                                                    #
        #                                                                                                  #
        ####################################################################################################



        #added this line to try to fix the memory leak
        with Chem.SDMolSupplier(sdf_file_path) as sdf_supplier:
            print("with statement active")
            # Iterate over molecules in the SDF file
            for mol in sdf_supplier:
                if mol is not None:
                    # Do something with the molecule (e.g., print or process it)
                    mol.GetPropsAsDict()

            ##use an rdkit atribute to get smile uistrings
            smile_string = Chem.MolToSmiles(mol)

            ####################################################################################################
            #                                                                                                  #
            #     Section which finds the indexes of all of the hydrogen atoms and stores these in a list      #
            #                                                                                                  #
            ####################################################################################################


            #adds the hydrogens to the molecule
            mol = Chem.AddHs(mol)

        #loops through the atoms in the molecule and hunts for hydrogens
        hydrogen_list = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                atom_number = atom.GetIdx()
                hydrogen_list.append(atom_number)


        ############################################################################
        #                                                                          #
        #     section which does string slicing to extract the hydrogen atoms      #
        #                                                                          #
        ############################################################################

        import re

        #gets a string which contains all the data stored past NMREDATA_ASSIGNMENT
        raw_data = str(mol.GetProp("NMREDATA_ASSIGNMENT"))

        # Extract NMR shifts
        nmr_shifts = re.findall(r'(\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*\d+\s*,\s*-?\d+\.\d+', raw_data)

        # Convert the extracted data into a dictionary
        nmr_shift_dict = {int(match[0]): float(match[1]) for match in nmr_shifts}

        ############################################################################
        #                                                                          #
        #          Correcting the shifts from DFT to literature emulated           #
        #                                                                          #
        ############################################################################
        print ("HELLO4")

        #                     A        B
        #pymol scaling is [-1.0594, 32.2293]
        # true shift = (DFT_shift - B / A)

        #a function which converts from dft shift to experimental
        def dft_scaler(dft_shift):
            A = -1.0594
            B = 32.2293
            corrected_shift = (dft_shift - B) / A
            return corrected_shift


        #loops through the H1 shifts and interpolates each value to one which matches experimental data
        corrected_shift_list = []

        #hacky error handling to stop molecules crashing that have atoms indeces in their atom index list which are out of range for the nmr_shift_dict
        for atom_index in hydrogen_list:
            try:
                nmr_shift_dict[atom_index] = dft_scaler(nmr_shift_dict[atom_index])
                corrected_shift_list.append(nmr_shift_dict[atom_index])
            except:
                print(f"ERROR molecule with index {molecule_index} failed to compute because of index ranging issues")
                return False

        ############################################################################
        #                                                                          #
        #                      Extracting HH coupling constants                    #
        #                                                                          #
        ############################################################################

        #a function which takes mol.GetProp("NMREDATA_J") and returns a pandas dataframe of J data
        def extract_j(raw_data):
            lines = raw_data.split('\n')

            j_data = []
            for line in lines:
                if line.strip():
                    parts = line.strip().split(',')
                    j_data.append({
                        'atom1': int(parts[0].strip()),
                        'atom2': int(parts[1].strip()),
                        'coupling_const': float(parts[2]),
                        'j': parts[3].strip()[:2],
                        'type': parts[3].strip()[2:]
                    })

            return pd.DataFrame.from_dict(j_data)
        #This function will start collecting data when it encounters a line with 'NMREDATA_J' and stop when it encounters another 'NMREDATA_J' line or the end of the input. It will only extract data from the 'NMREDATA_J' section.
        print("hello X")
        #extracts the J data in the form of a pd dataframe
        j_data = extract_j(mol.GetProp("NMREDATA_J"))

        #filters for HH coupling
        HH_data = j_data[j_data["type"]=="HH"]

        #normalises the coupling constants
        if len(HH_data["coupling_const"]) == 1:
                HH_data["norm_cc"] = 1
        else:
            HH_data["norm_cc"] = 2*((HH_data["coupling_const"] - np.min(HH_data["coupling_const"]))   /   (np.max(HH_data["coupling_const"]) - np.min(HH_data["coupling_const"]))       )  - 1

        ############################################################################
        #                                                                          #
        #                  Plots and saves cosy Correlation spectrum               #
        #                                                                          #
        ############################################################################

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.scatter([0,1],[0,1])

        print ("HELLO5")
        fig.savefig(f"{output_path}/{molecule_index}.png")

        #creates a figure and ax and then plots the spectrum
        print(corrected_shift_list)
        print("HELLO pspsp")
        
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        print("HELLO Z")
        
        ax.scatter(corrected_shift_list, corrected_shift_list, c  ="green")

        size = 100
        print("hello Y")



        #loops through all of the HH combinations and plots
        for index, row in HH_data.iterrows():
            # coulours points red or blue depending on their phase
            if row["norm_cc"] >= 0:
                ax.scatter(nmr_shift_dict[row['atom1']], nmr_shift_dict[row['atom2']], alpha = row["norm_cc"], c  ="red", edgecolors = "none", s = size*row["norm_cc"])
                ax.scatter(nmr_shift_dict[row['atom2']], nmr_shift_dict[row['atom1']], alpha = row["norm_cc"], c  ="red", edgecolors = "none", s = size*row["norm_cc"])
            else:
                ax.scatter(nmr_shift_dict[row['atom1']], nmr_shift_dict[row['atom2']], alpha = row["norm_cc"]*-1, c  ="blue", edgecolors = "none", s = size*row["norm_cc"]*-1)
                ax.scatter(nmr_shift_dict[row['atom2']], nmr_shift_dict[row['atom1']], alpha = row["norm_cc"]*-1, c  ="blue", edgecolors = "none", s = size*row["norm_cc"]*-1)



        #sets x and y limits so that all graphs are normnalized to teh same axis
        plt.xlim(-2, 12)
        plt.ylim(-2, 12)

        #removes the axis
        plt.axis('off')

        #inverts axes to emulate the backwards nature of NMR plots
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()


        #saves the plot to a file
        
        #annoyingly some mile strings contain "/" this confuses the file system
        #as a result i have commented out this line and removed the smile string from the file name
        
        #fig.savefig(f"{output_path}/{index},{smile_string}.png")
        print ("HELLO5")
        fig.savefig(f"{output_path}/{molecule_index}.png")
                    
                    
                    
        print ("HELLO6")
        plt.close(fig)
        
    except:
        print(f"MOLECULE {molecule_index} FAILED")


generate(f"../data/datasets/qm9bg/batch1/qm9bg_118.nmredata.sdf", "experimental_images")