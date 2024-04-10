####################################
#                                  #
#       WHAT THIS FILE DOES        #
#                                  #
####################################

#This file sorts through the dataset and classifies the molecules based on a certain trait e.g whether it is an alkene or not
#It then saves pickle files containing a lists of indeces which conform to the search
#e.g it saves alkenes.pkl which contains a list of the index of every alkene and non_alkenes.pkl which contains the indeces of every non alkene

####################################
#                                  #
#             PROBLEMS             #
#                                  #
####################################

#This file doesnt have major known problems
#The armoatic checker should probably be fixed if I want to use it as currently I believe it only checks for unsubstituted phenyls rather than all posible phenyls
#it could probably also better handle NaN values and invalid smile strings

from index_generation import read_pickle
from rdkit import Chem
import glob
import os
import sys

from rdkit.Chem.Fragments import fr_ketone


#importing my own stuff
from index_generation import molecule_plotter
from index_generation import save_to_pickle

# Reads the dictionary of indices
index_smile_dict = read_pickle("index_smile_dict.pkl")

trait = sys.argv[1]
overwrite = sys.argv[2]

def double_bond_checker(smile_string):
    mol = Chem.MolFromSmiles(smile_string)
    if mol:
        has_double_bond = any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in mol.GetBonds())
        if has_double_bond:
            return True
        else:
            return False
    else:
        return "Error Invalid SMILES string. Please check the format."

def alkene_checker(smile_string):
    mol = Chem.MolFromSmiles(smile_string)
    if mol:
        alkene = any((bond.GetBondType() == Chem.BondType.DOUBLE) and all(atom.GetAtomicNum() == 6 for atom in (bond.GetBeginAtom(), bond.GetEndAtom())) for bond in mol.GetBonds())
        if alkene:
            return True
        else:
            return False
    else:
        return "Error Invalid SMILES string. Please check the format."


def halogen_checker(smile_string):
    mol = Chem.MolFromSmiles(smile_string)
    
    if mol:
        #looks for the number of halogens (we know that halogens are the only molecules present)
        num_halogens = Chem.Fragments.fr_halogen(mol)
        # print(num_ketones)
        if num_halogens > 0:
            return True
        return False
    else:
        print("Error: Invalid SMILES string. Please check the format.")
        return False

def ester_checker(smile_string):
    mol = Chem.MolFromSmiles(smile_string)
    
    if mol:
        num_esters = Chem.Fragments.fr_ester(mol)
        # print(num_ketones)
        if num_esters > 0:
            return True
        return False
    else:
        print("Error: Invalid SMILES string. Please check the format.")
        return False

def aldehyde_checker(smile_string):
    mol = Chem.MolFromSmiles(smile_string)
    
    if mol:
        num_aldehydes = Chem.Fragments.fr_aldehyde(mol)
        # print(num_ketones)
        if num_aldehydes > 0:
            return True
        return False
    else:
        print("Error: Invalid SMILES string. Please check the format.")
        return False

def amide_checker(smile_string):
    mol = Chem.MolFromSmiles(smile_string)
    
    if mol:
        num_amides = Chem.Fragments.fr_amide(mol)
        # print(num_ketones)
        if num_amides > 0:
            return True
        return False

def ketone_checker(smile_string):
    mol = Chem.MolFromSmiles(smile_string)
    if mol:
        num_ketones = Chem.Fragments.fr_ketone(mol)
        # print(num_ketones)
        if num_ketones > 0:
            return True
        return False
    else:
        print("Error: Invalid SMILES string. Please check the format.")
        return False


def alcohol_checker(smile_string):
    mol = Chem.MolFromSmiles(smile_string)
    if mol:
        num_aliphatic_alcohols = Chem.Fragments.fr_Al_OH(mol)
        num_aromatic_alcohols = Chem.Fragments.fr_Ar_OH(mol)
        num_alcohols = num_aliphatic_alcohols + num_aromatic_alcohols
        if num_alcohols > 0:
            return True
        return False
    else:
        print("Error: Invalid SMILES string. Please check the format.")
        return False

def imine_checker(smile_string):
    mol = Chem.MolFromSmiles(smile_string)
    if mol:
        num_imines = Chem.Fragments.fr_Imine(mol)
        if num_imines > 0:
            return True
        return False
    else:
        print("Error: Invalid SMILES string. Please check the format.")
        return False

#BROKEN returns no thiols
def thiol_checker(smile_string):
    mol = Chem.MolFromSmiles(smile_string)
    if mol:
        num_thiols = Chem.Fragments.fr_SH(mol)
        if num_thiols > 0:
            return True
        return False
    else:
        print("Error: Invalid SMILES string. Please check the format.")
        return False


def epoxide_checker(smile_string):
    mol = Chem.MolFromSmiles(smile_string)
    if mol:
        num_epoxides = Chem.Fragments.fr_epoxide(mol)
        if num_epoxides > 0:
            return True
        return False
    else:
        print("Error: Invalid SMILES string. Please check the format.")
        return False
    
#ethe
def ether_checker(smile_string):
    mol = Chem.MolFromSmiles(smile_string)
    if mol:
        num_ethers = Chem.Fragments.fr_ether(mol)
        if num_ethers > 0:
            return True
        return False
    else:
        print("Error: Invalid SMILES string. Please check the format.")
        return False

# Define a dictionary mapping molecule traits to their corresponding checker functions
trait_checkers = {
    "alkene": alkene_checker,
    "ester": ester_checker,
    "amide": amide_checker,
    "aldehyde": aldehyde_checker,
    "halogen": halogen_checker,
    "ketone": ketone_checker,
    "alcohol": alcohol_checker,
    "imine": imine_checker,
    "thiol": thiol_checker,#broken
    "epoxide": epoxide_checker,
    "ether": ether_checker
    # Add more traits and corresponding checker functions here
}

def main(molecule_trait = "alkene", overwrite = "False"):
    positive_list = []
    negative_list = []
    positive_count = 0
    negative_count = 0
    molecule_count = 0 
    #generates a list which contains the names of each of the batch files stored in the images directory
    batch_list = os.listdir("images/COMBINED")

    # Check if a string is a key in the dictionary
    if (molecule_trait in trait_checkers) == False:
        print(f"Error {molecule_trait} is not a valid key in the dictionary of traits that this script had been written to process.")
        return

    checker_function = trait_checkers[molecule_trait]

    #loops through all of the batches in the batch directory
    for batch in batch_list:
        print(f"searching {batch}")
        #generates a list of all of the png files in the current batch
        png_list = os.listdir(f"images/COMBINED/{batch}")
        #print(png_list)
        #does some string slicing to create a new list which has stripped the indexes out of the .png file names
        indeces = [entry.split('.')[0][:-8] for entry in png_list]
        print(indeces)
        KeyboardInterrupt
        #loops through each of these indeces
        for index in indeces:
            #print(f"checking {index}")
            molecule_count +=1
            #checks if the index is in the dictionary
            if index in index_smile_dict:
                #gets the smile string
                smile_string = index_smile_dict[index]
                #checks if the smile string is a proper smiles and non a NaN
                if smile_string != "NaN":
                    #checks if this smile string is an alkene
                    if checker_function(smile_string):
                        positive_list.append(index)
                        positive_count += 1
                    elif checker_function(smile_string) == False:
                        negative_list.append(index)
                        negative_count += 1
                    else:
                        print("Error Invalid smile")
    if overwrite:
        save_to_pickle(positive_list, filename = f"molecule_lists/{molecule_trait}_list.pkl")
        save_to_pickle(negative_list, filename = f"molecule_lists/non_{molecule_trait}_list.pkl")


    print(f"There are {positive_count} valid {molecule_trait}s in the dataset")
    print(f"There are {negative_count} valid non {molecule_trait}s in the dataset")
    print(f"There are {molecule_count} molecules in the dataset")

main(molecule_trait = trait, overwrite = overwrite)