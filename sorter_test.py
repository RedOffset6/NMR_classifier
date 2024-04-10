####################################
#                                  #
#       WHAT THIS FILE DOES        #
#                                  #
####################################

#The image sorter attempts to sort molecules by functional group

#here i print a random sample of 100 molecules from the sorter and then manually check them to see if they all contain the correct functional group

import random
import sys
from index_generation import read_pickle, molecule_plotter

trait = sys.argv[1]

ester_present = read_pickle("ester_list.pkl")

# Set a seed for reproducibility
seed_value = 42
random.seed(seed_value)

def test_molecule(functional_group = "ester"):
    fg_present = read_pickle(f"molecule_lists/{functional_group}_list.pkl")
    # Randomly sample 100 elements from the original list
    sampled_list = random.sample(fg_present, 100)

    # Print or use the sampled_list as needed
    index_smile_dict = read_pickle("index_smile_dict.pkl")

    for index in sampled_list:
        molecule_plotter(index_smile_dict[index], filename = f"sorter_test_outputs/{functional_group}_test_outputs/{index}.png")

test_molecule(functional_group = trait)
