from cosy_generator import generate
import numpy as np
import glob
import os
import sys

#takes command line input for the batch
batches = sys.argv[1]

#batches = os.listdir("data")
output_path = "experimental_images"

#removes nmr log files from batches list
#batches = batches[:-1]

#loops through the batches if only one batch is submitted the list only includes one batch
for i, batch in enumerate(batches):
    print(f"{batch} PROCESSING HAS BEGUN")

    #
    #BC4 Version
    #

    # #creates a path for the batch and then generates a list of teh names of every sdf file in the batch
    # path = f"data/batch{batch}"
    # sdf_list  = os.listdir(path)
     
    # 
    #LOCAL VERSION
    #
    
    #creates a path for the batch and then generates a list of the names of every sdf file in the batch
    path = f"../data/datasets/qm9bg/batch{batch}"
    sdf_list  = os.listdir(path)


    #creates a list of percentages and a corresponding list of file indeces which correspond to a given percentage
    percentage_array = np.array([10,20,30,40,50,60,70,80,90])
    batch_size = len(sdf_list)
    checkpoint_indeces = (batch_size*percentage_array*0.01).astype(int)
    
    for index, sdf_file in enumerate(sdf_list[100:120]):
        print(f" The input path is {path}/{sdf_file} The output patch is: {output_path}/batch{batch}")
        generate(f"{path}/{sdf_file}", f"{output_path}/batch{batch}")

        if index in checkpoint_indeces:
            percentage_complete = percentage_array[np.where(checkpoint_indeces==index)]

            print(f"{percentage_complete}% COMPLETE")

    print(f"BATCH {i+1} COMPLETE")
