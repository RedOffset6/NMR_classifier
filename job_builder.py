import numpy as np
import glob
import os
import sys
import pickle as pkl

#Taking arguments
experiment_type = sys.argv[1]


path_to_batches = f"qm9_sdf_files"
output_path = f"{experiment_type}_images"


#gets a list of all of the batches
batches  = os.listdir(path_to_batches)

#remove nmr log files from the batch list 
#batches = batches[:-1]

job_list = []

#loops throigh each batch 
for batch in batches:
    #builds a path to the batch folder
    path_to_sdf_files = f"{path_to_batches}/{batch}"

    #gets a list of all of the sdf filenames
    sdf_filenames = os.listdir(path_to_sdf_files)
    print(f"there are {len(sdf_filenames)} sdf files in the sdf file name list ")
    for sdf_file in sdf_filenames:
        job = {}

        job["qm9_index"] = sdf_file[6:-13]
        job["batch"] = batch
        job["sdf_filepath"] = f"{path_to_sdf_files}/{sdf_file}"
        job["output_filepath"] = f"{output_path}/{batch}"
        job["status"] = "unattempted"
        job["image_type"] = experiment_type
        
        job_list.append(job)
        
print(f"There are {len(job_list)} jobs in the final job list")
with open(f'job_files/{experiment_type}_job_file.pkl', 'wb') as f:
   pkl.dump(job_list, f)
