from cosy_generator import generate
import pickle as pkl
import sys

#Taking arguments
experiment_type = sys.argv[1]

#reads the job file
with open(f'job_files/{experiment_type}_job_file.pkl', 'rb') as f:
    job_list = pkl.load(f)

#print(f"There are {len(job_list)} jobs in the job list")

#loops through the jobs in the job list
for i, job in enumerate(job_list):
    #checks to see if the job has been tried
    #print(f"The status of job {i} is {job['status']}")
    
    if job["status"] == "unattempted":
        print(f"The status of job {i} is {job['status']}")
        outcome = generate(job["sdf_filepath"], job["output_filepath"], image_type=job["image_type"])
        job["status"] = outcome
        job_list[i] = job

        if outcome == None:
            job["status"] = "failed"


        #saves the file every thousand jobs
        if i % 1000 == 0:
            with open(f'job_files/{experiment_type}_job_file.pkl', 'wb') as f:
                pkl.dump(job_list, f)

#save the job file at the end of computaion 
with open(f'job_files/{experiment_type}_job_file.pkl', 'wb') as f:
    pkl.dump(job_list, f)


with open(f'job_files/{experiment_type}_job_file.pkl', 'rb') as f:
    job_list = pkl.load(f)


print(f"Final job list: \n{job_list}")