import pickle as pkl
import sys

experiment_type = sys.argv[1]

#reads the job file
with open(f'job_files/{experiment_type}_job_file.pkl', 'rb') as f:
    job_list = pkl.load(f)

#counts up the number of completed jobs 
fail_count = 0 
success_count = 0 
unattempted_count = 0
unhandeled_count = 0

for job in job_list:
    status = job["status"]
    
    if status == "unattempted":
        unattempted_count +=1
    elif status == "failed":
        fail_count += 1
    elif status == "succeded":
        success_count += 1
    else:
        unhandeled_count += 1

total_count = len(job_list)

print(f"TOTAL JOBS: {total_count}")
print(f"COMPLETED JOBS: {success_count}")
print(f"FAILED JOBS: {fail_count}")
print(f"UNATTEMPTED JOBS: {unattempted_count}")
print(f"UNHANDLED ERRORS: {unhandeled_count}")
