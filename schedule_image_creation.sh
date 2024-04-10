#!/bin/bash

#SBATCH --job-name=BEN_IMAGE_GENERATION
#SBATCH --account=CHEM014742
#SBATCH --partition=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --time=6:00:00
#SBATCH --array=1-9
echo 'GENERATING IMAGES'

hostname
python3 image_creation.py ${SLURM_ARRAY_TASK_ID}





