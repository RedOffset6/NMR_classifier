#!/bin/bash

#SBATCH --job-name=BEN_IMAGE_GENERATION
#SBATCH --account=CHEM014742
#SBATCH --partition=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8GB
#SBATCH --time=06:00:00
echo 'Executing an image job'

#hostname

module load languages/miniconda

echo "before activation: $(which python)"

source activate image_classification_env

echo "after activation $(which python)"

python3 job_executer.py COMBINED





