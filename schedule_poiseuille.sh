#!/bin/bash

#SBATCH --job-name=Ben_poiseuille
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=900M

echo 'Computing the poiseuille distribution'

hostname
./poiseuille 400 50 0.6  0.01 10000 10000
