#!/bin/bash
#
#SBATCH --job-name=
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time=6:00 
#SBATCH --mem=4GB 
#SBATCH --output=%A.out
#SBATCH --error=%Aa.err

module purge
module load python/intel/2.7.12
module load pytorch/intel/20170125

cd /scratch/$USER/github/jaa-dl/assignment-1
python -u convnet3.py
