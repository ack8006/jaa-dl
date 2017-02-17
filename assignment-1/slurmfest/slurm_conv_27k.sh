#!/bin/bash
#
#SBATCH --job-name=conv_27k_batch4
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem=8GB 
#SBATCH --output=conv_27kb16d4_%A.out
#SBATCH --error=conv_27kb16d4_%A.err
#SBATCH --mail-user=act444@nyu.edu

module purge
module load python/intel/2.7.12
module load pytorch/intel/20170125

cd /scratch/act444/github/jaa-dl/assignment-1
python -u convnet3.py
