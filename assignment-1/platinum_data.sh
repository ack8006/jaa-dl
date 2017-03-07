#!/bin/bash
#
#SBATCH --job-name=convnet_joe
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=20GB
#SBATCH --output=convnet_%A.out
#SBATCH --error=convnet_%A.err
#SBATCH --mail-user=ak6179@nyu.edu

module purge
module load python/intel/2.7.12
module load pytorch/intel/20170226
module load torchvision/0.1.7

cd /home/ak6179/deep-learning/jaa-dl/assignment-1

python -u platinum_data.py > platinum_data.log
