#!/bin/bash
#
#SBATCH --job-name=
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time=24:00:00 
#SBATCH --mem=8GB 
#SBATCH --output=%A.out
#SBATCH --error=%Aa.err
#SBATCH --mail-user=ak6179@nyu.edu

module purge
module load python/intel/2.7.12
module load tensorflow/python2.7/20161220

cd /home/ak6179/jaa-dl/assignment-1/ladder/ladder-tensorflow/ladder
python -u ladder.py > ladder_tf_3000.log

