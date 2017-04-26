#!/bin/bash
#
#SBATCH --job-name=ak_experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p1080:1
#SBATCH --time=10:00:00
#SBATCH --mem=40000
#SBATCH --output=ak_experiment_%A.out
#SBATCH --error=ak_experiment_%A.err
#SBATCH --mail-user=ak6179@nyu.edu

# Log what we're running and where
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module purge
module load python3/intel/3.5.3
module load cuda/8.0.44

python3 -m pip install -U pip setuptools --user
python3 -m pip install cmake --user
python3 -m pip install numpy --upgrade --user
python3 -m pip install scipy --upgrade --user
python3 -m pip install http://download.pytorch.org/whl/cu80/torch-0.1.11.post5-cp35-cp35m-linux_x86_64.whl --upgrade --user
python3 -m pip install dominate --user

cd something

python3 train.py --dataroot datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan --display_id 0 --checkpoints_dir checkpoints > experiments/experiments.1.log
