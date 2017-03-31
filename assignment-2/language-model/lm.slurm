#!/bin/bash
#
#SBATCH --job-name=ak_experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p1080:1
#SBATCH --reservation=mhealth
#SBATCH --time=10:00:00
#SBATCH --mem=20000
#SBATCH --output=ak_experiment_%A.out
#SBATCH --error=ak_experiment_%A.err
#SBATCH --mail-user=ak6179@nyu.edu

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module purge
module load python3/intel/3.5.3
module load cuda/8.0.44

python3 -m pip install -U pip setuptools --user
python3 -m pip install cmake --user
python3 -m pip install numpy --upgrade --user
python3 -m pip install scipy --upgrade --user
python3 -m pip install https://download.pytorch.org/whl/cu80/torch-0.1.10.post2-cp35-cp35m-linux_x86_64.whl --upgrade --user
python3 -m pip install torchvision --upgrade --user

cd /scratch/ak6179/lm/language-model

python3 -u main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --encinit random --decinit random --weightinit zeros --data ../data/penn --nlayers 2 --clip 0.25 --save saved-models/model.11 > experiments/experiment.11.log
python3 -u main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied --encinit random --decinit random --weightinit zeros --data ../data/penn --nlayers 2 --clip 0.25 --save saved-models/model.12 > experiments/experiment.12.log
