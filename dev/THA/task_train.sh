#!/bin/sh

#SBATCH
#SBATCH --job-name=mxj_WH
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#sbatch --mail-type=end
#sbatch --mail-user=jwei9@jhu.edu

module load cuda cudnn/7.2.1 gcc python/anaconda-3.6
module unload parallel_studio_xe_2015/15.0.2

source activate deepdream
python train.py
source deactivate
