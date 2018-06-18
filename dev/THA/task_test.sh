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

module load cuda cudnn/5.1 gcc slurm feh anaconda-python/3.6
module unload parallel_studio_xe_2015/15.0.2

source activate deeplearning
python test.py -n inception_v3
source deactivate
