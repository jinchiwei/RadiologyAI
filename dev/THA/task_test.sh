#!/bin/sh

#SBATCH
#SBATCH --job-name=uni_bi_RAIL
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=end
#SBATCH --mail-user=jshin49@jhu.edu

module load cuda cudnn/5.1 gcc slurm feh anaconda-python/3.6
module unload parallel_studio_xe_2015/15.0.2

source activate jiwonenv
python test.py -n inception_v3
source deactivate
