#!/bin/sh
#SBATCH --job-name=mxj_inPR
#SBATCH --time=2:00:00
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
##sbatch --mail-type=end 
##sbatch --mail-user=jwei9@jhu.edu

module load cuda cudnn/5.1 gcc slurm feh anaconda-python/3.6
module unload parallel_studio_xe_2015/15.0.2

# unalias .
type activate
source activate deepdream
which python
echo $PATH
echo python path: $PYTHONPATH
conda info --envs
# python train.py
python just_import_torch.py
source deactivate
