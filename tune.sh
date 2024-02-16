#!/bin/bash
#SBATCH --time=0-2:00:00
#SBATCH --account=def-s2mclach
#SBATCH --mem=64000M            # memory per node
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8       # CPU cores/threads
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jh3chu@uwaterloo.ca
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Hello World"
nvidia-smi

# Load needed python and cuda modules
module load python/3.10 cuda cudnn

# Activate your enviroment
source ~/envs/lstm-transf/bin/activate
cd ~/workspace/syde770-project

# launch tensorboard
tensorboard --logdir="~/scratch/lstm-transformer/outputs" --host 0.0.0.0 --load_fast false &
# ssh -L localhost:6006:ng30203:6006 jh3chu@narval.computecanada.ca

# run script
python tune.py