#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --account=def-s2mclach
#SBATCH --mem=32000M            # memory per node
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20       # CPU cores/threads
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jh3chu@uwaterloo.ca
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Hello World"
nvidia-smi

# Load needed python and cuda modules
module load python/3.10 cuda cudnn

# Activate your enviroment
source ~/envs/syde770/bin/activate

cd ~/workspace/syde770-project

python tune_CNNLSTM.py