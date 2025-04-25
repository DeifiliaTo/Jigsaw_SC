#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --account=''
#SBATCH --output='four_1tflop.out'
#SBATCH --job-name='4_1tf'
#SBATCH --constraint='LSDF'
#SBATCH --exclusive

module purge
module restore pangu

source $SRC/../.venv/bin/activate
which python
# Change 5-digit MASTER_PORT as you wish, SLURM will raise Error if duplicated with others.
export MASTER_PORT=12340

# Get the first node name as master address.
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 # https://discuss.pytorch.org/t/torch-distributed-all-reduce-causes-memory-trashing/215024
export TORCH_NCCL_HIGH_PRIORITY=1


srun python -u $SRC/scripts/fourway.py --config_file config.yaml
