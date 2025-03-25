#!/bin/bash
#SBATCH --job-name=xla_hello
#SBATCH --output=xla_hello.log
#SBATCH --partition=trn1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH -t 0-0:30

source /opt/aws_neuron_venv_pytorch/bin/activate

export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export BUCKET_CAP_MB=512
export XLA_TRANSFER_SEED_ASYNC=1

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr}
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4) )
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} WORLD_SIZE=${WORLD_SIZE}"

srun python3 xla_hello.py

echo "Done"

