#!/usr/bin/env bash

export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export BUCKET_CAP_MB=512
export XLA_TRANSFER_SEED_ASYNC=1

export MASTER_ADDR=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4) )

if [ $SLURM_PROCID -eq 0 ] ; then
    echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
fi
DISTRIBUTED_ARGS="--nproc_per_node ${SLURM_NTASKS_PER_NODE} --nnodes ${SLURM_NNODES} --node_rank ${SLURM_NODEID}  --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"

echo "$(hostname): Rank ${SLURM_PROCID} of ${SLURM_NTASKS} will run with torchrun args ${DISTRIBUTED_ARGS}"

torchrun $DISTRIBUTED_ARGS xla_hello.py

