import sys
import os
import socket

# PyTorch
import torch
import torch.distributed as dist

# XLA (Accelerated Linear Algebra)
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Distributed XLA
import torch_xla.distributed.xla_backend

def main():
    hostname = socket.gethostname()
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group('xla', rank=rank, world_size=world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    xla_world_size = xlr.world_size()
    xla_rank = xlr.global_ordinal()
    xla_is_master = xm.is_master_ordinal()

    print(f'{hostname}: Hello from {rank} of {world_size}, XLA rank {xla_rank} of {xla_world_size} {"MASTER" if xla_is_master else ""}')

    dist.barrier()

if __name__ == '__main__':
    main()


