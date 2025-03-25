import sys
import torch

# XLA imports
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xlr

# Initialize XLA process group for torchrun
import torch_xla.distributed.xla_backend

def main():
    world_size = xlr.world_size()
    rank = xlr.global_ordinal()
    torch.distributed.init_process_group('xla')

    is_master = xm.is_master_ordinal()

    args = sys.argv

    print(f'Hello from {rank} of {world_size} {"MASTER" if is_master else ""}: argv={args}')


if __name__ == '__main__':
    main()


