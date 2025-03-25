#!/bin/bash
#SBATCH --job-name=xla_hello
#SBATCH --output=xla_hello.log
#SBATCH --partition=trn1
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH -t 0-0:30

source /opt/aws_neuron_venv_pytorch/bin/activate


srun ./xla_hello.sh

echo "Done"

