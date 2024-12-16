#!/bin/bash

#SBATCH --job-name='poker_bot'
#SBATCH --account=sdp
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=1-0:0
#SBATCH --output=./outputs/%x-%j.out
# time format: <days>-<hours>:<minutes>


# Path to container
container="/data/ai_club/containers/tensorflow-24.03-tf2-py3-cuda.sif"

# Command to run inside container
command="./train_bash.sh"

# Execute singularity container on node.
singularity exec --nv -B /data:/data ${container} ${command}