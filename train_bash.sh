#!/bin/bash

. /opt/conda/etc/profile.d/conda.sh
conda activate apollo

export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH="$CUDNN_PATH/lib":"/usr/local/cuda-12.4/compat/lib.real"
export PATH="$PATH":"/usr/local/cuda-12.4/compat/lib.real"
export TF_CPP_MIN_LOG_LEVEL=2

python train.py