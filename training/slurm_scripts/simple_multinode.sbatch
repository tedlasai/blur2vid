#!/bin/bash
#SBATCH --job-name=XYZ
#SBATCH --nodes=4
#SBATCH --mem=256gb
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output=output/slurm-%j-%N.out
#SBATCH --error=error/slurm-%j-%N.err
#SBATCH --qos=scavenger
#SBATCH --signal=B:USR1@300
#SBATCH --nodelist=lse-hpcnode[1,3,4,5,10-12]

#6 and 9 are messed up
#7 is sketchy as well

set -x -e

if [ -z "$1" ]
 then
  #quit if no job number is passed
  echo "No config file passed, quitting"
  exit 1
 else
  config_file=$1
fi

source ~/.bashrc
conda activate gencam
cd /datasets/sai/blur2vid/training

echo "START TIME: $(date)"

# needed until we fix IB issues
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens

# Training setup
GPUS_PER_NODE=4
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


#CMD="accelerate_test.py"
CMD="train_controlnet.py --config $config_file"
LAUNCHER="accelerate launch \
  --multi_gpu \
  --gpu_ids 0,1,2,3 \
  --num_processes $WORLD_SIZE \
  --num_machines $NNODES \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --rdzv_backend=c10d \
  --max_restarts 0 \
  --tee 3 \
"

# # NOT SURE THE FOLLOWING ENV VARS IS STRICTLY NEEDED (PROBABLY NOT)
# export CUDA_HOME=/usr/local/cuda-11.6
# export LD_PRELOAD=$CUDA_HOME/lib/libnccl.so
# export LD_LIBRARY_PATH=$CUDA_HOME/efa/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

handler()
{
  echo "Signal handler triggered at $(date)"

  sleep 120  # Let training save
  sbatch ${BASH_SOURCE[0]} $config_file
}

# register signal handler
trap handler SIGUSR1

clear; srun --cpu-bind=none --jobid $SLURM_JOB_ID $LAUNCHER $CMD & srun_pid=$!

wait

echo "END TIME: $(date)"
