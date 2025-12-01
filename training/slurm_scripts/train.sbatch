#!/bin/bash
#SBATCH --job-name=train_deblur
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=gpu4-8h
#SBATCH --signal=B:USR1@600
#SBATCH --cpus-per-task=24
#SBATCH --output=output/slurm-%j.out
#SBATCH --error=error/slurm-%j.err
#SBATCH --nodelist=lse-hpcnode[8]

#the signal time needs to be larger than the sleep in the handler function

# prepare your environment here
source ~/.bashrc
conda activate gencam
cd /datasets/sai/blur2vid/training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -z "$1" ]
 then
  #quit if no job number is passed
  echo "No config file passed, quitting"
  exit 1
 else
  config_file=$1
fi

handler()
{
  echo "function handler called at $(date)"
  # Send SIGUSR1 to the captured PID of the accelerate job
  if [ -n "$accelerate_pid" ]; then
    echo "Sending SIGUSR1 to accelerate PID: $accelerate_pid"
    python_id=$(ps --ppid $accelerate_pid -o pid=)
    kill -USR1 $python_id  # Send SIGUSR1 to the accelerate job
    sleep 300 # Wait for 5 minutes
  else
    echo "No accelerate PID found"
  fi
  echo "Resubmitting job with config file: $config_file"
  sbatch ${BASH_SOURCE[0]} $config_file
}

# register signal handler
trap handler SIGUSR1

echo "Starting job at $(date)"
#python train_controlnet.py #--config $config_file #& wait
accelerate launch --config_file accelerator_configs/accelerator_train_config.yaml --multi_gpu train_controlnet.py --config $config_file & 
accelerate_pid=$!

wait