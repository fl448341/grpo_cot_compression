#!/bin/bash
#SBATCH --job-name=phase1
#SBATCH --partition=common
#SBATCH --qos=fl448341_common
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=phase1/results/slurm-%j.out


# mimuw entropy 
set -e

export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source $HOME/venvs/rl/bin/activate

cd $SLURM_SUBMIT_DIR

echo "Node: $(hostname) | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m phase1.run_phase1 --n-eval 200 --out phase1/results --verbose
