#!/bin/bash
#SBATCH --job-name=simulated-annealing
#SBATCH --gres=gpu:1
#SBATCH --time=0-72:00
#SBATCH --ntasks=1
#SBATCH --array=1-30
#SBATCH --qos=general
#SBATCH --requeue

source activate torch

papermill -p db_path ../results/sst.db \
    -p ds_name sst \
    -p wandb_project sst_search \
    -p wandb_entity johntzwei \
    -p pool_size 1000 \
    -p search_size 100 \
    ./search_subset.ipynb /dev/null &

papermill -p db_path ../results/sst.db \
    -p ds_name sst \
    -p wandb_project sst_search \
    -p wandb_entity johntzwei \
    -p pool_size 1000 \
    -p search_size 100 \
    ./search_subset.ipynb /dev/null &

wait
