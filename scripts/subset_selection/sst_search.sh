#!/bin/bash
#SBATCH --job-name=simulated-annealing
#SBATCH --gres=gpu:1
#SBATCH --time=0-72:00
#SBATCH --ntasks=1
#SBATCH --array=1-30
#SBATCH --qos=general
#SBATCH --requeue

source activate torch

papermill -p db_path ./mnli_9-29.db \
    -p ds_name mnli \
    -p num_labels 3 \
    -p valid_split validation \
    -p test_split validation \
    -p wandb_project mnli_search \
    -p wandb_entity johntzwei \
    -p warmup_runs 100 \
    -p annealing_runs 2000 \
    -p pool_size 2000 \
    -p search_size 200 \
    -p max_steps 3000 \
    ./search_subset.ipynb /dev/null &

papermill -p db_path ./mnli_9-29.db \
    -p ds_name mnli \
    -p num_labels 3 \
    -p wandb_project mnli_search \
    -p wandb_entity johntzwei \
    -p warmup_runs 100 \
    -p annealing_runs 2000 \
    -p pool_size 2000 \
    -p search_size 200 \
    -p max_steps 3000 \
    ./search_subset.ipynb /dev/null &

wait
