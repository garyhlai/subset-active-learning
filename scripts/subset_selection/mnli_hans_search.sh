#!/bin/bash
#SBATCH --job-name=simulated-annealing
#SBATCH --gres=gpu:1
#SBATCH --time=0-72:00
#SBATCH --ntasks=1
#SBATCH --array=1-30
#SBATCH --qos=general
#SBATCH --requeue

source activate torch

papermill -p db_path ../../results/mnli_hans.db \
    -p ds_name mnli \
    -p test_dataset hans \
    -p eval_mapping \[0,1,1\] \
    -p num_labels 3 \
    -p valid_split validation_mismatched \
    -p test_split train \
    -p wandb_project mnli_hans_search \
    -p wandb_entity johntzwei \
    -p pool_size 2000 \
    -p search_size 200 \
    ./search_subset.ipynb /dev/null &

papermill -p db_path ../../results/mnli_hans.db \
    -p ds_name mnli \
    -p test_dataset hans \
    -p eval_mapping \[0,1,1\] \
    -p num_labels 3 \
    -p valid_split validation_mismatched \
    -p test_split train \
    -p wandb_project mnli_hans_search \
    -p wandb_entity johntzwei \
    -p pool_size 2000 \
    -p search_size 200 \
    ./search_subset.ipynb /dev/null &

wait
