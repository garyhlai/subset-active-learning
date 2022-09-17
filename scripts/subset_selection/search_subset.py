import sqlite3
from typing import Any, Optional
from pydantic import BaseModel, Extra, Field
from transformers import TrainingArguments, AutoModel, AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
import numpy as np
import json
import datasets
import wandb
import torch
from tqdm import tqdm
from subset_active_learning.subset_selection import select, preprocess



DB_PATH = "/home/glai/dev/subset-active-learning/local_bucket/new_sst.db"

training_args = select.SubsetTrainingArguments(eval_steps=3, max_steps=5)
searching_args = select.SubsetSearcherArguments(seed=0, db_path=DB_PATH)

processed_ds = preprocess.preprocess_sst2(training_args.model_card)

subset_trainer = select.SubsetTrainer(
    params=training_args, valid_ds=processed_ds["validation"], test_ds=processed_ds["test"]
)

data_pool = processed_ds["train"].shuffle(seed=searching_args.seed).select(range(searching_args.data_pool_size))
subset_searcher = select.SubsetSearcher(subset_trainer=subset_trainer, params=searching_args, data_pool=data_pool)

subset_searcher.search(n_runs=3)






