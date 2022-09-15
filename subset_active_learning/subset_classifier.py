from datasets import load_dataset
from transformers import AutoTokenizer
import json
import sqlite3
import pandas as pd
from collections import Counter
from pydantic.dataclasses import dataclass
from typing import List

import logging
from tabulate import tabulate

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class OptimalSubsetClassifierConfig:
    max_length: int = 66
    debug: bool = False
    model_name: str = "google/electra-small-discriminator"
    batch_size: int = 8
    max_steps: int = 20000


def get_df_from_db(db_path):
    with sqlite3.connect(db_path) as conn:
        # Now in order to read in pandas dataframe we need to know table name
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_name = cursor.fetchall()[0][0]
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    return df


def get_optimal_subset_data_indices(df):
    optimal_subset_idx = df["objective"].idxmax()

    optimal_subset_data_indices = set(json.loads(df.iloc[optimal_subset_idx].indexes))
    return optimal_subset_data_indices


def get_subset_unique_counts(df):
    # count the unique number of data points in each subset
    indexes = list(map(json.loads, df["indexes"].to_list()))
    unique_index_counts = list(map(lambda x: len(set(x)), indexes))
    subset_sizes = Counter(unique_index_counts)
    return subset_sizes


def preprocess(data, config, optimal_subset_data_indices):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def tokenize_func(examples, idx):
        tokenized = tokenizer(
            examples["sentence"], padding="max_length", max_length=config.max_length, truncation=True
        )
        tokenized["labels"] = 1 if idx in optimal_subset_data_indices else 0
        return tokenized

    ds = data.map(tokenize_func, remove_columns=data.column_names, batched=False, with_indices=True)

    ds.set_format(type="torch")

    return ds


def create_stratefied_split(
    positive_indices: List[str], negative_indices: List[str], split_points: float = (0.8, 0.9)
):
    """Split the dataset while keeping the overall class ratio (positive to negative) the same"""
    optimal_subset_data_indices_ls = list(positive_indices)
    non_optimal_subset_data_indices_ls = list(negative_indices)

    # get the positive examples
    split_indices = (
        round(len(optimal_subset_data_indices_ls) * split_points[0]),
        round(len(optimal_subset_data_indices_ls) * split_points[1]),
    )

    train_pos_indices = optimal_subset_data_indices_ls[: split_indices[0]]
    valid_pos_indices = optimal_subset_data_indices_ls[split_indices[0] : split_indices[1]]
    test_pos_indices = optimal_subset_data_indices_ls[split_indices[1] :]

    # get the negative examples
    split_indices = (
        round(len(non_optimal_subset_data_indices_ls) * split_points[0]),
        round(len(non_optimal_subset_data_indices_ls) * split_points[1]),
    )

    train_neg_indices = non_optimal_subset_data_indices_ls[: split_indices[0]]
    valid_neg_indices = non_optimal_subset_data_indices_ls[split_indices[0] : split_indices[1]]
    test_neg_indices = non_optimal_subset_data_indices_ls[split_indices[1] :]

    logger.info(
        tabulate(
            [
                ["train", len(train_pos_indices), len(train_neg_indices)],
                ["valid", len(valid_pos_indices), len(valid_neg_indices)],
                ["test", len(test_pos_indices), len(test_neg_indices)],
            ],
            headers=["dataset", "num positive examples", "num negative examples"],
        )
    )
    # combine
    train_indices = train_pos_indices + train_neg_indices
    valid_indices = valid_pos_indices + valid_neg_indices
    test_indices = test_pos_indices + test_neg_indices
    return train_indices, valid_indices, test_indices


def create_train_valid_test_debug_ds(optimal_subset_data_indices, config):
    # get the 1000 point dataset
    sst2 = load_dataset("sst")
    data_pool = sst2["train"].shuffle(seed=0).select(range(1000))
    ds = preprocess(data_pool, config, optimal_subset_data_indices)

    # split the dataset into train, valid, test (keeping proportion of positive to negative examples the same)
    data_pool_indices = set(range(1000))
    non_optimal_subset_data_indices = list(data_pool_indices - optimal_subset_data_indices)
    train_indices, valid_indices, test_indices = create_stratefied_split(
        list(optimal_subset_data_indices), list(non_optimal_subset_data_indices), split_points=(0.8, 0.9)
    )

    train_ds = ds.select(train_indices).shuffle(seed=0)
    valid_ds = ds.select(valid_indices).shuffle(seed=0)
    test_ds = ds.select(test_indices).shuffle(seed=0)
    debug_ds = train_ds.select(range(12))
    return train_ds, valid_ds, test_ds, debug_ds
