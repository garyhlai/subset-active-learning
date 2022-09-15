from subset_active_learning.subset_classifier import (
    create_stratefied_split,
    get_df_from_db,
    get_optimal_subset_data_indices,
    OptimalSubsetClassifierConfig,
    preprocess,
)
from datasets import load_dataset
import pytest


@pytest.fixture
def optimal_subset_data_indices():
    df = get_df_from_db("./subset_selection/sst_results.db")
    optimal_subset_data_indices = get_optimal_subset_data_indices(df)
    return optimal_subset_data_indices


def test_preprocess(optimal_subset_data_indices):
    # Set up
    sst2 = load_dataset("sst")
    data_pool = sst2["train"].shuffle(seed=0).select(range(1000))
    config = OptimalSubsetClassifierConfig(
        max_length=66, debug=False, model_name="google/electra-small-discriminator", batch_size=8
    )
    # Run
    ds = preprocess(data_pool, config, optimal_subset_data_indices)
    # test the labels are correct
    for idx, label in enumerate(ds["labels"]):
        if idx in optimal_subset_data_indices:
            assert int(label) == 1
        else:
            assert int(label) == 0


def test_create_stratefied_split(optimal_subset_data_indices):
    data_pool_indices = list(range(1000))
    non_optimal_subset_data_indices = list(set(data_pool_indices) - optimal_subset_data_indices)
    optimal_subset_data_indices = list(optimal_subset_data_indices)
    train_indices, valid_indices, test_indices = create_stratefied_split(
        optimal_subset_data_indices, non_optimal_subset_data_indices, split_points=(0.8, 0.9)
    )
    # check there are no overlaps between the three datasets
    assert len(set(train_indices + valid_indices + test_indices)) == len(train_indices + valid_indices + test_indices)
