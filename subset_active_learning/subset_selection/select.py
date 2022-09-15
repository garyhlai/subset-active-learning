import sqlite3
from typing import Optional
from pydantic import BaseModel
from transformers import TrainingArguments
from .utils import seed_everything
import numpy as np
import json


class SubsetSelectorArguments(BaseModel):
    training_args: Optional[TrainingArguments] = None
    # TrainingArguments(
    #     model_card="google/electra-small-discriminator",
    #     pretraining=True,
    #     max_steps=6000,
    #     eval_steps=300,
    #     learning_rate=1e-5,
    #     batch_size=8,
    #     # adam should default to correct_bias = True
    #     adam_epsilon=1e-6,
    #     adam_beta1=0.9,
    #     adam_beta2=0.999,
    #     max_grad_norm=1.0,
    #     warmup_ratio=0.1,
    #     weight_decay=0.01,
    # )
    db_name: str = "sst_results"
    seed: int = 0
    annealing_runs: int = 5000
    total_sample_size: int = 1000
    n_search: int = 100
    anneal_factor: float = 0.1


class SubsetSelector:
    def __init__(self, params: SubsetSelectorArguments):
        for field, value in params:  # unpack the params
            setattr(self, field, value)
        self.current_num_run = 0

    def _get_nth_best_subset(self, n):
        """Select a single example -- the nth best by test accuracy to swap out"""
        try:
            con = sqlite3.connect("%s.db" % self.params.db_name)
            cur = con.cursor()
            cur.execute("SELECT * FROM states ORDER BY objective DESC LIMIT 1 OFFSET %d" % n)
            r = cur.fetchone()
        finally:
            con.close()
        return np.array(json.loads(r[0]))

    def _create_new_subset_in_place(self, base_subset: np.ndarray) -> None:
        """Create a new subset by swapping out one sample from the base_subset
        and swapping in a new sample. This is done inplace for efficiency
        """
        all_indices = np.arange(self.params.total_sample_size)
        available_examples = np.setdiff1d(all_indices, base_subset)
        in_sample = np.random.choice(available_examples)
        out_sample_idx = np.random.randint(0, len(base_subset) - 1)
        # print(f"swapping out: {base_subset[out_sample_idx]} at index {out_sample_idx} | swapping in: {in_sample}")
        base_subset[out_sample_idx] = in_sample

    def _insert_run(self, subset_indices: np.ndarray, quality: float) -> None:
        try:
            con = sqlite3.connect("%s.db" % self.param.db_name)
            cur = con.cursor()
            cur.execute("INSERT INTO states VALUES ('%s', %.8f)" % (json.dumps(subset_indices.tolist()), quality))
            con.commit()
        finally:
            con.close()

    def select_new_subset(self, current_num_runs: int) -> np.ndarray:
        exploration_ratio = (
            (self.params.annealing_runs - current_num_runs) / self.params.annealing_runs
            if current_num_runs < self.params.annealing_runs
            else 0
        )
        nth_best = np.random.randint(0, int(exploration_ratio * current_num_runs))
        nth_best_subset = self._get_nth_best_subset(nth_best)  # get the nth best subset (size 100)
        self._create_new_subset_in_place(nth_best_subset)
        return nth_best_subset

    def train_subset(self, subset_indices: np.ndarray) -> float:
        pass

    def one_run(self):
        seed_everything(self.params.seed)
        new_subset = self.select_new_subset(self.current_num_run)
        new_quality = self.train_subset()
        self._insert_run(subset_indices=new_subset, quality=new_quality)
        self.current_num_run += 1

    def select(self):
        while True:
            self.one_run(self)
