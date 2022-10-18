from subset_active_learning.subset_selection.preprocess import preprocess_sst2
from subset_active_learning.subset_selection import select, preprocess, data_loader
import wandb
import datasets
import numpy as np
from typing import List
import time
from transformers import set_seed


def random_select_subset(seed: int, select_size: int, from_size: int):
    np.random.seed(seed)
    return np.random.choice(from_size, replace=False, size=select_size)


class CorrelationRun:
    def __init__(
        self,
        proxy_model_name: str,
        target_model_name: str,
        proxy_training_config: select.SubsetTrainingArguments,
        target_training_config: select.SubsetTrainingArguments,
        wandb_tags: List[int],
    ):
        self.proxy_model_name = proxy_model_name
        self.target_model_name = target_model_name
        self.proxy_training_config = proxy_training_config
        self.target_training_config = target_training_config
        self.proxy_ds = preprocess_sst2(proxy_model_name)
        self.target_ds = preprocess_sst2(target_model_name)
        self.wandb_tags = wandb_tags

    def one_comparison_run(
        self, data_seed: int, model_seed: int, subset_size: int = 500
    ):
        """
        While we want to vary the data seed to get different random subsets, 
        model_seed should be generally fixed to reduce noises during a correlation run.
        """
        self.data_seed = data_seed
        self.subset_size = subset_size
        # both models should share the same train_indices
        train_indices = random_select_subset(
            seed=data_seed,
            select_size=subset_size,
            from_size=len(self.proxy_ds["train"]),
        )

        self._one_model_run(
            model_name=self.target_model_name,
            model_seed=model_seed,
            train_ds=self.target_ds["train"].select(train_indices),
            valid_ds=self.target_ds["validation"],
            test_ds=self.target_ds["test"],
            config=self.target_training_config
        )
        self._one_model_run(
            model_name=self.proxy_model_name,
            model_seed=model_seed,
            train_ds=self.proxy_ds["train"].select(train_indices),
            valid_ds=self.proxy_ds["validation"],
            test_ds=self.proxy_ds["test"],
            config=self.proxy_training_config
        )

    def _one_model_run(
        self,
        model_name: str,
        model_seed: int, 
        train_ds: datasets.Dataset,
        valid_ds: datasets.Dataset,
        test_ds: datasets.Dataset,
        config: select.SubsetTrainingArguments
    ):
        
        wandb_run = wandb.init(
            project="subset-search-correlation", entity="johnny-gary", tags=self.wandb_tags+[str(self.data_seed), model_name, f"train_size-{self.subset_size}"]
        )
        wandb.log({"batch_size": config.batch_size})
        wandb.log({"model_seed": model_seed})
        set_seed(model_seed)
        print(f"##### Warning: Hard Setting Model Seed to {model_seed}")
        subset_trainer = select.SubsetTrainer(
            params=config,
            valid_ds=valid_ds,
            test_ds=test_ds,
        )
        start_time = time.time()
        subset_trainer.train(subset=train_ds, early_stopping=config.early_stopping, calculate_test_accuracy=True)
        wandb.log({"run_time": round(time.time() - start_time, 2)})
        wandb_run.finish()
