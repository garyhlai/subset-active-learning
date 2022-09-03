from datasets import load_dataset, load_metric
from pydantic.dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
from typing import Literal, Optional
from tqdm import tqdm
import wandb

from src.settings.subset_selection import SUBSET_SELECTION_POOL


class ActiveLearner:
    def __init__(self, config, training_args):
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.training_args = training_args

        set_seed(42)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.metric = load_metric("accuracy")

        self.sst2 = load_dataset("sst")
        self.original_train_ds = self.preprocess(self.sst2["train"]).shuffle(
            seed=0
        )  # need to preprocess/tokenize all of train_ds for uncertainty sampling
        self.valid_ds = self.preprocess(self.sst2["validation"])
        self.test_ds = self.preprocess(self.sst2["test"])
        self.train_data_indices = []

        # initialize model used for uncertainty sampling
        if self.config.strategy == "uncertainty_sampling":
            self.sampling_model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
        elif self.config.strategy == "subset_sampling":
            self.sampling_model = AutoModelForSequenceClassification.from_pretrained(
                config.subset_model_path, num_labels=2
            )

    def preprocess(self, data):
        data = data.rename_column("label", "scalar_label")
        data = data.map(lambda x: {"label": 0 if x["scalar_label"] < 0.5 else 1})

        def tokenize_func(examples):
            tokenized = self.tokenizer(
                examples["sentence"], padding="max_length", max_length=self.config.max_length, truncation=True
            )
            tokenized["labels"] = examples["label"]
            return tokenized

        ds = data.map(
            tokenize_func,
            remove_columns=data.column_names,
            batched=True,
        )
        ds.set_format(type="torch")
        return ds

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    @staticmethod
    def calculate_entropy(prob_dist):
        """
        Returns raw entropy

        Keyword arguments:
            prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0. e.g. tensor([0.0321, 0.6439, 0.0871, 0.2369])
        """
        log_probs = prob_dist * torch.log2(prob_dist)  # multiply each probability by its base 2 log
        raw_entropy = 0 - torch.sum(log_probs, axis=-1)
        return raw_entropy

    def get_preds(self):
        remaining_pool_indices = set(range(len(self.original_train_ds))) - set(self.train_data_indices)
        dl = DataLoader(
            self.original_train_ds.select(remaining_pool_indices), batch_size=self.config.batch_size, shuffle=False
        )
        self.sampling_model.eval()
        self.sampling_model.to(self.device)
        with torch.no_grad():
            preds = []
            for batch in tqdm(dl):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.sampling_model(**batch)
                preds.append(out.logits)
            preds = torch.cat(preds)
            preds = torch.nn.functional.softmax(preds, dim=-1)
        return preds

    def uncertainty_sampling(self, n_samples):
        preds = self.get_preds()
        entropys = self.calculate_entropy(preds)
        return torch.topk(entropys, k=n_samples).indices.tolist()

    def subset_sampling(self, n_samples, nth_step):
        if nth_step == 0:
            return SUBSET_SELECTION_POOL  # first step is trained on the subset selection data pool
        preds = self.get_preds()
        pos_scores = preds[:, 1]
        return torch.topk(pos_scores, k=n_samples).indices.tolist()

    def random_sampling(self, sampling_size):
        if not hasattr(self, "all_random_samples"):
            max_num_samples = self.config.sampling_sizes[-1]
            self.all_random_samples = np.random.choice(len(self.sst2["train"]), replace=False, size=max_num_samples)
        return self.all_random_samples[:sampling_size]

    def sample_data(self, n_new_samples, sampling_size, nth_step):
        if self.config.strategy == "random_sampling":
            selected_indices = self.random_sampling(sampling_size)
            return selected_indices
        elif self.config.strategy == "uncertainty_sampling":
            newly_selected_indices = self.uncertainty_sampling(n_new_samples)
            self.train_data_indices.extend(newly_selected_indices)
            return self.train_data_indices
        elif self.config.strategy == "subset_sampling":
            newly_selected_indices = self.subset_sampling(n_new_samples, nth_step)
            self.train_data_indices.extend(newly_selected_indices)
            return self.train_data_indices
        else:
            raise ValueError(f"Unknown strategy {self.config.strategy}")

    def step(self, n_new_samples, sampling_size, nth_step):
        """Take an active learning step"""
        ########### set up data #########
        # sample new data
        sampled_data = self.sample_data(n_new_samples, sampling_size, nth_step)
        # concatenate the sampled data with the original data
        train_data = self.sst2["train"].select(sampled_data)
        debug_data = self.sst2["train"].select(sampled_data[:8])

        self.train_ds = self.preprocess(train_data)
        self.debug_ds = self.preprocess(debug_data)

        print(
            f"Current train size: {self.train_ds} |  Sampling {n_new_samples} new samples | Config strategy is {self.config.strategy}"
        )

        if len(self.train_ds) != sampling_size:
            raise ValueError(
                f"wrong number of samples was selected. train_ds has length {len(self.train_ds)}, but sampling_size is {sampling_size}"
            )

        ########### set up training #########
        dir = f"./{self.config.strategy}/size_{sampling_size}" if not self.config.debug else "./debug"
        print(f"training_args: {self.training_args}")
        model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name, num_labels=2)
        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=self.train_ds if not self.config.debug else self.debug_ds,
            eval_dataset=self.valid_ds if not self.config.debug else self.debug_ds,
            compute_metrics=self.compute_metrics,
        )
        ######### train #######
        trainer.train()

        ######## tear down ########
        wandb.finish()

        if self.config.strategy == "uncertainty_sampling":
            self.sampling_model = model  # we use model from the previous active learning step as the uncertainty sampling model for the new step.

        ######## test ########
        if not self.config.debug:
            outputs = trainer.predict(self.test_ds)
            with open(f"{dir}/test_set_evaluation_{sampling_size}.pkl", "wb") as f:
                pickle.dump(outputs, f)

        ######## save train_ds ########
        with open(
            f"{dir}/train_ds_{self.config.strategy}_{sampling_size}.pkl",
            "wb",
        ) as f:
            pickle.dump(self.train_ds, f)

    def train(self):
        for i, sampling_size in enumerate(self.config.sampling_sizes):
            n_new_samples = (
                sampling_size if i == 0 else self.config.sampling_sizes[i] - self.config.sampling_sizes[i - 1]
            )
            self.step(n_new_samples=n_new_samples, sampling_size=sampling_size, nth_step=i)


@dataclass(frozen=True)
class ActiveLearnerConfig:
    subset_model_path: Optional[str]
    max_length: int = 66
    debug: bool = False
    model_name: str = "google/electra-small-discriminator"
    strategy: Literal["random_sampling", "uncertainty_sampling", "subset_sampling"] = "random_sampling"
    sampling_sizes: tuple = (1000, 2000, 3000, 4000)
    max_steps: int = 20000
    batch_size: int = 8
