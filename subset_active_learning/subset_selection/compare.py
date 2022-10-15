import time
from typing import List

from subset_active_learning.subset_selection import select, preprocess, data_loader
import wandb
import numpy as np
import datasets

from transformers import AutoModelForSequenceClassification, get_scheduler, set_seed
from tqdm import tqdm
import torch


class CustomSubsetTrainer(select.SubsetTrainer):
    def __init__(
        self, params: select.SubsetTrainingArguments, valid_ds, test_ds, num_workers=0
    ) -> None:
        self.num_workers = num_workers
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = datasets.load_metric(self.params.metric)
        self.val_dataloader = data_loader.DataLoader(
            dataset=valid_ds, shuffle=False, batch_size=self.params.batch_size
        )
        self.test_dataloader = data_loader.DataLoader(
            dataset=test_ds, shuffle=False, batch_size=self.params.batch_size
        )

    def _train(self, model, train_dataset, tolerance=2):
        steps = 0
        epochs = 0
        best_acc = None
        patience = 0
        pbar = tqdm(total=self.params.max_steps)
        train_dataloader = data_loader.DataLoader(
            train_dataset, shuffle=True, batch_size=self.params.batch_size
        )
        it = iter(train_dataloader)

        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=self.params.learning_rate,
            betas=(self.params.adam_beta1, self.params.adam_beta2),
            eps=self.params.adam_epsilon,
            weight_decay=self.params.weight_decay,
        )
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.params.warmup_ratio * self.params.max_steps,
            num_training_steps=self.params.max_steps,
        )

        while steps < self.params.max_steps:
            # training
            model.train()
            total_loss = 0.0
            try:
                batch = next(it)
            except Exception:
                epochs += 1
                it = iter(train_dataloader)
                batch = next(it)
            steps += 1

            loss = model(**batch).loss
            total_loss += float(loss)
            wandb.log({"loss": loss})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.params.max_grad_norm
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            pbar.set_description(
                "Epoch: %d, Avg batch loss: %.2f" % (epochs, total_loss / steps)
            )
            pbar.update(1)

            if steps % self.params.eval_steps == 0:
                model.eval()
                eval_dict = self._evaluate(model, self.val_dataloader)
                wandb.log({"sst:val_acc": eval_dict["accuracy"]})
                # early stopping
                if not best_acc or eval_dict["accuracy"] > best_acc:
                    best_acc = eval_dict["accuracy"]
                else:
                    patience += 1
                if patience >= tolerance:
                    break

    def _evaluate(self, model, val_dataloader):
        model.eval()
        val_pbar = tqdm(total=len(val_dataloader))
        for batch in val_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            self.metric.add_batch(predictions=predictions, references=batch["labels"])
            val_pbar.update(1)
        eval_dict = self.metric.compute()
        val_pbar.set_description('Acc: %.2f' % eval_dict['accuracy'])
        return eval_dict


class ComparisonRun:
    def __init__(
        self,
        train_ds: datasets.Dataset,
        valid_ds: datasets.Dataset,
        test_ds: datasets.Dataset,
        data_seed: int,
        model_seed: int, 
        save_path: str = None
    ):
        self.train_ds, self.valid_ds, self.test_ds = train_ds, valid_ds, test_ds
        self.data_seed = data_seed
        self.model_seed = model_seed
        self.save_path = save_path

    def one_run(
        self,
        wandb_tags: List[str],
        config: select.SubsetTrainingArguments,
        num_workers: int = 0,
        use_custom_subset_trainer: bool = False,
    ):
        wandb_tags.append(str(self.data_seed))
        wandb_run = wandb.init(
            project="subset-search-gpu-opt", entity="johnny-gary", tags=wandb_tags
        )
        wandb.log({"batch_size": config.batch_size})
        set_seed(self.model_seed)
        print(f"##### Warning: Hard Setting Model Seed to {self.model_seed}")
        subset_trainer = (
            CustomSubsetTrainer(
                params=config,
                valid_ds=self.valid_ds,
                test_ds=self.test_ds,
                num_workers=num_workers,
            )
            if use_custom_subset_trainer
            else select.SubsetTrainer(
                params=config,
                valid_ds=self.valid_ds,
                test_ds=self.test_ds,
                num_workers=num_workers,
            )
        )
        start_time = time.time()
        subset_trainer.train(
            subset=self.train_ds, calculate_test_accuracy=True
        )
        wandb.log({"run_time": round(time.time() - start_time, 2)})
        wandb_run.finish()

    def run_batch_comparison(
        self,
        small_batch_config: select.SubsetTrainingArguments,
        large_batch_config: select.SubsetTrainingArguments,
    ):
        """
        - train small batch size until early stopping
        - train large batch size until early stopping
        """
        self.one_run(
            wandb_tag=[f"small_batch_{small_batch_config.batch_size}"],
            config=small_batch_config,
        )
        self.one_run(
            wandb_tag=[f"large_batch_{large_batch_config.batch_size}"],
            config=large_batch_config,
        )
