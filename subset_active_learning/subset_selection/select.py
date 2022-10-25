import sqlite3
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, get_scheduler
from .utils import seed_everything
import numpy as np
import json
import datasets
import wandb
import torch
from tqdm.notebook import tqdm
import os


class SubsetTrainingArguments(BaseModel):
    model_card: str = "google/electra-small-discriminator"
    pretraining: bool = True
    max_steps: int = 6000
    eval_steps: int = 300
    learning_rate: float = 1e-5
    batch_size: int = 8
    # adam should default to correct_bias = True
    adam_epsilon: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    metric: str = "accuracy"
    # dataset arguments
    eval_mapping: list = []
    num_labels: int = 2

   
    
class SubsetTrainer(): 
    def __init__(self, params: SubsetTrainingArguments, valid_ds, test_ds) -> None: 
        self.params = params
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.metric = datasets.load_metric(self.params.metric)
        self.val_dataloader = torch.utils.data.DataLoader(valid_ds, shuffle=False, batch_size=self.params.batch_size, pin_memory=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=self.params.batch_size, pin_memory=True)

    def train_one_step(self, subset: datasets.Dataset) -> float:
        model = AutoModelForSequenceClassification.from_pretrained(self.params.model_card, num_labels=self.params.num_labels)
        model.to(self.device)
        self._train(model, subset)
        eval_dict = self._evaluate(model, self.test_dataloader, self.params.eval_mapping)
        eval_dict = {"sst2_test:%s" % k: v for k, v in eval_dict.items()}
        new_quality = eval_dict["sst2_test:accuracy"]
        wandb.log(eval_dict)
        return new_quality

    def _train(self, model, train_dataset, tolerance=1):
        steps = 0
        epochs = 0
        best_acc = None
        patience = 0
        pbar = tqdm(total=self.params.max_steps)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=self.params.batch_size, pin_memory=True)
        it = iter(train_dataloader)

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.params.learning_rate, betas=(self.params.adam_beta1, self.params.adam_beta2), eps=self.params.adam_epsilon, weight_decay=self.params.weight_decay)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=self.params.warmup_ratio*self.params.max_steps, num_training_steps=self.params.max_steps)

        while steps < self.params.max_steps:
            # training
            model.train()
            total_loss = 0.
            try:
                batch = next(it)
            except:
                epochs += 1
                it = iter(train_dataloader)
                batch = next(it)
            steps += 1
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.cpu()
            wandb.log({'loss' : loss})
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.params.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            pbar.set_description('Epoch: %d, Avg batch loss: %.2f' % (epochs, total_loss / steps))
            pbar.update(1)

            if steps % self.params.eval_steps == 0:
                model.eval()
                eval_dict = self._evaluate(model, self.val_dataloader, eval_mapping={})
                wandb.log({'sst:val_acc' : eval_dict['accuracy']})
                # early stopping
                if not best_acc or eval_dict['accuracy'] > best_acc:
                    best_acc = eval_dict['accuracy']
                else:
                    patience += 1
                if patience >= tolerance:
                    break

    def _evaluate(self, model, val_dataloader, eval_mapping: list):
        model.eval()
        val_pbar = tqdm(total=len(val_dataloader))
        for batch in val_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().tolist()

            if len(eval_mapping) > 0:
                predictions = list(map(lambda x: eval_mapping[x], predictions))

            self.metric.add_batch(predictions=predictions, references=batch["labels"])
            val_pbar.update(1)
        eval_dict = self.metric.compute()
        val_pbar.set_description('Acc: %.2f' % eval_dict['accuracy'])
        return eval_dict


class SubsetSearcherArguments(BaseModel): 
    db_path: str = "sst_results"
    wandb_project: str = "subset_search"
    wandb_entity: str = "johnny-gary"
    seed: int = 0
    warmup_runs: int = 100
    annealing_runs: int = 2000
    offset_idx: int = 0
    data_pool_size: int = 1000
    optimal_subset_size: int = 100

class SubsetSearcher:
    def __init__(
        self,
        data_pool: datasets.Dataset,
        subset_trainer: SubsetTrainer,
        params: SubsetSearcherArguments
    ):
        self.params = params
        self.seed, self.db_path, self.data_pool_size, self.warmup_runs, self.annealing_runs, \
                self.optimal_subset_size, self.wandb_project, self.wandb_entity, self.offset_idx = (
            params.seed,
            params.db_path,
            params.data_pool_size,
            params.warmup_runs,
            params.annealing_runs,
            params.optimal_subset_size,
            params.wandb_project,
            params.wandb_entity,
            params.offset_idx
        )
        self.data_pool = data_pool
        self.subset_trainer = subset_trainer
        if not os.path.exists(self.db_path): 
            self.initialize_db(self.db_path)
    
    def initialize_db(self, db_path: str):
        with sqlite3.connect(db_path) as con:
            cur = con.cursor()
            cur.execute('''CREATE TABLE states
                        (indexes text, objective real)''')
            cur.execute('''CREATE INDEX idx_objective 
                            ON states (objective);''')
            
            # start from a random sample
            indexes = np.arange(self.offset_idx, self.offset_idx + self.data_pool_size)
            random_subset_indices = np.random.choice(indexes, size=self.optimal_subset_size, replace=False)
            if (num_unique_samples := len(set(random_subset_indices))) != self.optimal_subset_size:
                raise ValueError(f"Unexpected number of indices are selected. Expected {self.optimal_subset_size}, got {num_unique_samples}")
            cur.execute("INSERT INTO states VALUES ('%s', 0)" % json.dumps(random_subset_indices.tolist()))
            con.commit()
            

    def _get_nth_best_subset(self, n: int) -> np.ndarray:
        """Select the nth best by test accuracy"""
        try:
            con = sqlite3.connect(self.db_path)
            cur = con.cursor()
            cur.execute("SELECT * FROM states ORDER BY objective DESC LIMIT 1 OFFSET %d" % n)
            r = cur.fetchone()
        finally:
            con.close()
        return np.array(json.loads(r[0])) - self.offset_idx

    def _create_new_subset_in_place(self, base_subset: np.ndarray) -> np.ndarray:
        """Create a new subset by swapping out one sample from the base_subset
        and swapping in a new sample. This is done inplace for efficiency but it produces side effect of modifying the `base_subset`
        """
        all_indices = np.arange(self.data_pool_size)
        available_examples = np.setdiff1d(all_indices, base_subset)
        in_sample = np.random.choice(available_examples)
        out_sample_idx = np.random.randint(0, len(base_subset) - 1)
        base_subset[out_sample_idx] = in_sample
        if (num_unique_samples := len(set(base_subset))) != self.optimal_subset_size: 
            raise ValueError(f"Expected {self.optimal_subset_size} unique samples for the subset; got {num_unique_samples}")
        return base_subset

    def _insert_run(self, subset_indices: np.ndarray, quality: float) -> None:
        try:
            con = sqlite3.connect(self.db_path)
            cur = con.cursor()
            cur.execute("INSERT INTO states VALUES ('%s', %.8f)" % (json.dumps((subset_indices + self.offset_idx).tolist()), quality))
            con.commit()
        finally:
            con.close()

    def select_new_subset(self, current_num_runs: int) -> np.ndarray:
        # should be extended by different classes
        pass

    def _get_num_runs(self):
        try:
            con = sqlite3.connect(self.db_path)
            cur = con.cursor()
            cur.execute("SELECT COUNT(objective) FROM states")
            r = cur.fetchone()[0]
            con.commit()
        finally:
            con.close()
        return r

    def one_run(self):
        seed_everything(self.seed)
        current_num_run = self._get_num_runs()
        new_subset_indices = self.select_new_subset(current_num_run)
        wandb_run = wandb.init(project=self.wandb_project, entity=self.wandb_entity, tags=[self.subset_trainer.params.model_card])
        wandb.log({"model_card": self.subset_trainer.params.model_card})
        wandb.log({"pool_size": self.data_pool_size})
        wandb.log({"search_size": self.optimal_subset_size})
        wandb.log({"indices": json.dumps(new_subset_indices.tolist())})
        new_subset=self.data_pool.select(new_subset_indices)
        new_quality = self.subset_trainer.train_one_step(new_subset)
        self._insert_run(subset_indices=new_subset_indices, quality=new_quality)
        wandb_run.finish()

    def search(self, n_runs):
        for n in range(n_runs): 
            print(f"### RUN {n}")
            self.one_run()

    def search_til_manual_termination(self):
        while True:
            self.one_run()


class GreedySearcher(SubsetSearcher):
    def select_new_subset(self, current_num_runs: int) -> np.ndarray:
        base_subset = self._get_nth_best_subset(0) 
        self._create_new_subset_in_place(base_subset)
        return base_subset # the altered base_subset
 

class AnnealingSearcher(SubsetSearcher):
    def select_new_subset(self, current_num_runs: int) -> np.ndarray:
        # 0 <= exploration ratio <= 1; the closer to 0, the greedier
        exploration_ratio = max(0, (self.annealing_runs - current_num_runs) / self.annealing_runs)
        # 0 <= nth best <= int(exploration_ratio * current_num_runs); the closer to 0, the greedier
        nth_best = np.random.randint(0, max(1, int(exploration_ratio * current_num_runs)))
        base_subset = self._get_nth_best_subset(nth_best) 
        self._create_new_subset_in_place(base_subset)
        return base_subset # the altered base_subset
 

class WarmupAnnealingSearcher(SubsetSearcher):
    def select_new_subset(self, current_num_runs: int) -> np.ndarray:
        if current_num_runs < self.warmup_runs:
            # create new subset
            base_subset = np.random.choice(self.data_pool_size, size=self.optimal_subset_size, replace=False)
        elif current_num_runs < self.annealing_runs + self.warmup_runs:
            # 0 <= exploration ratio <= 1; the closer to 0, the greedier
            exploration_ratio = max(0, (self.annealing_runs - current_num_runs) / self.annealing_runs)
            # 0 <= nth best <= int(exploration_ratio * current_num_runs); the closer to 0, the greedier
            nth_best = np.random.randint(0, max(1, int(exploration_ratio * current_num_runs)))
            base_subset = self._get_nth_best_subset(nth_best) 
        else:
            base_subset = self._get_nth_best_subset(0) 

        # swap one example
        self._create_new_subset_in_place(base_subset)
        return base_subset # the altered base_subset


class GeneticSearcher(SubsetSearcher):
    def select_new_subset(self, current_num_runs: int) -> np.ndarray:
        if current_num_runs < self.warmup_runs:
            # create new subset
            base_subset = np.random.choice(self.data_pool_size, size=self.optimal_subset_size, replace=False)
        else:
            # 0 <= exploration ratio <= 1; the closer to 0, the greedier
            mother = np.random.randint(0, max(1, self.annealing_runs))
            father = np.random.randint(0, max(1, self.annealing_runs))

            m_subset = self._get_nth_best_subset(mother) 
            f_subset = self._get_nth_best_subset(father) 
            
            # sex
            base_subset = np.concatenate([m_subset, f_subset])
            base_subset = np.unique(base_subset)
            base_subset = np.random.choice(base_subset, size=self.optimal_subset_size, replace=False)

        # mutation
        self._create_new_subset_in_place(base_subset)
        return base_subset # the altered base_subset
