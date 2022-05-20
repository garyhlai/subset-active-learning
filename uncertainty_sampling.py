from datasets import load_dataset, load_metric
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm


def calculate_entropy(prob_dist):
    """
    Returns raw entropy

    Keyword arguments:
        prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0. e.g. tensor([0.0321, 0.6439, 0.0871, 0.2369])
    """
    log_probs = prob_dist * torch.log2(prob_dist)  # multiply each probability by its base 2 log
    raw_entropy = 0 - torch.sum(log_probs, axis=-1)
    return raw_entropy


def uncertainty_sampling(model, ds, n_samples):
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    # calculate entropys for each sample
    model.eval()
    with torch.no_grad():
        preds = []
        for batch in tqdm(dl):
            out = model(**batch)
            preds.append(out.logits)
        preds = torch.cat(preds)
        preds = torch.nn.functional.softmax(preds, dim=-1)
        entropys = calculate_entropy(preds)

    # select data
    return torch.topk(entropys, n_samples).tolist()
