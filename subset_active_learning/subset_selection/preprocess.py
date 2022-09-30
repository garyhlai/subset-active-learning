from transformers import AutoTokenizer
import datasets

def get_dataset(ds_name: str, model_card: str) -> datasets.DatasetDict:
    if ds_name == 'sst2':
        ds = preprocess_sst2(model_card)
    elif ds_name == 'mnli':
        ds = preprocess_mnli(model_card)
    elif ds_name == 'qqp':
        ds = preprocess_qqp(model_card)
    return ds

def preprocess_sst2(model_card: str) -> datasets.DatasetDict:
    sst2 = datasets.load_dataset("sst")
    max_length = 66
    tokenizer = AutoTokenizer.from_pretrained(model_card)

    def tokenize_function(examples, field="sentence"):
        return tokenizer(examples[field], padding="max_length", max_length=max_length, truncation=True)

    tokenized_sst2 = sst2.map(tokenize_function, batched=False)
    tokenized_sst2 = tokenized_sst2.rename_column("label", "scalar_label")
    tokenized_sst2 = tokenized_sst2.map(lambda x: {"labels": 0 if x["scalar_label"] < 0.5 else 1})
    tokenized_sst2.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    return tokenized_sst2

def preprocess_mnli(model_card: str) -> datasets.DatasetDict:
    sst2 = datasets.load_dataset('glue', 'mnli')
    max_length = 63
    tokenizer = AutoTokenizer.from_pretrained(model_card)

    def tokenize_function(examples, field='sentence'):
        full_str = '%s %s %s' % (examples['premise'], tokenizer.sep_token, examples['hypothesis'])
        return tokenizer(full_str, padding='max_length', max_length=int(max_length)*2+3, truncation=True)

    tokenized_sst2 = sst2.map(tokenize_function, batched=False)
    tokenized_sst2 = tokenized_sst2.rename_column("label", "labels")
    tokenized_sst2.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    return tokenized_sst2

def preprocess_qqp(model_card: str) -> datasets.DatasetDict:
    sst2 = datasets.load_dataset('glue', 'qqp')
    max_length = 24
    tokenizer = AutoTokenizer.from_pretrained(model_card)

    def tokenize_function(examples, field='sentence'):
        full_str = '%s %s %s' % (examples['question1'], tokenizer.sep_token, examples['question2'])
        return tokenizer(full_str, padding='max_length', max_length=int(max_length)*2+3, truncation=True)

    tokenized_sst2 = sst2.map(tokenize_function, batched=False)
    tokenized_sst2 = tokenized_sst2.rename_column("label", "labels")
    tokenized_sst2.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    return tokenized_sst2
