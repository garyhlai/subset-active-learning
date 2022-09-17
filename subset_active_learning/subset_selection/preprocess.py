from transformers import AutoTokenizer
import datasets


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
