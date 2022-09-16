def tokenize_function(examples, field="sentence"):
    return tokenizer(examples[field], padding="max_length", max_length=max_length, truncation=True)
