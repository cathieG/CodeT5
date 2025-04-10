import re
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    T5ForConditionalGeneration,
    RobertaTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)


# Load flattened data
def load_data():
    
    # if you wish to use the provided flattened data:
    df_train = pd.read_csv("data/processed/flattened_train.csv")
    df_valid = pd.read_csv("data/processed/flattened_valid.csv")
    df_test = pd.read_csv("data/proessed/flattened_test.csv")

    # if you ran preprocessing.py and wish to use the data you flattened,
    # comment out the three lines above and use the three lines below instead:

    #df_train = pd.read_csv("data/flattened_train.csv")
    #df_valid = pd.read_csv("data/flattened_valid.csv")
    #df_test = pd.read_csv("data/flattened_test.csv")

    return df_train, df_valid, df_test

# Load Model & Tokenizer
def load_model_and_tokenizer(model_checkpoint="Salesforce/codet5-small"):
    tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
    tokenizer.add_tokens(["<mask>"])
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

# Preprocess_function
def preprocess_function(examples, tokenizer):
    inputs = examples["flattened_code"]
    targets = examples["target_block"]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize Data, converts pandas Dataframes into Hugging Face datasets
def tokenize_data(df_train, df_valid, df_test, tokenizer):
    hf_train = Dataset.from_pandas(df_train)
    hf_valid = Dataset.from_pandas(df_valid)
    hf_test = Dataset.from_pandas(df_test)

    tokenized_train = hf_train.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_valid = hf_valid.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_test = hf_test.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    return DatasetDict({
        "train": tokenized_train,
        "validation": tokenized_valid,
        "test": tokenized_test
    })

# Training Arguments
def setup_training_args(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return TrainingArguments(
        output_dir=output_dir,
        report_to="none",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        logging_steps=100,
        push_to_hub=False,
    )

# Train and Evaluate using the Trainer API
def train_and_evaluate(model, tokenizer, datasets, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    metrics = trainer.evaluate(datasets["test"])
    print("Test Loss Metrics:", metrics)


def zip_model(output_dir):
    shutil.make_archive(output_dir, 'zip', output_dir)
    print(f"Model saved and zipped to {output_dir}.zip")


def main():

    print("Start loading flattened data")
    df_train, df_valid, df_test = load_data()
    print("Finished loading data")

    print("Now loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer()

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU found. Using CPU.")
    print("Finished loading model and tokenizer")

    print("Tokenizing datasets")
    datasets = tokenize_data(df_train, df_valid, df_test, tokenizer)
    print("Finished tokenizing datasets")

    print("Start training")
    training_args = setup_training_args("output_model")
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    train_and_evaluate(model, tokenizer, datasets, training_args)
    print("Task finished.")

    print("Start zipping model")
    zip_model("output_model")


if __name__ == "__main__":
    main()



