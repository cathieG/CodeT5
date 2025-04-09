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
    df_train = pd.read_csv("flattened_train.csv")
    df_valid = pd.read_csv("flattened_valid.csv")
    df_test = pd.read_csv("flattened_test.csv")
    return df_train, df_valid, df_test


def main():

    print("Start loading flattened data")
    df_train, df_valid, df_test = load_data()





