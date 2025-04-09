import pandas as pd
import numpy as np
from datasets import Dataset
import re
import os
import torch


def flatten_input(dataframe):
    flattened_list = []

    for index, row in dataframe.iterrows():
        flat_code = []
        code = row["cleaned_method"]
        mask_detail = row["target_block"]

        for line in code.splitlines():
            if mask_detail.replace(" ", "") == line.replace(" ", ""):
                flat_code.append(" <mask>")
            else:
                flat_code.append(line)

        flattened_code = re.sub(r"\s+", " ", " ".join(flat_code)).strip()
        flattened_list.append(flattened_code)

    # Return new column as a Series
    return pd.Series(flattened_list, index=dataframe.index)



def main():

    # read files from repository
    df_train = pd.read_csv("data/raw/ft_train.csv")
    df_test = pd.read_csv("data/raw/ft_test.csv")
    df_valid = pd.read_csv("data/raw/ft_valid.csv")

    # Select only the first two columns: 'cleaned_method' and 'target_block'
    df_train = df_train.iloc[:, 0:2]
    df_test = df_test.iloc[:, 0:2]
    df_valid = df_valid.iloc[:, 0:2]

    # Apply flattening
    df_train["flattened_code"] = flatten_input(df_train)
    df_test["flattened_code"] = flatten_input(df_test)
    df_valid["flattened_code"] = flatten_input(df_valid)

    # Save results
    df_train.to_csv("data/flattened_train.csv", index=False)
    df_test.to_csv("data/flattened_test.csv", index=False)
    df_valid.to_csv("data/flattened_valid.csv", index=False)



if __name__ == "__main__":
    main()