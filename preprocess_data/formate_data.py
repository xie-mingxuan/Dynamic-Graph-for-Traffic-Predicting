import numpy as np
import pandas as pd

def format_csv(path: str):
    """
    make native dataset to a format dataset, in order to satisfy the coming processes
    :param path: the ABSOLUTE PATH of the dataset
    :return: a format dataset
    """

    with open(path, 'r') as f:
        user_data = json.load(f)

