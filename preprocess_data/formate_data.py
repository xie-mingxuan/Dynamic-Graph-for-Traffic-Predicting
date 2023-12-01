import numpy as np
import pandas as pd


def format_csv(path: str, trans: str = "railway"):
	"""
    make native dataset to a format dataset, in order to satisfy the coming processes
    :param path: the ABSOLUTE PATH of the dataset
    :return: a format dataset
    """

	with open(path, 'r', encoding="utf-8", errors="ignore") as f:
		user_data = pd.read_csv(f, header=None, dtype=str)

	# make the whole interact process into 2 links: get_on (label = 0) and get_off (label = 1)
	for index, row in user_data.iterrows():
		if not (trans == "railway" and row[2] == "R") or (trans == "bus" and row[2] == "B"):
			continue

		get_on_info = []
		get_off_info = []

		user_id = row[1]
		get_on_time = row[4]
		get_on_station = row[5]
		get_off_time = row[9]
		get_off_station = row[10]


if __name__ == '__main__':
	format_csv('../DG_data/t000.csv')
