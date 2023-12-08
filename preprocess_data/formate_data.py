import numpy as np
import pandas as pd
from datetime import datetime


def format_csv(path: str, trans: str = "railway"):
    """
    make native dataset to a format dataset, in order to satisfy the coming processes
    :param trans: method of transportation, railway or bus, default is railway
    :param path: path of the dataset
    :return: a format dataset
    """

    with open(path, 'r', encoding="utf-8", errors="ignore") as f:
        user_data = pd.read_csv(f, header=None, dtype=str)

    user_id_map = {}
    user_id_list = []
    unique_user_id_index = 0
    station_id_map = {}
    station_id_list = []
    unique_station_id_index = 0
    time_list = []
    label_list = []
    index_list = []
    feature_list = []
    real_index = 0

    # make the whole interact process into 2 links: get_on (label = 0) and get_off (label = 1)
    for index, row in user_data.iterrows():
        if not (trans == "railway" and row[2] == "R") or (trans == "bus" and row[2] == "B"):
            continue

        user = row[1]
        get_on_time = int(datetime.strptime(row[4], '%Y/%m/%d %H:%M:%S').timestamp())
        get_on_station = row[5]
        get_off_time = int(datetime.strptime(row[9], '%Y/%m/%d %H:%M:%S').timestamp())
        get_off_station = row[10]

        # convert user id to int
        if user not in user_id_map:
            user_id_map[user] = unique_user_id_index
            unique_user_id_index += 1
        # convert station id to int
        if get_on_station not in station_id_map:
            station_id_map[get_on_station] = unique_station_id_index
            unique_station_id_index += 1
        if get_off_station not in station_id_map:
            station_id_map[get_off_station] = unique_station_id_index
            unique_station_id_index += 1

        # mark get_on data as label 0, get_off data as label 1
        user_id_list.append(user_id_map[user])
        station_id_list.append(station_id_map[get_on_station])
        time_list.append(get_on_time)
        label_list.append(0)
        index_list.append(real_index * 2)
        feature_list.append([0.0 for _ in range(172)])

        user_id_list.append(user_id_map[user])
        station_id_list.append(station_id_map[get_off_station])
        time_list.append(get_off_time)
        label_list.append(1)
        index_list.append(real_index * 2 + 1)
        feature_list.append([0.0 for _ in range(172)])

        real_index += 1

    df = pd.DataFrame({'u': user_id_list,
                       'i': station_id_list,
                       'ts': time_list,
                       'label': label_list,
                       'idx': index_list})
    df = df.sort_values(by="ts", ascending=True)

    return df, np.array(feature_list)


if __name__ == '__main__':
    format_csv('../DG_data/t000/t000.csv')
