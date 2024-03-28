import copy

import numpy as np
import torch

from utils.DataLoader import Data


def get_personal_statistic_possibility(dataset: Data):
    """
    get the possibility of each edge
    :param dataset: the train/evaluate/test dataset
    :return: the possibility of nodes' get_off at each station
    """
    possibility_matrix = {}
    for user_id in dataset.src_node_ids:
        if user_id not in possibility_matrix:
            possibility_matrix[user_id] = {
                # 初始化 last_get_on_station 为 0，表示用户没有在任何站点上过车
                "last_get_on_station": 0,
                "matrix": [{
                    "count": [0 for _ in range(392)],
                    "possibility": [0.0 for _ in range(392)],
                    "total": 0
                } for _ in range(392)]}
    return possibility_matrix


def get_total_statistic_possibility(dataset: Data):
    """
    statistic the possibility of each edge
    :return: the possibility of stations' get_off at each station
    """
    possibility_matrix = {}
    for user_id in dataset.src_node_ids:
        if user_id not in possibility_matrix:
            # 初始化 last_get_on_station 为 0，表示用户没有在任何站点上过车
            possibility_matrix[user_id] = 0

    # possibility_matrix["matrix"][i][392] 代表编号为 i 的车站所有的上下车对数
    possibility_matrix["matrix"] = [[0 for _ in range(393)] for _ in range(392)]
    possibility_matrix["possibility"] = [[0.0 for _ in range(392)] for _ in range(392)]
    return possibility_matrix


def update_personal_possibility_matrix(predict_get_on: bool, possibility_matrix: dict, user_id_list: list, station_id_list: list, get_on_off_label_list: np.ndarray):
    """
    update the possibility matrix
    :param predict_get_on: to predict get on or get off
    :param possibility_matrix: the possibility matrix
    :param user_id_list: the user id list
    :param station_id_list: the station id list
    :param get_on_off_label_list: the label of get on or get off
    :return: None
    """

    get_off_indices = np.where(get_on_off_label_list == 1) if not predict_get_on else np.where(get_on_off_label_list == 0)
    get_on_indices = np.where(get_on_off_label_list == 0) if not predict_get_on else np.where(get_on_off_label_list == 1)
    get_on_station_list = station_id_list[get_on_indices]
    get_off_station_list = station_id_list[get_off_indices]
    get_on_user_list = user_id_list[get_on_indices]
    get_off_user_list = user_id_list[get_off_indices]

    assert len(get_on_user_list) == len(get_on_station_list)
    assert len(get_off_user_list) == len(get_off_station_list)

    for i in range(len(get_on_user_list)):
        # 更新用户对应的上车站为 station_id
        possibility_matrix[get_on_user_list[i]]["last_get_on_station"] = get_on_station_list[i]

    ret_matrix = []
    if len(get_off_user_list) == 0:
        return torch.tensor(ret_matrix)

    for i in range(len(get_off_user_list)):
        last_get_on_station = possibility_matrix[get_off_user_list[i]]["last_get_on_station"]
        # 如果这个用户没有上过车，那么就返回全 0 矩阵
        if last_get_on_station == 0:
            ret_matrix.append([0.0 for _ in range(392)])
            continue
        # 如果这个用户已经上过车，就更新对应的上车站信息，同时返回更新之前的概率矩阵
        possibility_matrix[get_off_user_list[i]]["matrix"][last_get_on_station - 1]["count"][get_off_station_list[i] - 1] += 1
        possibility_matrix[get_off_user_list[i]]["matrix"][last_get_on_station - 1]["total"] += 1
        # 返回的是没有加上本次更新的概率
        ret_matrix.append(copy.deepcopy(possibility_matrix[get_off_user_list[i]]["matrix"][last_get_on_station - 1]["possibility"]))

    for user_id in get_off_user_list:
        last_get_on_station = possibility_matrix[user_id]["last_get_on_station"]
        if last_get_on_station == 0 or possibility_matrix[user_id]["matrix"][last_get_on_station - 1]["total"] == 0:
            continue
        for i in range(392):
            possibility_matrix[user_id]["matrix"][last_get_on_station - 1]["possibility"][i] = possibility_matrix[user_id]["matrix"][last_get_on_station - 1]["count"][i] / \
                                                                                               possibility_matrix[user_id]["matrix"][last_get_on_station - 1]["total"]
        # 重新恢复用户未上车的状态
        possibility_matrix[user_id]["last_get_on_station"] = 0
    return torch.tensor(ret_matrix)


def update_total_possibility_matrix(predict_get_on: bool, possibility_matrix: dict, user_id_list: list, station_id_list: list, get_on_off_label_list: np.ndarray):
    get_off_indices = np.where(get_on_off_label_list == 1) if not predict_get_on else np.where(get_on_off_label_list == 0)
    get_on_indices = np.where(get_on_off_label_list == 0) if not predict_get_on else np.where(get_on_off_label_list == 1)
    get_on_station_list = station_id_list[get_on_indices]
    get_off_station_list = station_id_list[get_off_indices]
    get_on_user_list = user_id_list[get_on_indices]
    get_off_user_list = user_id_list[get_off_indices]

    assert len(get_on_user_list) == len(get_on_station_list)
    assert len(get_off_user_list) == len(get_off_station_list)

    for i in range(len(get_on_user_list)):
        # 更新用户对应的上车站为 station_id
        possibility_matrix[get_on_user_list[i]] = get_on_station_list[i]

    ret_matrix = []
    if len(get_off_user_list) == 0:
        return torch.tensor(ret_matrix)

    for i in range(len(get_off_user_list)):
        last_get_on_station = possibility_matrix[get_off_user_list[i]]
        ret_matrix.append(possibility_matrix["possibility"][last_get_on_station - 1])

    for i in range(len(get_off_user_list)):
        last_get_on_station = possibility_matrix[get_off_user_list[i]]
        # 如果这个用户已经上过车，就更新对应的上下车站信息
        if last_get_on_station != 0:
            possibility_matrix["matrix"][last_get_on_station - 1][392] += 1
            possibility_matrix["matrix"][last_get_on_station - 1][get_off_station_list[i] - 1] += 1
            for j in range(392):
                possibility_matrix["possibility"][last_get_on_station - 1][j] = possibility_matrix["matrix"][last_get_on_station - 1][j] / possibility_matrix["matrix"][last_get_on_station - 1][392]

        # 重新恢复用户未上车的状态
        possibility_matrix[get_off_user_list[i]] = 0
    return torch.tensor(ret_matrix)
