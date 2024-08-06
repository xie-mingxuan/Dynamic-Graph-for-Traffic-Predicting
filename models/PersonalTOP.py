import numpy as np
import torch
import torch.nn as nn

from models.modules import TimeEncoder
from utils import HistoricalInteractionSampler


def init_map(station_map: dict, user: int):
    if user not in station_map:
        station_map[user] = {}
        for i in range(1, 393):
            station_map[user][i] = []


class PersonalTOPModel(nn.Module):

    def __init__(self):
        """
        Statistic Personal TOP model.
        """
        super(PersonalTOPModel, self).__init__()

        self.get_on_user_station_map = {}  # 用于预测在某个车站上车的概率，因此以下车站作为查询 key
        self.get_off_user_station_map = {}  # 用于预测在某个车站下车的概率，因此以上车站作为查询 key
        self.user_action_record = {}  # 用户上次的行为记录

    def train_model(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, get_on_off_labels: np.ndarray):
        """
        Compute the possibilities of src nodes and dst nodes.
        :param get_on_off_labels: the get on or get off labels
        :param src_node_ids: the source node ids
        :param dst_node_ids: the destination node ids
        :return: the possibilities
        """

        for src_node_id, dst_node_id, label in zip(src_node_ids, dst_node_ids, get_on_off_labels):
            if src_node_id not in self.user_action_record:
                # 之前没有记录，初始化 map 并添加记录
                init_map(self.get_on_user_station_map, src_node_id)
                init_map(self.get_off_user_station_map, src_node_id)
                self.user_action_record[src_node_id] = {
                    "user_id": src_node_id,
                    "station": dst_node_id,
                    "label": label,
                }
                continue

            if label == 0:
                # 如果要添加的是上车记录
                if self.user_action_record[src_node_id]["label"] == 0:
                    # 如果之前有上车记录，这次还是上车，就直接更新上车记录
                    self.user_action_record[src_node_id]["station"] = dst_node_id
                else:
                    # 如果之前有下车记录，这次是上车，就更新上车 map，并更新为上车记录
                    last_get_off_station = self.user_action_record[src_node_id]["station"]
                    self.get_on_user_station_map[src_node_id][last_get_off_station].append(dst_node_id)
                    self.user_action_record[src_node_id]["station"] = dst_node_id
                    self.user_action_record[src_node_id]["label"] = 0

            else:
                # 如果要添加的是下车记录
                if self.user_action_record[src_node_id]["label"] == 1:
                    # 如果之前有下车记录，这次是下车，就直接更新下车记录
                    self.user_action_record[src_node_id]["station"] = dst_node_id
                else:
                    # 如果之前有上车记录，这次是下车，就更新下车 map，并更新为下车记录
                    last_get_on_station = self.user_action_record[src_node_id]["station"]
                    self.get_off_user_station_map[src_node_id][last_get_on_station].append(dst_node_id)
                    self.user_action_record[src_node_id]["station"] = dst_node_id
                    self.user_action_record[src_node_id]["label"] = 1

    def get_possibility(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, get_on_off_labels: np.ndarray, negative_sampling: bool = False):

        possibilities = []

        for src_node_id, dst_node_id, label in zip(src_node_ids, dst_node_ids, get_on_off_labels):
            if src_node_id not in self.user_action_record:
                # 如果之前没有对这个用户的记录，则认为概率为 0.5，并添加对应的记录
                possibilities.append(0.5)
                init_map(self.get_on_user_station_map, src_node_id)
                init_map(self.get_off_user_station_map, src_node_id)
                self.user_action_record[src_node_id] = {
                    "user_id": src_node_id,
                    "station": dst_node_id,
                    "label": label,
                }
                continue

            if label == 0:
                # 如果要查询的是上车概率
                if self.user_action_record[src_node_id]["label"] == 0:
                    # 如果之前有上车记录，这次还是上车，就直接返回 0，并更新上车记录
                    possibilities.append(0.0)
                    if not negative_sampling:
                        self.user_action_record[src_node_id]["station"] = dst_node_id
                else:
                    # 如果之前有下车记录，这次是上车，就返回概率，并更新上车记录
                    last_get_off_station = self.user_action_record[src_node_id]["station"]
                    if src_node_id not in self.get_on_user_station_map:
                        possibilities.append(0.5)
                    else:
                        frequency = self.get_on_user_station_map[src_node_id][last_get_off_station].count(dst_node_id)
                        if len(self.get_on_user_station_map[src_node_id][last_get_off_station]) == 0:
                            possibilities.append(0.5)
                        else:
                            possibilities.append(frequency / len(self.get_on_user_station_map[src_node_id][last_get_off_station]))

                    if not negative_sampling:
                        self.user_action_record[src_node_id]["station"] = dst_node_id
                        self.user_action_record[src_node_id]["label"] = 0
            else:
                # 如果要查询的是下车概率
                if self.user_action_record[src_node_id]["label"] == 1:
                    # 如果之前有下车记录，这次还是下车，就直接返回 0，并更新下车记录
                    possibilities.append(0.0)
                    if not negative_sampling:
                        self.user_action_record[src_node_id]["station"] = dst_node_id
                else:
                    # 如果之前有上车记录，这次是下车，就返回概率，并更新下车记录
                    last_get_on_station = self.user_action_record[src_node_id]["station"]
                    if src_node_id not in self.get_off_user_station_map:
                        possibilities.append(0.5)
                    else:
                        frequency = self.get_off_user_station_map[src_node_id][last_get_on_station].count(dst_node_id)
                        if len(self.get_off_user_station_map[src_node_id][last_get_on_station]) == 0:
                            possibilities.append(0.5)
                        else:
                            possibilities.append(frequency / len(self.get_off_user_station_map[src_node_id][last_get_on_station]))
                    if not negative_sampling:
                        self.user_action_record[src_node_id]["station"] = dst_node_id
                        self.user_action_record[src_node_id]["label"] = 1
        return torch.tensor(possibilities)
