import numpy as np
import torch
import torch.nn as nn


class TrafficLoss(nn.Module):
    def __init__(self, method: str = 'railway', direction: str = 'get_off'):
        super(TrafficLoss, self).__init__()
        self.direction = direction
        if method == 'railway':
            self.distance_matrix = torch.tensor(np.load("../DG_data/station_distance.npy"))
            self.station_matrix = torch.tensor(np.load("../DG_data/station_count.npy.npy"))

    def forward(self, pred, target, label):
        pred = pred.long()
        target = target.long()

        # As the index of station starts from 1, but information of the station starts from 0, we need to minus 1
        loss = torch.log(1 + self.distance_matrixp[pred - 1, target - 1] * self.station_matrix[pred - 1, target - 1])
        if self.direction == 'get_off':
            loss = loss[label == 1]
        return torch.mean(loss)
