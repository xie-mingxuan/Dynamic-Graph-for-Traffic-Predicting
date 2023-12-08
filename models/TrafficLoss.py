import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class TrafficLoss(nn.Module):
    def __init__(self, method: str = 'railway', direction: str = 'get_off', distance_matrix: Tensor = None, station_matrix: Tensor = None, lambda_dist: float = 1.0, lambda_stat: float = 1.0):
        super(TrafficLoss, self).__init__()
        self.direction = direction
        if method == 'railway':
            self.distance_matrix = distance_matrix
            self.station_matrix = station_matrix
        self.lambda_dist = lambda_dist
        self.lambda_stat = lambda_stat
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target):
        # Standard cross entropy loss
        # Since the target indices are 1-indexed, we subtract 1 from them
        ce_loss = self.cross_entropy_loss(pred, target - 1)

        # Average cross entropy loss over valid indices
        avg_ce_loss = torch.mean(ce_loss)

        # Predicted indices
        predicted_indices = torch.argmax(pred, dim=1)

        # Distance and station loss components
        # Since the target indices are 1-indexed, we subtract 1 from them
        dist_loss = self.distance_matrix[target - 1, predicted_indices - 1]
        stat_loss = self.station_matrix[target - 1, predicted_indices - 1]

        # Normalize or scale dist_loss and stat_loss
        normalized_dist_loss = torch.mean(dist_loss) / torch.max(self.distance_matrix)
        normalized_stat_loss = torch.mean(stat_loss) / torch.max(self.station_matrix)

        # Combine losses
        total_loss = avg_ce_loss + self.lambda_dist * normalized_dist_loss + self.lambda_stat * normalized_stat_loss
        return total_loss
