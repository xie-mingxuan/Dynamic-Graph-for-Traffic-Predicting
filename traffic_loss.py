import torch
import torch.nn as nn


class TrafficLoss(nn.Module):
    def __init__(self, type: str = 'railway'):
        super(TrafficLoss, self).__init__()
        self.railway_matrix = torch.load('railway_matrix.pt')
        self.bus_matrix = torch.load('bus_matrix.pt')

    def forward(self, pred, target):
        per_sample_loss = self.criterion(pred, target)
        return torch.mean(per_sample_loss)
