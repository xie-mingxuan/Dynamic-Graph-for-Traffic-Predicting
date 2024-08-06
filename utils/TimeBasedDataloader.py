import numpy as np
from torch.utils.data import Dataset, DataLoader


class TimeBasedDataset(Dataset):
    def __init__(self, timestamp_list, time_gap):
        self.timestamp_list = timestamp_list
        self.time_gap = time_gap
        self.batch_indices_list = self._generate_batch_indices()

    def _generate_batch_indices(self):
        batch_indices_list = []
        current_idx = 0
        while current_idx < len(self.timestamp_list):
            start_idx = current_idx
            current_timestamp = self.timestamp_list[start_idx]
            while current_idx < len(self.timestamp_list) and (self.timestamp_list[current_idx] - current_timestamp) < self.time_gap:
                current_idx += 1
            if current_idx == start_idx:
                current_idx += 1
            batch_indices = list(range(start_idx, current_idx))
            batch_indices_list.append(batch_indices)
        return batch_indices_list

    def __len__(self):
        return len(self.batch_indices_list)

    def __getitem__(self, idx):
        return np.array(self.batch_indices_list[idx])
