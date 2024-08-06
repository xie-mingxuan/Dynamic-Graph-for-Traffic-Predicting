import numpy as np


class HistoricalInteractionSampler:
    def __init__(self, adj_list: list):
        # list of each node's neighbor ids, edge ids and interaction times, which are sorted by interaction times
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        # the list at the first position in adj_list is empty, hence, sorted() will return an empty list for the first position
        # its corresponding value in self.nodes_neighbor_ids, self.nodes_edge_ids, self.nodes_neighbor_times will also be empty with length 0
        for node_idx, per_node_neighbors in enumerate(adj_list):
            # per_node_neighbors is a list of tuples (neighbor_id, edge_id, timestamp)
            # sort the list based on timestamps, sorted() function is stable
            # Note that sort the list based on edge id is also correct, as the original data file ensures the interactions are chronological
            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_per_node_neighbors]))
            self.nodes_edge_ids.append(np.array([x[1] for x in sorted_per_node_neighbors]))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_per_node_neighbors]))

    def sample_interactions(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        Sample historical interactions for a node
        :param node_ids: the node id
        :param node_interact_times: the timestamp
        :return: a list of sampled nodes
        """

        nodes_neighbor_ids = []
        # each entry in position (i,j) represents the id of the edge with src node node_ids[i] and dst node nodes_neighbor_ids[i][j] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_edge_ids = []
        # each entry in position (i,j) represents the interaction time between src node node_ids[i] and dst node nodes_neighbor_ids[i][j], before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_times = []

        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times = self.find_interactions_before(node_id=node_id, timestamp=node_interact_time)
            nodes_neighbor_ids.append(node_neighbor_ids)
            nodes_edge_ids.append(node_edge_ids)
            nodes_neighbor_times.append(node_neighbor_times)
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def find_interactions_before(self, node_id: int, timestamp: float):
        i = np.searchsorted(self.nodes_neighbor_times[node_id], timestamp)
        return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i]
