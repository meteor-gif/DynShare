import numpy as np


class UsersNeighborsFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_items = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (user, item, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[3])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_items.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[2] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[3] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    find interactions before

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_items[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbors(self, source_nodes, timestamps, num_neighbors=20):
    """
    get temporal neighbors
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_num_neighbors = num_neighbors if num_neighbors > 0 else 1
    neighbors = np.zeros((len(source_nodes), tmp_num_neighbors)).astype(
      np.int32)  
    items = np.zeros((len(source_nodes), tmp_num_neighbors)).astype(
      np.int32)  
    edge_times = np.zeros((len(source_nodes), tmp_num_neighbors)).astype(
      np.float32)  
    edge_idxs = np.zeros((len(source_nodes), tmp_num_neighbors)).astype(
      np.int32)  
    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_items, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  

      if len(source_neighbors) > 0 and num_neighbors > 0:
        if self.uniform:  # uniform sampling of previous neighbors
          sampled_idx = np.random.randint(0, len(source_neighbors), num_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          items[i, :] = source_items[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          items[i, :] = items[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # recent sampling of previous neighbors
          source_edge_times = source_edge_times[-num_neighbors:]
          source_items = source_items[-num_neighbors:]
          source_neighbors = source_neighbors[-num_neighbors:]
          source_edge_idxs = source_edge_idxs[-num_neighbors:]

          assert (len(source_neighbors) <= num_neighbors)
          assert (len(source_items) <= num_neighbors)
          assert (len(source_edge_times) <= num_neighbors)
          assert (len(source_edge_idxs) <= num_neighbors)

          neighbors[i, num_neighbors - len(source_neighbors):] = source_neighbors
          items[i, num_neighbors - len(source_items):] = source_items
          edge_times[i, num_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, num_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, items, edge_idxs, edge_times


class ItemsNeighborsFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_inviters = []
    self.node_to_voters = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (inviter, voter, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[3])
      self.node_to_inviters.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_voters.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[2] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[3] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_inviters[src_idx][:i], self.node_to_voters[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbors(self, source_nodes, timestamps, num_neighbors=20):
    assert (len(source_nodes) == len(timestamps))

    tmp_num_neighbors = num_neighbors if num_neighbors > 0 else 1
    neighbors = np.zeros((len(source_nodes), tmp_num_neighbors)).astype(
      np.int32)  
    items = np.zeros((len(source_nodes), tmp_num_neighbors)).astype(
      np.int32)  
    edge_times = np.zeros((len(source_nodes), tmp_num_neighbors)).astype(
      np.float32)  
    edge_idxs = np.zeros((len(source_nodes), tmp_num_neighbors)).astype(
      np.int32)  

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_items, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  

      if len(source_neighbors) > 0 and num_neighbors > 0:
        if self.uniform:  
          sampled_idx = np.random.randint(0, len(source_neighbors), num_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          items[i, :] = source_items[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          items[i, :] = items[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          source_edge_times = source_edge_times[-num_neighbors:]
          source_items = source_items[-num_neighbors:]
          source_neighbors = source_neighbors[-num_neighbors:]
          source_edge_idxs = source_edge_idxs[-num_neighbors:]

          assert (len(source_neighbors) <= num_neighbors)
          assert (len(source_items) <= num_neighbors)
          assert (len(source_edge_times) <= num_neighbors)
          assert (len(source_edge_idxs) <= num_neighbors)

          neighbors[i, num_neighbors - len(source_neighbors):] = source_neighbors
          items[i, num_neighbors - len(source_items):] = source_items
          edge_times[i, num_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, num_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, items, edge_idxs, edge_times

