import torch
from torch import nn
import numpy as np
from utils.utils import TemporalAttentionLayer

class NeighborsEmbeddingsAggregation(torch.nn.Module):
  def __init__(self, memory, neighbors_finder, users_features, items_static_features, device, time_encoder,
               users_dim, items_dim, time_dim, num_layers, num_heads=2, dropout=0.1):
    super(NeighborsEmbeddingsAggregation, self).__init__()

    self.memory = memory
    self.neighbors_finder = neighbors_finder
    self.users_features = users_features
    self.items_static_features = items_static_features
    self.device = device
    self.time_encoder = time_encoder
    
    self.users_dim = users_dim
    self.items_dim = items_dim
    self.time_dim = time_dim
    

    self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
      users_dim=users_dim,
      items_dim=items_dim,
      time_dim=time_dim,
      output_dimension=users_dim,
      num_heads=num_heads,
      dropout=dropout)
      for _ in range(num_layers)])

  def compute_embeddings(self, nodes_idxs, timestamps, num_layers, num_neighbors):

    pass

  def neighbors_aggregation(self, num_layers, source_nodes_features, source_nodes_time_embeddings,
                neighbor_embeddings,
                edges_time_embeddings, edges_features, mask):
    attention_model = self.attention_models[num_layers - 1]

    source_embedding, _ = attention_model(source_nodes_features,
                                          source_nodes_time_embeddings,
                                          neighbor_embeddings,
                                          edges_time_embeddings,
                                          edges_features,
                                          mask)

    return source_embedding




class InvitersNeighborsEmbeddingsAggregation(NeighborsEmbeddingsAggregation):


  def compute_embeddings(self, inviters_idxs, timestamps, num_layers, num_neighbors):

    assert (num_layers >= 0)

    
    timestamps_tensor = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim = 1)

    # source nodes time span = t - t = 0
    inviters_timestamps_embeddings = self.time_encoder(torch.zeros_like(timestamps_tensor))

    inviters_features = torch.from_numpy(self.users_features[inviters_idxs, : self.users_dim]).float().to(self.device) + self.memory.get_users_memory(inviters_idxs)[:, : self.users_dim]

    if num_layers == 0:
      return inviters_features
    else:
      inviters_conv_features = self.compute_embeddings(inviters_idxs, 
                                                        timestamps, 
                                                        num_layers=num_layers - 1, 
                                                        num_neighbors=num_neighbors)
      neighbors_voters_idxs, neighbors_items_idxs, neighbors_edges_idxs, neighbors_edges_timestamps = self.neighbors_finder.get_temporal_neighbors(inviters_idxs,
                                                                                                            timestamps,
                                                                                                            num_neighbors=num_neighbors)

      neighbors_voters_idxs_tensor = torch.from_numpy(neighbors_voters_idxs).long().to(self.device)

      neighbors_items_idxs_tensor = torch.from_numpy(neighbors_items_idxs).long().to(self.device)

      neighbors_edges_idxs_tensor = torch.from_numpy(neighbors_edges_idxs).long().to(self.device)

      neighbors_edges_time_deltas = timestamps[:, np.newaxis] - neighbors_edges_timestamps

      neighbors_edges_time_deltas_tensor = torch.from_numpy(neighbors_edges_time_deltas).float().to(self.device)

      neighbors_voters_idxs = neighbors_voters_idxs.flatten()
      neighbors_voters_embeddings = self.compute_embeddings(neighbors_voters_idxs,
                                                   np.repeat(timestamps, num_neighbors),
                                                   num_layers=num_layers - 1,
                                                   num_neighbors=num_neighbors)

      effective_num_neighbors = num_neighbors if num_neighbors > 0 else 1
      neighbors_voters_embeddings = neighbors_voters_embeddings.view(len(inviters_idxs), effective_num_neighbors, -1)
      neighbors_edges_time_deltas_embeddings = self.time_encoder(neighbors_edges_time_deltas_tensor)

      neighbors_items_features = torch.cat((torch.from_numpy(self.items_static_features[neighbors_items_idxs, :]).float().to(self.device), self.memory.get_items_memory(neighbors_items_idxs)), dim=-1)
      mask = neighbors_voters_idxs_tensor == 0

      inviters_embeddings = self.neighbors_aggregation(num_layers, inviters_conv_features,
                                        inviters_timestamps_embeddings,
                                        neighbors_voters_embeddings,
                                        neighbors_edges_time_deltas_embeddings,
                                        neighbors_items_features,
                                        mask)

      return inviters_embeddings


class VotersNeighborsEmbeddingsAggregation(NeighborsEmbeddingsAggregation):


  def compute_embeddings(self, voters_idxs, timestamps, num_layers, num_neighbors):

    assert (num_layers >= 0)

    timestamps_tensor = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

    # source nodes time span = t - t = 0
    voters_timestamps_embeddings = self.time_encoder(torch.zeros_like(timestamps_tensor))
    voters_features = torch.from_numpy(self.users_features[voters_idxs, self.users_dim: ]).float().to(self.device) + self.memory.get_users_memory(voters_idxs)[:, self.users_dim: ]

    if num_layers == 0:
      return voters_features
    else:
      voters_conv_features = self.compute_embeddings(voters_idxs, 
                                                        timestamps, 
                                                        num_layers=num_layers - 1, 
                                                        num_neighbors=num_neighbors)
      neighbors_inviters_idxs, neighbors_items_idxs, neighbors_edges_idxs, neighbors_edges_timestamps = self.neighbors_finder.get_temporal_neighbors(voters_idxs,
                                                                                                            timestamps,
                                                                                                            num_neighbors=num_neighbors)

      neighbors_inviters_idxs_tensor = torch.from_numpy(neighbors_inviters_idxs).long().to(self.device)

      neighbors_items_idxs_tensor = torch.from_numpy(neighbors_items_idxs).long().to(self.device)

      neighbors_edges_idxs_tensor = torch.from_numpy(neighbors_edges_idxs).long().to(self.device)

      neighbors_edges_time_deltas = timestamps[:, np.newaxis] - neighbors_edges_timestamps

      neighbors_edges_time_deltas_tensor = torch.from_numpy(neighbors_edges_time_deltas).float().to(self.device)

      neighbors_inviters_idxs = neighbors_inviters_idxs.flatten()

      neighbors_inviters_embeddings = self.compute_embeddings(neighbors_inviters_idxs,
                                                   np.repeat(timestamps, num_neighbors),
                                                   num_layers=num_layers - 1,
                                                   num_neighbors=num_neighbors)

      effective_num_neighbors = num_neighbors if num_neighbors > 0 else 1
      neighbors_inviters_embeddings = neighbors_inviters_embeddings.view(len(voters_idxs), effective_num_neighbors, -1)
      neighbors_edges_time_deltas_embeddings = self.time_encoder(neighbors_edges_time_deltas_tensor)

      neighbors_items_features = torch.cat((torch.from_numpy(self.items_static_features[neighbors_items_idxs, :]).float().to(self.device), self.memory.get_items_memory(neighbors_items_idxs)), dim=-1)
      mask = neighbors_inviters_idxs_tensor == 0

      voters_embeddings = self.neighbors_aggregation(num_layers, voters_conv_features,
                                        voters_timestamps_embeddings,
                                        neighbors_inviters_embeddings,
                                        neighbors_edges_time_deltas_embeddings,
                                        neighbors_items_features,
                                        mask)

      return voters_embeddings



class ItemsNeighborsEmbeddingsAggregation(NeighborsEmbeddingsAggregation):

  def __init__(self, memory, neighbors_finder, users_features, items_static_features, device, time_encoder,
               users_dim, items_dim, time_dim, num_layers, num_heads=2, dropout=0.1):
    super(ItemsNeighborsEmbeddingsAggregation, self).__init__(memory, neighbors_finder, users_features, items_static_features, device, time_encoder,
                                                              users_dim, items_dim, time_dim, num_layers, num_heads, dropout)

    self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
      users_dim=items_dim,
      items_dim=users_dim + items_dim,
      time_dim=time_dim,
      output_dimension=items_dim,
      num_heads=num_heads,
      dropout=dropout)
      for _ in range(num_layers)])


  def compute_embeddings(self, items_idxs, timestamps, num_layers, num_neighbors):

    assert (num_layers >= 0)

    
    timestamps_tensor = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim = 1)

    # source nodes time span = t - t = 0
    items_timestamps_embeddings = self.time_encoder(torch.zeros_like(timestamps_tensor))

    items_features = torch.from_numpy(self.items_static_features[items_idxs]).float().to(self.device) + self.memory.get_items_memory(items_idxs)

    if num_layers == 0:
      return items_features
    else:
      items_conv_features = self.compute_embeddings(items_idxs, 
                                                        timestamps, 
                                                        num_layers=num_layers - 1, 
                                                        num_neighbors=num_neighbors)
      neighbors_inviters_idxs, neighbors_voters_idxs, neighbors_edges_idxs, neighbors_edges_timestamps = self.neighbors_finder.get_temporal_neighbors(items_idxs,
                                                                                                            timestamps,
                                                                                                            num_neighbors=num_neighbors)

      
      neighbors_inviters_embeddings = self.memory.get_users_memory(neighbors_inviters_idxs)

      neighbors_voters_embeddings = self.memory.get_users_memory(neighbors_voters_idxs)

      neighbors_edges_idxs_tensor = torch.from_numpy(neighbors_edges_idxs).long().to(self.device)

      neighbors_edges_time_deltas = timestamps[:, np.newaxis] - neighbors_edges_timestamps

      neighbors_edges_time_deltas_tensor = torch.from_numpy(neighbors_edges_time_deltas).float().to(self.device)

      effective_num_neighbors = num_neighbors if num_neighbors > 0 else 1
      neighbors_inviters_embeddings = neighbors_inviters_embeddings.view(len(neighbors_inviters_idxs), effective_num_neighbors, -1)
      neighbors_voters_embeddings = neighbors_voters_embeddings.view(len(neighbors_voters_idxs), effective_num_neighbors, -1)
      neighbors_edges_time_deltas_embeddings = self.time_encoder(neighbors_edges_time_deltas_tensor)

      mask = neighbors_edges_idxs_tensor == 0

      items_embeddings = self.neighbors_aggregation(num_layers, items_conv_features,
                                        items_timestamps_embeddings,
                                        neighbors_inviters_embeddings,
                                        neighbors_edges_time_deltas_embeddings,
                                        neighbors_voters_embeddings,
                                        mask)

      return items_embeddings



