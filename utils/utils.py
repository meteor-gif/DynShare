import torch
from torch import nn
import numpy as np
import math
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

class MergeLayer(torch.nn.Module):
  '''
  Score function
  '''
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class TemporalAttentionLayer(torch.nn.Module):
  """
  Temporal attention layer
  """

  def __init__(self, users_dim, items_dim, time_dim,
               output_dimension, num_heads=2,
               dropout=0.1):
    super(TemporalAttentionLayer, self).__init__()

    self.num_heads = num_heads

    self.users_dim = users_dim
    self.items_dim = items_dim
    self.time_dim = time_dim

    self.query_dim = users_dim + time_dim
    self.key_dim = users_dim + time_dim + items_dim

    self.merger = MergeLayer(self.query_dim, users_dim, users_dim, output_dimension)

    self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim,
                                                   kdim=self.key_dim,
                                                   vdim=self.key_dim,
                                                   num_heads=num_heads,
                                                   dropout=dropout)

  def forward(self, source_nodes_features, source_nodes_time_features, neighbors_features,
              neighbors_time_features, neighbors_edges_features, neighbors_padding_mask):

    source_nodes_features_unrolled = torch.unsqueeze(source_nodes_features, dim=1)

    # query and key
    query = torch.cat([source_nodes_features_unrolled, source_nodes_time_features], dim=2)
    key = torch.cat([neighbors_features, neighbors_edges_features, neighbors_time_features], dim=2)

    query = query.permute([1, 0, 2])  
    key = key.permute([1, 0, 2])  


    # mask
    invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
    neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False

    # attention
    attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=key,
                                                              key_padding_mask=neighbors_padding_mask)


    # output
    attn_output = attn_output.squeeze()
    attn_output_weights = attn_output_weights.squeeze()

    attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
    attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)

    attn_output = self.merger(attn_output, source_nodes_features)

    return attn_output, attn_output_weights

class EarlyStopMonitor(object):
  def __init__(self, max_round=5, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


'''
Time encoder from fucntional analysis
'''
class TimeEncode(torch.nn.Module):
  
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
        
    def forward(self, ts):
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)
        map_ts = ts * self.basis_freq.view(1, 1, -1)
        map_ts += self.phase.view(1, 1, -1)
        
        harmonic = torch.cos(map_ts)

        return harmonic 


def eval_edge_prediction(model, negative_edge_sampler, inviters_idxs, items_idxs, voters_idxs, 
                         timestamps_idxs, edges_idxs, 
                         inviter_num_neighbors, voter_num_neighbors, item_num_neighbors, batch_size):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_acc, val_f1 = [], []
  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(inviters_idxs)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      start_idxs = k * TEST_BATCH_SIZE
      end_idxs = min(num_test_instance, start_idxs + TEST_BATCH_SIZE)
      inviters_idxs_batch = inviters_idxs[start_idxs: end_idxs]
      items_idxs_batch = items_idxs[start_idxs: end_idxs]
      voters_idxs_batch = voters_idxs[start_idxs: end_idxs]
      timestamps_batch = timestamps_idxs[start_idxs: end_idxs]
      edges_idxs_batch = edges_idxs[start_idxs: end_idxs]

      size = len(inviters_idxs_batch)
      _, negative_voters_batch = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model(inviters_idxs_batch, items_idxs_batch, voters_idxs_batch,
                                 negative_voters_batch, timestamps_batch, 
                                 inviter_num_neighbors, voter_num_neighbors, item_num_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      pred_label = pred_score > 0.5
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_acc.append((pred_label == true_label).mean())
      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))
      val_f1.append(f1_score(true_label, pred_label))
      

  return np.mean(val_acc), np.mean(val_ap), np.mean(val_auc), np.mean(val_f1)


def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst