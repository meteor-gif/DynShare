from time import time
import torch
from torch import nn
class ExplicitUsersHistoryAggregation(torch.nn.Module):
    def __init__(self, users_dim, items_dim, time_dim):
        super(ExplicitUsersHistoryAggregation, self).__init__()

        self.history_updater = nn.GRUCell(input_size=items_dim + users_dim + time_dim, hidden_size=users_dim // 2)

    def forward(self, users_embeddings_history, items_embeddings_minus, others_embeddings_minus, time_diffs_tensor):
        input_edges_features = torch.cat([others_embeddings_minus, items_embeddings_minus, time_diffs_tensor], dim=1)
        users_embeddings = self.history_updater(input_edges_features, users_embeddings_history)

        return users_embeddings

class ImplicitUsersHistoryAggregation(torch.nn.Module):
    def __init__(self, users_dim, items_dim, time_dim):
        super(ImplicitUsersHistoryAggregation, self).__init__()

        self.history_updater = nn.GRUCell(input_size=items_dim + users_dim + time_dim, hidden_size=users_dim // 2)

    def forward(self, users_embeddings_history, items_embeddings_minus, others_embeddings_minus, time_diffs_tensor):
        input_edges_features = torch.cat([others_embeddings_minus, items_embeddings_minus, time_diffs_tensor], dim=1)
        users_embeddings = self.history_updater(input_edges_features, users_embeddings_history)

        return users_embeddings


class ItemsHistoryAggregation(torch.nn.Module):
    def __init__(self, users_dim, items_dim):
        super(ItemsHistoryAggregation, self).__init__()

        self.history_updater = nn.GRUCell(input_size=users_dim * 2, hidden_size=items_dim)

    def forward(self, items_embeddings_history, inviters_embeddings_minus, voters_embeddings_minus):
        users_embeddings_minus = torch.cat([inviters_embeddings_minus, voters_embeddings_minus], dim=1)
        items_embeddings = self.history_updater(users_embeddings_minus, items_embeddings_history)

        return items_embeddings
