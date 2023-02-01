from time import time
import torch
from torch import nn
import logging
import numpy as np
from modules.history_neighbors_aggregation import InvitersNeighborsEmbeddingsAggregation, VotersNeighborsEmbeddingsAggregation, ItemsNeighborsEmbeddingsAggregation
from modules.projection import TimeProjectionEmbedddings
from modules.nodes_history_aggregation import ExplicitUsersHistoryAggregation, ImplicitUsersHistoryAggregation, ItemsHistoryAggregation
from modules.memory import Memory
from utils.utils import MergeLayer, TimeEncode
class SHARE(torch.nn.Module):
    def __init__(self, inviters_neighbors_finder, voters_neighbors_finder, items_neighbors_finder, users_features, items_features,
                 mean_time_shift_inviters, std_time_shift_inviters, mean_time_shift_voters, std_time_shift_voters, device,
                 users_dim=128 * 2, items_dim=128 * 2, time_dim=128, num_layers=2, num_heads=2, drop_out=0.1):
        super(SHARE, self).__init__()
        
        self.num_layers = num_layers 
        self.inviters_neighbors_finder = inviters_neighbors_finder
        self.voters_neighbors_finder = voters_neighbors_finder
        self.items_neighbors_finder = items_neighbors_finder
        self.logger = logging.getLogger(__name__)
        self.users_features = users_features
        self.items_static_features = items_features
        
        self.users_dim = users_dim
        self.items_dim = items_dim
        self.time_dim = time_dim

        self.mean_time_shift_inviters = mean_time_shift_inviters
        self.std_time_shift_inviters = std_time_shift_inviters
        self.mean_time_shift_voters= mean_time_shift_voters
        self.std_time_shift_voters = std_time_shift_voters
        
        self.device = device
        self.time_encoder = TimeEncode(expand_dim=time_dim)
        self.memory = Memory(users_features.shape[0], self.items_static_features.shape[0], users_dim, items_dim // 2, device=device)

        self.inviters_neighbors_aggregation = InvitersNeighborsEmbeddingsAggregation(self.memory, self.inviters_neighbors_finder, self.users_features, self.items_static_features, 
                                                                            self.device, self.time_encoder, self.users_dim // 2, self.items_dim, 
                                                                            self.time_dim, num_layers, num_heads=2, dropout=0.1)
        self.voters_neighbors_aggregation = VotersNeighborsEmbeddingsAggregation(self.memory, self.voters_neighbors_finder, self.users_features, self.items_static_features, 
                                                                            self.device, self.time_encoder, self.users_dim // 2, self.items_dim, 
                                                                            self.time_dim, num_layers, num_heads=2, dropout=0.1)
        self.items_neighbors_aggregation = ItemsNeighborsEmbeddingsAggregation(self.memory, self.items_neighbors_finder, self.users_features, self.items_static_features, 
                                                                            self.device, self.time_encoder, self.users_dim, self.items_dim // 2, 
                                                                            self.time_dim, num_layers=1, num_heads=2, dropout=0.1)

        self.explicit_history_updater = ExplicitUsersHistoryAggregation(self.users_dim, self.items_dim, self.time_dim)
        self.implicit_history_updater = ImplicitUsersHistoryAggregation(self.users_dim, self.items_dim, self.time_dim)
        self.items_history_updater = ItemsHistoryAggregation(self.users_dim, self.items_dim // 2)
        
        self.inviters_embeddings_projection = TimeProjectionEmbedddings(self.users_dim // 2)
        self.voters_embeddings_projection = TimeProjectionEmbedddings(self.users_dim // 2)
        # self.items_embeddings_projection = TimeProjectionEmbedddings(self.users_dim // 2)

        self.explicit_minus_layer = nn.Linear(users_dim * 3 // 2, users_dim // 2)
        self.implicit_minus_layer = nn.Linear(users_dim * 3 // 2, users_dim // 2)
        self.items_merge_layer = nn.Linear(items_dim, items_dim // 2)

        self.affinity_score = MergeLayer(users_dim, users_dim, users_dim, 1)

        self.users_layer_norm = nn.LayerNorm(users_dim // 2, elementwise_affine=False)
        self.items_layer_norm = nn.LayerNorm(items_dim // 2, elementwise_affine=False)
        self.projection_layer_norm = nn.LayerNorm(users_dim // 2, elementwise_affine=False)

        
    def forward(self, inviters_idxs_cut, items_idxs_cut, voters_idxs_cut, negative_voters_idxs_cut, 
                timestamps_cut, inviter_num_neighbors, voter_num_neighbors, item_num_neighbors):
        
        n_samples = len(inviters_idxs_cut)


        # memory embeddings
        inviters_embeddings_cut = self.memory.get_users_memory(inviters_idxs_cut)
        inviters_invitation_cut = inviters_embeddings_cut[:, :self.users_dim // 2]
        inviters_vote_cut = inviters_embeddings_cut[:, self.users_dim // 2:]

        voters_embeddings_cut = self.memory.get_users_memory(voters_idxs_cut)
        voters_invitation_cut = voters_embeddings_cut[:, :self.users_dim // 2]
        voters_vote_cut = voters_embeddings_cut[:, self.users_dim // 2:]

        negative_voters_embeddings_cut = self.memory.get_users_memory(negative_voters_idxs_cut)
        negative_voters_invitation_cut = negative_voters_embeddings_cut[:, :self.users_dim // 2]
        negative_voters_vote_cut = negative_voters_embeddings_cut[:, self.users_dim // 2:]

        items_dynamic_embeddings_cut = self.memory.get_items_memory(items_idxs_cut)
        items_static_embeddings_cut = torch.from_numpy(self.items_static_features[items_idxs_cut]).float().to(self.device)
        
        # memory last update
        inviters_invitation_time_diffs = ((torch.from_numpy(timestamps_cut).float().to(self.device) - self.memory.inviters_last_update[inviters_idxs_cut]) - self.mean_time_shift_inviters) / self.std_time_shift_inviters
        inviters_vote_time_diffs = ((torch.from_numpy(timestamps_cut).float().to(self.device) - self.memory.voters_last_update[inviters_idxs_cut]) - self.mean_time_shift_voters) / self.std_time_shift_voters
        voters_invitation_time_diffs = ((torch.from_numpy(timestamps_cut).float().to(self.device) - self.memory.inviters_last_update[voters_idxs_cut]) - self.mean_time_shift_inviters) / self.std_time_shift_inviters
        voters_vote_time_diffs = ((torch.from_numpy(timestamps_cut).float().to(self.device) - self.memory.voters_last_update[voters_idxs_cut]) - self.mean_time_shift_voters) / self.std_time_shift_voters
        negative_voters_invitation_time_diffs = ((torch.from_numpy(timestamps_cut).float().to(self.device) - self.memory.inviters_last_update[negative_voters_idxs_cut]) - self.mean_time_shift_inviters) / self.std_time_shift_inviters
        negative_voters_vote_time_diffs = ((torch.from_numpy(timestamps_cut).float().to(self.device) - self.memory.voters_last_update[negative_voters_idxs_cut]) - self.mean_time_shift_voters) / self.std_time_shift_voters

        # neighbors aggregation
        inviters_invitation_neighbors = self.inviters_neighbors_aggregation.compute_embeddings(inviters_idxs_cut, timestamps_cut, self.num_layers, inviter_num_neighbors)
        inviters_vote_neighbors = self.voters_neighbors_aggregation.compute_embeddings(inviters_idxs_cut, timestamps_cut, self.num_layers, voter_num_neighbors)
        voters_vote_neighbors = self.voters_neighbors_aggregation.compute_embeddings(voters_idxs_cut, timestamps_cut, self.num_layers, voter_num_neighbors)
        voters_invitation_neighbors = self.inviters_neighbors_aggregation.compute_embeddings(voters_idxs_cut, timestamps_cut, self.num_layers, inviter_num_neighbors)
        negative_voters_vote_neighbors = self.voters_neighbors_aggregation.compute_embeddings(negative_voters_idxs_cut, timestamps_cut, self.num_layers, voter_num_neighbors)
        negative_voters_invitation_neighbors = self.inviters_neighbors_aggregation.compute_embeddings(negative_voters_idxs_cut, timestamps_cut, self.num_layers, inviter_num_neighbors)
        items_neighbors = self.items_neighbors_aggregation.compute_embeddings(items_idxs_cut, timestamps_cut, 1, item_num_neighbors)

        # projection
        inviters_invitation_projection = self.projection_layer_norm(self.inviters_embeddings_projection.compute_time_projection_embeddings(inviters_invitation_cut, inviters_invitation_time_diffs))
        inviters_vote_projection = self.projection_layer_norm(self.voters_embeddings_projection.compute_time_projection_embeddings(inviters_vote_cut, inviters_vote_time_diffs))
        voters_invitation_projection = self.projection_layer_norm(self.inviters_embeddings_projection.compute_time_projection_embeddings(voters_invitation_cut, voters_invitation_time_diffs))
        voters_vote_projection = self.projection_layer_norm(self.voters_embeddings_projection.compute_time_projection_embeddings(voters_vote_cut, voters_vote_time_diffs))
        negative_voters_invitation_projection = self.projection_layer_norm(self.inviters_embeddings_projection.compute_time_projection_embeddings(negative_voters_invitation_cut, negative_voters_invitation_time_diffs))
        negative_voters_vote_projection = self.projection_layer_norm(self.voters_embeddings_projection.compute_time_projection_embeddings(negative_voters_vote_cut, negative_voters_vote_time_diffs))
 
        # embeddings minus

        inviters_invitation_minus = self.explicit_minus_layer(torch.cat((inviters_invitation_cut, inviters_invitation_neighbors, inviters_invitation_projection), dim=1))
        inviters_vote_minus = self.implicit_minus_layer(torch.cat((inviters_vote_cut, inviters_vote_neighbors, inviters_vote_projection), dim=1))
        voters_vote_minus = self.explicit_minus_layer(torch.cat((voters_vote_cut, voters_vote_neighbors, voters_vote_projection), dim=1))
        voters_invitation_minus = self.implicit_minus_layer(torch.cat((voters_invitation_cut, voters_invitation_neighbors, voters_invitation_projection), dim=1))
        negative_voters_vote_minus = self.explicit_minus_layer(torch.cat((negative_voters_vote_cut, negative_voters_vote_neighbors, negative_voters_vote_projection), dim=1))
        negative_voters_invitation_minus = self.implicit_minus_layer(torch.cat((negative_voters_invitation_cut, negative_voters_invitation_neighbors, negative_voters_invitation_projection), dim=1))

        items_dynamic_embeddings_minus = self.items_merge_layer(torch.cat((items_dynamic_embeddings_cut, items_neighbors), dim=1))
        

        # embeddings minus
        inviters_embeddings_minus = torch.cat((inviters_invitation_minus, inviters_vote_minus), dim=1)
        voters_embeddings_minus = torch.cat((voters_invitation_minus, voters_vote_minus), dim=1)
        negative_voters_embeddings_minus = torch.cat((negative_voters_invitation_minus, negative_voters_vote_minus), dim=1)
        items_embeddings_minus = torch.cat((items_static_embeddings_cut, items_dynamic_embeddings_minus), dim=1) # note
        
        
        pos_score = torch.squeeze(self.affinity_score(inviters_embeddings_minus + items_embeddings_minus, voters_embeddings_minus))
        neg_score = torch.squeeze(self.affinity_score(inviters_embeddings_minus + items_embeddings_minus, negative_voters_embeddings_minus))
        
        # embeddings update
        
        inviters_invitation_time_diffs_tensor = torch.squeeze(self.time_encoder(torch.unsqueeze(inviters_invitation_time_diffs, dim=1)), dim=1)
        inviters_vote_time_diffs_tensor = torch.squeeze(self.time_encoder(torch.unsqueeze(inviters_vote_time_diffs, dim=1)), dim=1)
        voters_invitation_time_diffs_tensor = torch.squeeze(self.time_encoder(torch.unsqueeze(voters_invitation_time_diffs, dim=1)), dim=1)
        voters_vote_time_diffs_tensor = torch.squeeze(self.time_encoder(torch.unsqueeze(voters_vote_time_diffs, dim=1)), dim=1)

        new_inviters_invitation_cut = self.users_layer_norm(self.explicit_history_updater(inviters_invitation_minus, items_embeddings_minus, 
                                                            voters_embeddings_minus, inviters_invitation_time_diffs_tensor))
        new_voters_vote_cut = self.users_layer_norm(self.explicit_history_updater(voters_vote_minus, items_embeddings_minus, 
                                                            inviters_embeddings_minus, voters_vote_time_diffs_tensor))
        new_inviters_vote_cut = self.users_layer_norm(self.implicit_history_updater(inviters_vote_minus, items_embeddings_minus, voters_embeddings_minus,  inviters_vote_time_diffs_tensor))
        new_voters_invitation_cut = self.users_layer_norm(self.implicit_history_updater(voters_invitation_minus, items_embeddings_minus, inviters_embeddings_minus,  voters_invitation_time_diffs_tensor))
        
        new_items_dynamic_embeddings_cut = self.items_layer_norm(self.items_history_updater(items_dynamic_embeddings_cut, inviters_embeddings_minus, voters_embeddings_minus))
        
        new_inviters_embeddings_cut = torch.cat((new_inviters_invitation_cut, new_inviters_vote_cut), dim=1)
        new_voters_embeddings_cut = torch.cat((new_voters_vote_cut, new_voters_invitation_cut), dim=1)
        new_items_embeddings_cut = torch.cat((items_static_embeddings_cut, new_items_dynamic_embeddings_cut), dim=1)
        

        # memory
        self.memory.update_users_memory(inviters_idxs_cut, new_inviters_embeddings_cut.data.clone())
        self.memory.update_items_memory(items_idxs_cut, new_items_dynamic_embeddings_cut.data.clone())
        self.memory.update_users_memory(voters_idxs_cut, new_voters_embeddings_cut.data.clone())
        self.memory.inviters_last_update[inviters_idxs_cut] = torch.from_numpy(timestamps_cut).float().to(self.device).data.clone()
        self.memory.voters_last_update[voters_idxs_cut] = torch.from_numpy(timestamps_cut).float().to(self.device).data.clone()
        self.memory.items_last_update[items_idxs_cut] = torch.from_numpy(timestamps_cut).float().to(self.device).data.clone()

        
        return pos_score.sigmoid(), neg_score.sigmoid()

    

    