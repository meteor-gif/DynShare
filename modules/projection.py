import torch
from torch import nn
from torch.nn import functional as F
import math

class TimeProjectionEmbedddings(torch.nn.Module):
    def __init__(self,  users_dim):
        super(TimeProjectionEmbedddings, self).__init__()

        class NormalLinear(nn.Linear):
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer_early = NormalLinear(1, users_dim)
        self.embedding_layer_late = NormalLinear(1, users_dim)
        self.transformer_vector = nn.Parameter(torch.Tensor(users_dim))
        self.transformer_vector.data.normal_(0, 1. / math.sqrt(users_dim))

    def compute_time_projection_embeddings(self, nodes_embeddings, nodes_time_diffs):
        projection_alpha = torch.sum(nodes_embeddings * self.transformer_vector, dim=1)
        expectation_time = torch.sqrt(math.pi / (2 * torch.exp(projection_alpha)))
        expectation_time = (expectation_time - torch.mean(expectation_time)) / torch.std(expectation_time)
        time_diffs_expection = nodes_time_diffs - expectation_time
        time_diffs_expection = torch.unsqueeze(time_diffs_expection, 1)
        nodes_embeddings_early = nodes_embeddings * (1 + self.embedding_layer_early(time_diffs_expection))
        nodes_embeddings_late = nodes_embeddings * (1 + self.embedding_layer_late(time_diffs_expection))
        nodes_embeddings_projection = nodes_embeddings_early.masked_fill_((time_diffs_expection >= 0), 0) + nodes_embeddings_late.masked_fill_((time_diffs_expection < 0), 0)
            

        return nodes_embeddings_projection

