import torch
from torch import nn
import math
from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

  def __init__(self, num_users, num_items, users_embeddings_dimension, items_embeddings_dimension, device='cpu'):
    super(Memory, self).__init__()
    self.num_users = num_users
    self.num_items = num_items
    self.users_embeddings_dimension = users_embeddings_dimension
    self.items_embeddings_dimension = items_embeddings_dimension
    self.device = device
  
    self.__init_memory__()

  def __init_memory__(self):
    """
    Initializes the memory to all zeros. It should be called at the start of each epoch.
    """
    # Treat memory as parameter so that it is saved and loaded together with the model
    self.users_memory = nn.Parameter(torch.zeros((self.num_users, self.users_embeddings_dimension)).to(self.device),
                               requires_grad=False)
    self.users_memory.data.normal_(0, 1 / math.sqrt(self.users_embeddings_dimension))
    self.items_memory = nn.Parameter(torch.zeros((self.num_items, self.items_embeddings_dimension)).to(self.device),
                               requires_grad=False)
    self.items_memory.data.normal_(0, 1 / math.sqrt(self.items_embeddings_dimension))
    self.inviters_last_update = nn.Parameter(torch.zeros(self.num_users).to(self.device),
                                    requires_grad=False)
    self.voters_last_update = nn.Parameter(torch.zeros(self.num_users).to(self.device),
                                    requires_grad=False)
    self.items_last_update = nn.Parameter(torch.zeros(self.num_items).to(self.device),
                                    requires_grad=False)
    


  def get_users_memory(self, users_idxs):
    return self.users_memory[users_idxs, :]

  def get_items_memory(self, items_idxs):
    return self.items_memory[items_idxs, :]

  def update_users_memory(self, users_idxs, values):
    self.users_memory[users_idxs, :] = values

  def update_items_memory(self, items_idxs, values):
    self.items_memory[items_idxs, :] = values

  def detach_memory(self):
    self.users_memory.detach_()
    self.items_memory.detach_()
    self.inviters_last_update.detach_()
    self.voters_last_update.detach_()
    self.items_last_update.detach_()
