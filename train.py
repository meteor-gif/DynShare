import math
import logging
import time
import random
import sys
import argparse
from io import BytesIO

import torch
import pandas as pd
import numpy as np
#import numba
from sklearn.preprocessing import scale

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from modules.share import SHARE
from modules.temporal_neighbors_finder import UsersNeighborsFinder, ItemsNeighborsFinder
from modules.memory import Memory
from utils.utils import EarlyStopMonitor, RandEdgeSampler, eval_edge_prediction
from utils.utils import compute_time_statistics

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(2020)

### Argument and global variables
parser = argparse.ArgumentParser('Interface for SHARE experiments on asymmetric link predictions')
parser.add_argument('--batch_size', type=int, default=4096, help='batch_size')
parser.add_argument('--inviter_num_neighbors', type=int, default=5, help='number of inviter neighbors to sample')
parser.add_argument('--voter_num_neighbors', type=int, default=5, help='number of voter neighbors to sample')
parser.add_argument('--item_num_neighbors', type=int, default=5, help='number of item neighbors to sample')
parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--num_epoches', type=int, default=50, help='number of epochs')
parser.add_argument('--num_layers', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--users_dim', type=int, default=128 * 2, help='Dimensions of the users embedding')
parser.add_argument('--items_dim', type=int, default=128 * 2, help='Dimensions of the items embedding')
parser.add_argument('--time_dim', type=int, default=128, help='Dimensions of the time embedding')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.batch_size
INVITER_NUM_NEIGHBORS = args.inviter_num_neighbors
VOTER_NUM_NEIGHBORS = args.voter_num_neighbors
ITEM_NUM_NEIGHBORS = args.item_num_neighbors
NUM_EPOCHES = args.num_epoches
NUM_HEADS = args.num_heads
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NUM_LAYERS = args.num_layers
LEARNING_RATE = args.lr
USERS_DIM = args.users_dim
ITEMS_DIM = args.items_dim
TIME_DIM = args.time_dim

current_time = str(time.strftime('%Y-%m-%d-%H-%M', time.localtime(int(time.time()))))
MODEL_SAVE_PATH = './saved_models/share.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/share_{epoch}.pth'

### set up logger
### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/log-.log'.format(current_time))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


### Load data and train val test split
inviters = np.load('/data/inviter.npy')
items = np.load('/data/item.npy')
voters = np.load('/data/voter.npy')
timestamps = np.load('/data/timestamp.npy')
users_features = None
items_features = np.load('/data/item_feature.npy')
edges_idxs = np.load('/data/idx.npy')

# timestamps = scale(timestamps + 1)

items_features = scale(items_features)
validation_index = int(len(timestamps) * 0.70)
test_index = int(len(timestamps) * 0.85)

max_users_idx = max(max(inviters), max(voters))
max_items_idx = max(items)

random.seed(2020)

users_distinct_all = set(np.unique(np.hstack([inviters, voters])))
num_users_distinct_all = len(users_distinct_all)


train_inviters = inviters[:validation_index]
train_voters = voters[:validation_index]
train_items = items[:validation_index]
train_timestamps = timestamps[:validation_index]
train_edges_idxs = edges_idxs[:validation_index]

# validation and test with all edges
validation_inviters = inviters[validation_index: test_index]
validation_voters = voters[validation_index: test_index]
validation_items = items[validation_index: test_index]
validation_timestamps = timestamps[validation_index: test_index]
validation_edges_idxs = edges_idxs[validation_index: test_index]

test_inviters = inviters[test_index:]
test_voters = voters[test_index:]
test_items = items[test_index:]
test_timestamps = timestamps[test_index:]
test_edges_idxs = edges_idxs[test_index:]


### Initialize the data structure for graph and edge sampling
# build the graph for fast query
# graph only contains the training data (with 10% nodes removal)
train_inviters_adj_list = [[] for _ in range(max_users_idx + 1)]
train_voters_adj_list = [[] for _ in range(max_users_idx + 1)]
train_items_adj_list = [[] for _ in range(max_items_idx + 1)]
for inviter, item, voter, edge_idx, timestamp in zip(train_inviters, train_items, train_voters, train_edges_idxs, train_timestamps):
    train_inviters_adj_list[inviter].append((voter, item, edge_idx, timestamp))
    train_voters_adj_list[voter].append((inviter, item, edge_idx, timestamp))
    train_items_adj_list[item].append((inviter, voter, edge_idx, timestamp))
train_inviters_neighbors_finder = UsersNeighborsFinder(train_inviters_adj_list, uniform=UNIFORM)
train_voters_neighbors_finder = UsersNeighborsFinder(train_voters_adj_list, uniform=UNIFORM)
train_items_neighbors_finder = ItemsNeighborsFinder(train_items_adj_list, uniform=UNIFORM)

# full graph with all the data for the test and validation purpose
full_inviters_adj_list = [[] for _ in range(max_users_idx + 1)]
full_voters_adj_list = [[] for _ in range(max_users_idx + 1)]
full_items_adj_list = [[] for _ in range(max_items_idx + 1)]
for inviter, item, voter, edge_idx, timestamp in zip(inviters, items, voters, edges_idxs, timestamps):
    full_inviters_adj_list[inviter].append((voter, item, edge_idx, timestamp))
    full_voters_adj_list[voter].append((inviter, item, edge_idx, timestamp))
    full_items_adj_list[item].append((inviter, voter, edge_idx, timestamp))
full_inviters_neighbors_finder = UsersNeighborsFinder(full_inviters_adj_list, uniform=UNIFORM)
full_voters_neighbors_finder = UsersNeighborsFinder(full_voters_adj_list, uniform=UNIFORM)
full_items_neighbors_finder = ItemsNeighborsFinder(full_items_adj_list, uniform=UNIFORM)


train_rand_sampler = RandEdgeSampler(train_inviters, train_voters, seed=2020)
val_rand_sampler = RandEdgeSampler(inviters, voters, seed=0)
test_rand_sampler = RandEdgeSampler(inviters, voters, seed=1)


if users_features is None:
    users_features = np.zeros((max_users_idx + 1, USERS_DIM))
    # users_features = np.eye(max_users_idx + 1)
if items_features is None:
    items_features = np.zeros((max_items_idx + 1, ITEMS_DIM))

mean_time_shift_inviters, std_time_shift_inviters, mean_time_shift_voters, std_time_shift_voters = compute_time_statistics(inviters, voters, timestamps)
## Model initialize
device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')
share = SHARE(train_inviters_neighbors_finder, train_voters_neighbors_finder, train_items_neighbors_finder, users_features, items_features, 
                            mean_time_shift_inviters, std_time_shift_inviters, mean_time_shift_voters, std_time_shift_voters,
                            device=device, users_dim=USERS_DIM, items_dim=ITEMS_DIM, time_dim=TIME_DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, drop_out=DROP_OUT)
optimizer = torch.optim.Adam(share.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
share = share.to(device)

train_num_instances = len(train_inviters)
num_batches = math.ceil(train_num_instances / BATCH_SIZE)

logger.info('num of training instances: {}'.format(train_num_instances))
logger.info('num of batches per epoch: {}'.format(num_batches))


early_stopper = EarlyStopMonitor()
for epoch in range(NUM_EPOCHES):
    start_epoch_time = time.time()
    # Training 
    # training use only training graph

    share.memory.__init_memory__()
    share.inviters_neighbors_finder = train_inviters_neighbors_finder
    share.voters_neighbors_finder = train_voters_neighbors_finder
    share.items_neighbors_finder = train_items_neighbors_finder
    acc, ap, f1, auc, m_loss = [], [], [], [], []
    logger.info('start {} epoch'.format(epoch))

    for k in range(num_batches):
         
        start_batch_time = time.time()
        percent = 100 * k / num_batches
        if k % int(0.3 * num_batches) == 0:
            logger.info('progress: {0:10.4f}'.format(percent))

        start_idx = k * BATCH_SIZE
        end_idx = min(train_num_instances - 1, start_idx + BATCH_SIZE)

        inviters_idxs_batch, items_idxs_batch, voters_idxs_batch = train_inviters[start_idx: end_idx], train_items[start_idx: end_idx], train_voters[start_idx: end_idx]
        timestamps_batch = train_timestamps[start_idx: end_idx]

        size = len(inviters_idxs_batch)
        _, negative_voters_idxs_batch = train_rand_sampler.sample(size)
        
        
        with torch.no_grad():
          pos_label = torch.ones(size, dtype=torch.float, device=device)
          neg_label = torch.zeros(size, dtype=torch.float, device=device)

        optimizer.zero_grad()
        share = share.train()
        start_share_time = time.time()
        pos_prob, neg_prob = share(inviters_idxs_batch, items_idxs_batch, voters_idxs_batch, negative_voters_idxs_batch, 
                                   timestamps_batch, INVITER_NUM_NEIGHBORS, VOTER_NUM_NEIGHBORS, ITEM_NUM_NEIGHBORS)

        loss = criterion(pos_prob, pos_label)
        loss += criterion(neg_prob, neg_label)
        
        loss.backward()
        optimizer.step()
        share.memory.detach_memory()
        # get training results
        with torch.no_grad():
            share = share.eval()
            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            acc.append((pred_label == true_label).mean())
            ap.append(average_precision_score(true_label, pred_score))
            f1.append(f1_score(true_label, pred_label))
            m_loss.append(loss.item())
            auc.append(roc_auc_score(true_label, pred_score))
        

    # validation phase use all information
    share.inviters_neighbors_finder = full_inviters_neighbors_finder
    share.items_neighbors_finder = full_items_neighbors_finder
    share.voters_neighbors_finder = full_voters_neighbors_finder
    val_acc, val_ap, val_auc, val_f1 = eval_edge_prediction(share, val_rand_sampler, 
                                                            validation_inviters, validation_items, validation_voters, 
                                                            validation_timestamps, validation_edges_idxs, 
                                                            INVITER_NUM_NEIGHBORS,VOTER_NUM_NEIGHBORS, ITEM_NUM_NEIGHBORS, BATCH_SIZE)
    
    epoch_time = time.time() - start_epoch_time
    
    logger.info('epoch: {} took {:.2f}s'.format(epoch, epoch_time))
    logger.info('epoch: {}:'.format(epoch))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info('train acc: {}, train auc: {}, train ap: {}, train_f1:{}'.format(np.mean(acc), np.mean(auc), np.mean(ap), np.mean(f1)))
    logger.info('val acc: {}, val auc: {}, val ap: {}, val f1: {}'.format(val_acc, val_auc, val_ap, val_f1))


    if early_stopper.early_stop_check(val_ap):
        logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        share.load_state_dict(torch.load(best_model_path))
        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        share.eval()
        break
    else:
        torch.save(share.state_dict(), get_checkpoint_path(epoch))


# testing phase use all information
share.inviters_neighbors_finder = full_inviters_neighbors_finder
share.items_neighbors_finder = full_items_neighbors_finder
share.voters_neighbors_finder = full_voters_neighbors_finder
test_acc, test_ap, test_auc, test_f1 = eval_edge_prediction(share, test_rand_sampler, test_inviters, test_items, test_voters, 
                                                            test_timestamps, test_edges_idxs, 
                                                            INVITER_NUM_NEIGHBORS, VOTER_NUM_NEIGHBORS, ITEM_NUM_NEIGHBORS, BATCH_SIZE)

# print('Test statistics: -- auc: {}, ap: {}'.format(test_auc, test_ap))
logger.info('Test statistics: -- acc: {}, auc: {}, ap: {}, f1: {}'.format(test_acc, test_auc, test_ap, test_f1))
logger.info('Saving SHARE model')
torch.save(share.state_dict(), MODEL_SAVE_PATH)
logger.info('SHARE models saved')

 




