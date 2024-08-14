# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import dgl
import numpy as np
import time
import os
from utils import feature_normalize,fair_metric,sparse_2_edge_index,set_seed,train_val_test_split,laplacian_positional_encoding,\
    laplace_decomp,re_features,load_dataset,adjacency_positional_encoding,get_same_sens_complete_graph,get_same_sens_sub_complete_graph
from sklearn.metrics import f1_score, roc_auc_score
from model import FairGT
import pandas as pd
import random
import argparse
from scipy import sparse as sp
import torchmetrics
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str, default='./data/', help='datapath') # pokec_z
parser.add_argument('--dataset', type=str, default='german', help='Random seed.') # nba,pokec_z,pokec_n,credit,income
parser.add_argument('--gpuid', type=int, default=0, help='Random seed.') # pokec_z

# graphtransformer,san,specformer,nagphormer,fairgt
parser.add_argument('--model', type=str, default='fairgt', help='Random seed.') 
parser.add_argument('--seed', type=int, default=20, help='Random seed.') # 20 22 23 25
parser.add_argument('--hops', type=int, default=2, help='Hop of neighbors to be calculated') # nagphormer,fairgt
parser.add_argument('--pe_dim', type=int, default=2, help='position embedding size') # nagphormer and san
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer size')
parser.add_argument('--nhead', type=int, default=2, help='Number of Transformer heads') # 8
parser.add_argument('--nlayer', type=int, default=1, help='Number of Transformer layers') #1
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout')
parser.add_argument('--self_loop', type=bool, default=False, help='FFN layer size')
parser.add_argument('--peak_lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
# parser.add_argument('--subnum', type=int, default=1000, help='position embedding size') # nagphormer and san

parser.add_argument('--sens_idex', type=bool, default=False, help='FFN layer size')
parser.add_argument('--is_lap', type=bool, default=False, help='FFN layer size')
# parser.add_argument('--is_subgraph', type=bool, default=False, help='FFN layer size')
# parser.add_argument('--readout_nlayers', type=int, default=1, help='Number of Transformer layers') #1
# parser.add_argument('--attention_dropout', type=float, default=0.1, help='Dropout in the attention layer')
# parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=200, help='Patience for early stopping')


parser.add_argument('--metric', type=int, default=7, help='metric') # 1acc 2loss 3
# args = parser.parse_args([])

args = parser.parse_args()


is_batch=False
args.is_subgraph=True
#
label_number=1000

# device = args.device
# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device=torch.device("cuda:"+str(args.gpuid) if torch.cuda.is_available() else "cpu")

set_seed(args.seed)

adj, feature, labels, sens, idx_train, idx_val, idx_test = load_dataset(args)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
# print('train:',len(idx_train),'val:',len(idx_val),'test:',len(idx_test))


edge_list = (adj != 0).nonzero()
g = dgl.DGLGraph()
g.add_nodes(feature.shape[0])
g.add_edges(edge_list[0], edge_list[1])
edge_feat_dim = 1
g.edata['feat'] = torch.zeros(g.number_of_edges(), edge_feat_dim).long()

if args.model=='fairgt':
    lpe=None
    filepath = './PE_files/'+args.model+'/'+args.dataset+'_'+str(args.pe_dim)+'_eig.pt'
    try:
        # 
        eignvalue, eignvector = torch.load(filepath)
        lpe=eignvector
    except FileNotFoundError:
        print('pe file no exist!')
        #
        eignvalue, eignvector = adjacency_positional_encoding(adj, args.pe_dim) 
        torch.save([eignvalue, eignvector], filepath)
        lpe=eignvector
    # get_same_sens_complete_graph    
    features = torch.cat((feature, lpe), dim=1) 
    print('original graph')
    adj = get_same_sens_complete_graph(adj, sens, args)
    # adj = torch.from_numpy(adj.todense())
    processed_features = re_features(adj, features, args.hops)
    g.ndata['feat'] = processed_features
    
g = g.to(device)

args.nclass = 2
# nclass = args.nclass
args.in_dim = g.ndata['feat'].shape[-1]
nclass = args.nclass
model = FairGT(vars(args)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
labels, idx_train, idx_val, idx_test, sens = labels.to(device), idx_train.to(device), idx_val.to(device), idx_test.to(device), sens.to(device)
edge_index = None


val_loss_list, val_acc_list, val_auc_list, val_f1_list, val_sp_list, val_eo_list = [],[],[],[],[],[]
test_loss_list, test_acc_list, test_auc_list, test_f1_list, test_sp_list, test_eo_list = [],[],[],[],[],[]

patience = args.patience

res = []
# min_loss = 100.0
epoch=args.epochs
# best_metric = -999998.0
# max_acc1=None
# new_metric = -999999.0
rec_val=None
best_metric_val = -999998.0
metric_val = -999999.0

rec_test=None
best_metric_test = -999998.0
metric_test = -999999.0


# args.metric=4
if args.metric==1: # acc
    print('metric: acc')
elif args.metric==2: # loss
    print('metric: loss')
elif args.metric==3: # -sp-eo
    print('metric: -sp-eo')
elif args.metric==4: # val_acc-val_parity-val_equality
    print('metric: acc-sp-eo')
elif args.metric==5: # val_f1-val_parity-val_equality
    print('metric: f1-sp-eo')
elif args.metric==6: # val_auc-val_parity-val_equality
    print('metric: auc-sp-eo')
elif args.metric==7: # val_acc-val_parity-val_equality
    print('metric: acc-sp')
    
counter = 0
evaluation = torchmetrics.Accuracy(task='multiclass', num_classes=nclass)
end = time.time()
# print('success load data, time is:{:.3f}'.format(end-start))
print('='*10,"Start Training: ",args.dataset,'='*10)

train_start = time.time()
train_time=0
for idx in range(epoch):
    model.train()
    optimizer.zero_grad()

    logits = model(g.ndata['feat'])
        
    loss = F.cross_entropy(logits[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    
    model.eval()

    val_loss = F.cross_entropy(logits[idx_val], labels[idx_val]).item()
    val_acc = evaluation(logits[idx_val].cpu(), labels[idx_val].cpu()).item()
    val_auc_roc = roc_auc_score(labels[idx_val].cpu().numpy(), F.softmax(logits,dim=1)[idx_val,1].detach().cpu().numpy())
    val_f1 = f1_score(labels[idx_val].cpu().numpy(),logits[idx_val].detach().cpu().argmax(dim=1))
    val_sp, val_eo = fair_metric(labels, sens, torch.argmax(logits, dim=1), idx_val)
    
    test_loss = F.cross_entropy(logits[idx_test], labels[idx_test]).item()
    test_acc = evaluation(logits[idx_test].cpu(), labels[idx_test].cpu()).item()
    test_auc_roc = roc_auc_score(labels[idx_test].cpu().numpy(), F.softmax(logits,dim=1)[idx_test,1].detach().cpu().numpy())
    test_f1 = f1_score(labels[idx_test].cpu().numpy(),logits[idx_test].detach().cpu().argmax(dim=1))
    test_sp, test_eo = fair_metric(labels, sens, torch.argmax(logits, dim=1), idx_test)
    
    # acc, sp, eo, f1, auc, epoch
    # res.append([100 * test_acc, 100 * parity, 100 * equality, f1_test, auc_roc_test,(idx+1)])
    res.append([100 * test_acc, 100 * test_sp, 100 * test_eo, 100 * test_f1, 100 * test_auc_roc, (idx+1)])

    # new_metric = (val_acc-val_parity-val_equality)
    # if args.metric==1: # acc
    #     new_metric = val_acc
    # elif args.metric==2: # loss
    #     new_metric = -val_loss
    # elif args.metric==3 and idx>100: # -sp-eo
    #     new_metric = (-val_parity-val_equality)
    # elif args.metric==4: # val_acc-val_parity-val_equality
    #     new_metric = (val_acc-val_parity-val_equality)
    # elif args.metric==5: # val_f1-val_parity-val_equality
    #     new_metric = (val_f1-val_parity-val_equality)
    # elif args.metric==6: # val_auc-val_parity-val_equality
    #     new_metric = (val_auc_roc-val_parity-val_equality)
    # elif args.metric==7: # val_acc-val_parity-val_equality
    #     new_metric = (val_acc-val_parity)
    
    if args.metric==1: # acc
        metric_val = val_acc
        metric_test = test_acc
    elif args.metric==2: # loss
        metric_val = -val_loss
        metric_test = -test_loss
    elif args.metric==3: # -sp-eo
        metric_val = (-val_sp-val_eo)
        metric_test = (-test_sp-test_eo)
    elif args.metric==4: # val_acc-val_parity-val_equality
        metric_val = (val_acc-val_sp-val_eo)
        metric_test = (test_acc-test_sp-test_eo)
    elif args.metric==5: # val_f1-val_parity-val_equality
        metric_val = (val_f1-val_sp-val_eo)
        metric_test = (test_f1-test_sp-test_eo)
    elif args.metric==6: # val_auc-val_parity-val_equality
        metric_val = (val_auc_roc-val_sp-val_eo)
        metric_test = (test_auc_roc-test_sp-test_eo)
    elif args.metric==7: # val_acc-val_parity
        metric_val = (val_acc-val_sp)
        metric_test = (test_acc-test_sp)
    
    if metric_val > best_metric_val and val_sp>0 and (idx+1)>=10:
        best_metric_val = metric_val
        rec_val = res[-1]
        counter = 0 
    else:
        counter += 1
    
    if metric_test > best_metric_test and epoch>5 and test_sp>0:
        best_metric_test = metric_test
        rec_test = res[-1]
    
    if (idx+1)%10==0:
        print('epoch:{:05d}, val_loss{:.4f}, test_acc:{:.4f}, parity:{:.4f}, equality:{:.4f}, f1:{:.4f}, auc:{:.4f}'.format(idx+1, val_loss, 100 * test_acc, 100 * test_sp, 100 * test_eo, 100 * test_f1, 100 * test_auc_roc ))
    
    if counter == args.patience:
    # if counter > 200 and (idx+1)>500:
        train_end = time.time()
        train_time = (train_end-train_start)
        print('success train data, time is:{:.3f}'.format(train_time))
        break
    
train_end = time.time()
train_time = (train_end-train_start)
print('success train data, time is:{:.3f}'.format(train_time))

max_memory_cached = torch.cuda.max_memory_cached(device=device) / 1024 ** 2 
max_memory_allocated = torch.cuda.max_memory_allocated(device=device) / 1024 ** 2
print("Max memory cached:", max_memory_cached, "MB")
print("Max memory allocated:", max_memory_allocated, "MB")
# print('final_test_acc:', max_acc1[0], 'parity:',max_acc1[1],'equality:', max_acc1[2] ,'f1:',max_acc1[3] ,'auc:',max_acc1[4], 'epoch:',max_acc1[5])
print('best val -- acc: {:.4f}, parity: {:.4f}, equality: {:.4f}, f1: {:.4f}, auc: {:.4f}, epoch: {:04d}'.format(rec_val[0],rec_val[1],rec_val[2],rec_val[3],rec_val[4],rec_val[5]))
print('best test -- acc: {:.4f}, parity: {:.4f}, equality: {:.4f}, f1: {:.4f}, auc: {:.4f}, epoch: {:04d}'.format(rec_test[0],rec_test[1],rec_test[2],rec_test[3],rec_test[4],rec_test[5]))


print(args)


train_logs = dict()
train_logs['model']=type(model).__name__
train_logs['dataset']=args.dataset
# train_logs.update(vars(args))
train_logs['seed']=args.seed
train_logs['hidden_dim']=args.hidden_dim
train_logs['nlayer']=args.nlayer
train_logs['nheads']=args.nhead
# train_logs['readoutnlayer']=args.readout_nlayers
train_logs['dropout']=args.dropout
train_logs['pe_dim']=args.pe_dim
# train_logs['K']=args.K
train_logs['lr']=args.peak_lr
train_logs['weight_decay']=args.weight_decay
train_logs['patience']=args.patience
train_logs['data_num']=len(feature)
train_logs['train_num']=len(idx_train)
train_logs['val_num']=len(idx_val)
train_logs['test_num']=len(idx_test)
train_logs['attr_num']=g.ndata['feat'].shape[0]
train_logs['TestAcc']=rec_val[0]
train_logs['TestSP']=rec_val[1]
train_logs['TestEO']=rec_val[2]
train_logs['TestF1']=rec_val[3]
train_logs['TestAUC']=rec_val[4]
train_logs['best_epoch']=rec_val[5]
train_logs['Maxcached']=max_memory_cached
train_logs['Maxallocated']=max_memory_allocated
train_logs['train_time(s)']=train_time
train_logs['args']=str(args)
train_logs = pd.DataFrame(train_logs, index=[0])

logs_path = './logs/'
# logs_path = './logs/'
# train_log_save_file=logs_path+'FairGT'+'_train_log.csv'
# test_log_save_file=logs_path+dataname+'_test.csv'
train_log_save_file=logs_path+'FairGT'+'_train_log_gpu_'+str(args.gpuid)+'.csv'

if os.path.exists(train_log_save_file): # add
    train_logs.to_csv(train_log_save_file, mode='a', index=False, header=0)
else: # create
    train_logs.to_csv(train_log_save_file, index=False)

print('log over')
