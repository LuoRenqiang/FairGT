import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import pickle
# from torch_sparse import spspmm
import os
import re
import copy
import networkx as nx
import numpy as np
# import scipy.sparse as sp
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import random
import dgl
import time
import pandas as pd



def load_dataset(args):
    datapath = args.datapath
    dataname = args.dataset +'/'
    if args.dataset=='nba':
        # edge_df = pd.read_csv('../data/nba/' + 'nba_relationship.txt', sep='\t')
        edges_unordered = np.genfromtxt(datapath + dataname + 'nba_relationship.txt').astype('int')
        # node_df = pd.read_csv(os.path.join('../dataset/nba/', 'nba.csv'))
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'nba.csv'))
        print('load edge data')
        predict_attr = 'SALARY'
        labels = idx_features_labels[predict_attr].values
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        sens_attr = "country"
        # labels = y
        adj_start = time.time()
        # feature = node_df[node_df.columns[2:]]
        feature = idx_features_labels[header]
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = ["country"])

        # idx = node_df['user_id'].values # for relations
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        print('create adj time is {:.3f}'.format((adj_end-adj_start)))
        # print('adj created!')
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        feature = feature_normalize(feature)
        feature = torch.FloatTensor(feature)
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        
        # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,args.seed)
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
        
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels))
        return adj, feature, labels, sens, idx_train, idx_val, idx_test # 不包含label [0,1(大于1的转成1)]以外的值的id

    elif args.dataset=='pokec_z':
        edges_unordered = np.genfromtxt(datapath + dataname + 'region_job_relationship.txt').astype('int')
        predict_attr = 'I_am_working_in_field'
        sens_attr = 'region'
        print('Loading {} dataset'.format(args.dataset))
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'region_job.csv'))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)

        # header.remove(sens_attr)
        # header.remove(predict_attr)
        feature = idx_features_labels[header]
        # feature=feature_normalize(idx_features_labels[header])
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        #-----
        adj_start = time.time()
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = ["region"])

        # idx = node_df['user_id'].values # for relations
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        feature = feature_normalize(feature)
        feature = torch.FloatTensor(feature)
        # return feature
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,args.seed)
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
       
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels))
        return adj, feature, labels, sens, idx_train, idx_val, idx_test
    elif args.dataset=='pokec_n':
        edges_unordered = np.genfromtxt(datapath + dataname + 'region_job_2_relationship.txt').astype('int')
        predict_attr = 'I_am_working_in_field'
        sens_attr = 'region'
        print('Loading {} dataset'.format(args.dataset))
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'region_job_2.csv'))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)

        # header.remove(sens_attr)
        # header.remove(predict_attr)
        feature = idx_features_labels[header]
        # feature=feature_normalize(idx_features_labels[header])
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        #-----
        adj_start = time.time()
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = ["region"])

        # idx = node_df['user_id'].values # for relations
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        feature = feature_normalize(feature)
        feature = torch.FloatTensor(feature)
        # return feature
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,args.seed)
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
       
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels))
        return adj, feature, labels, sens, idx_train, idx_val, idx_test
    elif args.dataset=='credit':
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'credit.csv'))
        edges_unordered = np.genfromtxt(datapath + dataname + 'credit_edges.txt').astype('int')
        sens_attr="Age"
        predict_attr="NoDefaultNextMonth"
        print('Loading {} dataset'.format(args.dataset))
        # header = list(idx_features_labels.columns)
        header = list(idx_features_labels.columns)
        header.remove('Single')
        header.remove(predict_attr)
        
        feature = idx_features_labels[header]
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = ["Age"])
        adj_start = time.time()
        idx = np.arange(feature.shape[0]) 
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        feature = feature_normalize(feature)
        feature = torch.FloatTensor(feature)
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
       
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels))
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test
    
    elif args.dataset=='income':
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'income.csv'))
        edges_unordered = np.genfromtxt(datapath + dataname + 'income_edges.txt').astype('int')
        sens_attr="race"
        predict_attr="income"
        print('Loading {} dataset'.format(args.dataset))
        header = list(idx_features_labels.columns) #list将括号里的内容变为数组
        header.remove(predict_attr) #header.remove删除括号内的东西
        feature = idx_features_labels[header]
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = ["race"])
        adj_start = time.time()
        idx = np.arange(feature.shape[0]) 
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        feature = feature_normalize(feature)
        feature = torch.FloatTensor(feature)
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels))
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test
    
    elif args.dataset=='german':
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'german.csv'))
        edges_unordered = np.genfromtxt(datapath + dataname + 'german_edges.txt').astype('int')
        print('Loading {} dataset'.format(args.dataset))
        sens_attr="Gender"
        predict_attr="GoodCustomer"
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove('OtherLoansAtStore')
        header.remove('PurposeOfLoan')
        
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0
        feature = idx_features_labels[header]
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = ["Gender"])
        
        # features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        feature = sp.csr_matrix(feature, dtype=np.float32)
        labels = idx_features_labels[predict_attr].values
        labels[labels == -1] = 0
        
        adj_start = time.time()
        idx = np.arange(feature.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature.todense())
        # feature = feature_normalize(feature)
        feature = torch.FloatTensor(feature)
        labels = torch.LongTensor(labels) 
        
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels))
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test
    
    elif args.dataset=='bail':
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'bail.csv'))
        edges_unordered = np.genfromtxt(datapath + dataname + 'bail_edges.txt').astype('int')
        print('Loading {} dataset'.format(args.dataset))
        sens_attr="WHITE"
        predict_attr="RECID"
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        
        feature = idx_features_labels[header]
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = ["WHITE"])
        
        adj_start = time.time()
        idx = np.arange(feature.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        feature = feature_normalize(feature)
        feature = torch.FloatTensor(feature)
        labels = torch.LongTensor(labels) 
        
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels))
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test
        
def train_val_test_split(labels,train_ratio=0.5,val_ratio=0.25,seed=20,label_number=1000):
    import random
    random.seed(seed)
    label_idx_0 = np.where(labels==0)[0]  # 只要label为0和1的
    label_idx_1 = np.where(labels==1)[0]  # 
    random.shuffle(label_idx_0) 
    random.shuffle(label_idx_1)
    position1 = train_ratio
    position2 = train_ratio + val_ratio
    idx_train = np.append(label_idx_0[:min(int(position1 * len(label_idx_0)), label_number//2)], 
                          label_idx_1[:min(int(position1 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(position1 * len(label_idx_0)):int(position2 * len(label_idx_0))], 
                        label_idx_1[int(position1 * len(label_idx_1)):int(position2 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(position2 * len(label_idx_0)):],
                         label_idx_1[int(position2 * len(label_idx_1)):])
    print('train,val,test:',len(idx_train),len(idx_val),len(idx_test))
    return idx_train, idx_val, idx_test



def sparse_2_edge_index(adj):   
    edge_index_origin = adj.nonzero()
    edge_index = torch.stack([torch.from_numpy(edge_index_origin[0]).long(), torch.from_numpy(edge_index_origin[1]).long()])
    return edge_index

def fair_metric(y, sens, output, idx):
    val_y = y[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

    pred_y = (output[idx].squeeze() > 0).type_as(y).cpu().numpy()

    parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1))

    return parity, equality


def feature_normalize(feature): 
    '''sum_norm'''
    feature = np.array(feature)
    rowsum = feature.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return feature / rowsum


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def accuracy(output, labels): # logits,label()
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_batch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix

# SAN position encoding
def laplace_decomp(g, max_freqs):

    # Laplacian
    n = g.number_of_nodes()
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L.toarray()) # 前m小
    EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)
    
    if n<max_freqs:
        # g.ndata['EigVecs'] = F.pad(EigVecs, (0, max_freqs-n), value=float('nan'))
        EigVecs = F.pad(EigVecs, (0, max_freqs-n), value=float('nan'))
    else:
        # g.ndata['EigVecs']= EigVecs
        EigVecs = EigVecs
        
    #Save eigenvales and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(EigVals)))) #Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    
    if n<max_freqs:
        EigVals = F.pad(EigVals, (0, max_freqs-n), value=float('nan')).unsqueeze(0)
    else:
        EigVals=EigVals.unsqueeze(0)
        
    #Save EigVals node features
    # g.ndata['EigVals'] = EigVals.repeat(g.number_of_nodes(),1).unsqueeze(2)
    EigVals = EigVals.repeat(g.number_of_nodes(),1).unsqueeze(2)
    return EigVecs, EigVals

# GraphTransformer position encoding
def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    # Laplacian

    #adjacency_matrix(transpose, scipy_fmt="csr")
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() # 第二小开始，从小到大

    return lap_pos_enc

def adjacency_positional_encoding(g, pos_enc_dim):
    # adj = g.adjacency_matrix_scipy(return_edge_ids=False)
    # adj = g.adj_sparse('coo',return_edge_ids=False)
    # adj = g.adjacency_matrix(scipy_fmt="coo")
    eignvalue, eignvector = sp.linalg.eigsh(g, which='LM', k=pos_enc_dim)
    # eignvalue, eignvector = sp.linalg.eigsh(g.adjacency_matrix_scipy(return_edge_ids=False).astype(float), which='LM', k=pos_enc_dim)
    eignvalue = torch.from_numpy(eignvalue).float()
    eignvector = torch.from_numpy(eignvector).float()
    return eignvalue, eignvector

def re_features(adj, features, K): 
    if K==0:
        return features.unsqueeze(1)
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1]) # (N, 1, K+1, d )
    
    for i in range(features.shape[0]): # node id

        nodes_features[i, 0, 0, :] = features[i]

    x = features + torch.zeros_like(features)

    for i in range(K): # 0 -> K-1

        x = torch.matmul(adj, x)

        for index in range(features.shape[0]):

            nodes_features[index, 0, i + 1, :] = x[index]        

    nodes_features = nodes_features.squeeze()


    return nodes_features


def nor_matrix(adj, a_matrix):

    nor_matrix = torch.mul(adj, a_matrix)
    row_sum = torch.sum(nor_matrix, dim=1, keepdim=True)
    nor_matrix = nor_matrix / row_sum

    return nor_matrix


# 保留多少比例的边
def set_seed(seed = 20):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

# 只保留相同边的属性的结构,并在上面drop操作
# def get_only_homo_edges(adj, keepratio, onlysame=False, seed=22):
# 仅保留部分敏感属性相同的边
def get_only_homo_edges(adj, sens, keepratio, seed=22):
    # set_seed(seed)
    edge_index = adj.nonzero().T # [2e, 2] --> [2, 2e] # 双向图 21242
    
    # edge_index_direct = edge_index[(edge_index[0]-edge_index[1])>=0] # 转向单向图增减边 # 10621 
    edge_index_direct = edge_index[:,(edge_index[0]-edge_index[1])<=0] # 转向单向图增减边 # 10621  [2, 2e]-->[2, e]
    
    print('total edge num:',edge_index_direct.shape[1])
    # is_same_attr = (sens[edge_index[0]]==sens[edge_index[1]]) # 边连接是否相同的 True=1 False=0
    is_same_attr = (sens[edge_index_direct[0]]==sens[edge_index_direct[1]]) # 10621 边连接是否相同的 True=1 False=0
    
    # print('same attribute edge num:',is_same_attr.shape[1])
    is_same_indices = torch.nonzero(is_same_attr).squeeze() # 得到属性相同的边的序号 1x7686
    print('same attribute edge num:',is_same_indices.shape[0])
    
    is_same_rand_arg = torch.randperm(is_same_indices.size(0)) # 得到扰乱后的索引 
    keep_same_indices=None
    if keepratio<1.0:
        keep_same_indices = is_same_indices[is_same_rand_arg[:int(keepratio * is_same_indices.size(0))]] # 随机保留ratio的同属性边索引
    else:
        keep_same_indices = is_same_indices
    print('keep edge num:', len(keep_same_indices))
    # keep_direct_edge_index = None
    keep_direct_edge_index = edge_index_direct[:,keep_same_indices]
    # if not onlysame:
    #     edge_index_same = edge_index[:,(sens[edge_index_direct[0]]==sens[edge_index_direct[1]])]
    #     keep_direct_edge_index = torch.cat([keep_direct_edge_index,edge_index_same],dim=1)

    new_graph=dgl.to_bidirected(dgl.graph((keep_direct_edge_index[0],keep_direct_edge_index[1]), num_nodes=adj.shape[0]))
    print('number of edges:',new_graph.number_of_edges()//2)
    
    new_adj = new_graph.adj().to_dense()
    return new_adj

# 仅保留部分相异边
def get_only_hetero_edges(adj, sens, keepratio, seed=22):
    # set_seed(seed)
    edge_index = adj.nonzero().T # [2e, 2] --> [2, 2e] # 双向图 21242
    edge_index_direct = edge_index[:,(edge_index[0]-edge_index[1])<=0] # 转向单向图增减边 # 10621  [2, 2e]-->[2, e]
    
    print('total edge num:',edge_index_direct.shape[1])
    is_same_attr = (sens[edge_index_direct[0]]==sens[edge_index_direct[1]]) # 10621 边连接是否相同的 True=1 False=0
    
    is_diff_indices = torch.nonzero(torch.eq(is_same_attr, False)).squeeze() # 不同的边。 2935
    print('diff attribute edge num:',is_diff_indices.shape[0])
    
    is_diff_rand_arg = torch.randperm(is_diff_indices.size(0)) # 得到扰乱后的索引
    
    keep_diff_indices=None
    if keepratio<1.0:
        keep_diff_indices = is_diff_indices[is_diff_rand_arg[:int(keepratio * is_diff_indices.size(0))]] # 随机保留ratio的同属性边索引
    else:
        keep_diff_indices = is_diff_indices
    print('keep edge num:', len(keep_diff_indices))
    
    
    keep_direct_edge_index = edge_index_direct[:,keep_diff_indices]
    new_graph=dgl.to_bidirected(dgl.graph((keep_direct_edge_index[0],keep_direct_edge_index[1])))
    print('number of edges:',new_graph.number_of_edges()//2)
    new_adj = new_graph.adj().to_dense()
    return new_adj

# 将敏感属性相同的点形成完全子图
def get_same_sens_complete_graph(adj, sens, args):
    filepath = './adj_files/'+args.dataset+'_'+'same_sens_complete_adj.pt'
    # src_, dst_ = None, None
    new_adj = None
    try:
        # 尝试读取pt文件
        print('processed adj exists!')
        new_adj = torch.load(filepath)
    except FileNotFoundError:
        print('no exist!')
        homo_start = time.time()
        node_number = adj.shape[0]
        srcs, dsts = [],[] 
        for key in torch.unique(sens):
            node_indices = (sens==key).nonzero().squeeze()
            # node_indices[0].unsqueeze(0)
            repeat_num = len(node_indices)
            src = node_indices.repeat_interleave(repeat_num)
            dst = node_indices.repeat(repeat_num)
            print('key=',key, 'num:',len(src))
            srcs.append(src)
            dsts.append(dst)

        src_ = torch.cat(srcs)
        dst_ = torch.cat(dsts)
        # 如果文件不存在，保存一个空的Tensor对象到指定路径
        # eignvalue, eignvector = adjacency_positional_encoding(adj, args.pe_dim) 
        new_graph=dgl.remove_self_loop(dgl.graph((src_, dst_), num_nodes=node_number))
        # new_adj = new_graph.adj() # old dgl
        new_adj = new_graph.adj_external() # new dgl
        torch.save(new_adj, filepath)
        # lpe=eignvector
        time_cost = time.time()-homo_start
        print('create adj time is {:.3f}'.format(time_cost))

    return new_adj.to_dense()

# 根据id list生成src和dst
def construct_complete_graph_from_ids(ids): # C
    repeat_num = len(ids)
    src = ids.repeat_interleave(repeat_num)
    dst = ids.repeat(repeat_num)
    return src, dst

# 根据id list和子图数量，生成完全子图组成的全图的scr和dst
def construct_sub_complete_graph(id_list,subnum=1000,seed=20):
    torch.manual_seed(seed)
    shuffled_ids = id_list[torch.randperm(len(id_list))]
    sub_sequences = torch.split(shuffled_ids, subnum)
    srcs, dsts=[],[]
    sub_nums = []
    print('subgraph num:',len(sub_sequences))
    for idx, sub_seq in enumerate(sub_sequences):
        # print('subnode num:',len(sub_seq))
        sub_nums.append(len(sub_seq))
        src, dst= construct_complete_graph_from_ids(sub_seq)
        srcs.append(src)
        dsts.append(dst)
    src_ = torch.cat(srcs)
    dst_ = torch.cat(dsts)
    print('last_subnum:',sub_nums[-1])
    return src_,dst_

# 将敏感属性相同的点,切分后形成不同簇的完全子图，但是返回一个大adj
def get_same_sens_sub_complete_graph(adj, sens, subnum, args):
    filepath = './adj_files/'+args.dataset+'_'+'same_sens_sub_complete_'+str(subnum)+'adj.pt'
    # src_, dst_ = None, None
    new_adj = None
    try:
        # 尝试读取pt文件
        print('processed adj exists!')
        new_adj = torch.load(filepath)
    except FileNotFoundError:
        print('no exist!')
        homo_start = time.time()
        node_number = adj.shape[0]
        srcs, dsts = [],[] 
        for key in torch.unique(sens):
            node_indices = (sens==key).nonzero().squeeze()
            # node_indices[0].unsqueeze(0)
            src, dst = construct_sub_complete_graph(node_indices, subnum)
            # repeat_num = len(node_indices)
            # src = node_indices.repeat_interleave(repeat_num)
            # dst = node_indices.repeat(repeat_num)
            print('key=',key, 'num:',len(src))
            srcs.append(src)
            dsts.append(dst)

        src_ = torch.cat(srcs)
        dst_ = torch.cat(dsts)
        # 如果文件不存在，保存一个空的Tensor对象到指定路径
        # eignvalue, eignvector = adjacency_positional_encoding(adj, args.pe_dim) 
        new_graph=dgl.remove_self_loop(dgl.graph((src_, dst_), num_nodes=node_number)) # 
        new_adj = new_graph.adj()
        torch.save(new_adj, filepath)
        # lpe=eignvector
        time_cost = time.time()-homo_start
        print('create adj time is {:.3f}'.format(time_cost))

    return new_adj.to_dense()

# 将敏感属性不同的点形成完全子图
def get_diff_sens_complete_graph(adj, sens):
    node_number = adj.shape[0]
    srcs, dsts = [],[] 
    import itertools

    # a b c 属性的组合
    uni_sens = torch.unique(sens)

    for keys in list(itertools.permutations(uni_sens, 2)):
        node_indices0 = (sens==keys[0]).nonzero().squeeze()
        node_indices1 = (sens==keys[1]).nonzero().squeeze() # 可能不一样
        repeat_num0 = len(node_indices1)
        repeat_num1 = len(node_indices0)
        
        src = node_indices0.repeat_interleave(repeat_num0)
        dst = node_indices1.repeat(repeat_num1)
        print('keys=',keys[0],keys[1], 'num:',len(src))
        
        srcs.append(src)
        dsts.append(dst)
    src_ = torch.cat(srcs)
    dst_ = torch.cat(dsts)

    new_graph=dgl.remove_self_loop(dgl.graph((src_, dst_), num_nodes=node_number))
    print('number of edges:',new_graph.number_of_edges()//2)
    return new_graph.adj().to_dense()

# 保留现有敏感属性相异边和部分敏感属性相同的边
def get_all_hetero_and_partial_homo_edges(adj, sens, keepratio, seed=22):
# set_seed(seed)
    # keepratio = 0.5
    edge_index = adj.nonzero().T # [2e, 2] --> [2, 2e] # 双向图 21242

    # edge_index_direct = edge_index[(edge_index[0]-edge_index[1])>=0] # 转向单向图增减边 # 10621 
    edge_index_direct = edge_index[:,(edge_index[0]-edge_index[1])<=0] # 转向单向图增减边 # 10621  [2, 2e]-->[2, e]

    print('total edge num:',edge_index_direct.shape[1])
    # is_same_attr = (sens[edge_index[0]]==sens[edge_index[1]]) # 边连接是否相同的 True=1 False=0
    is_same_attr = (sens[edge_index_direct[0]]==sens[edge_index_direct[1]]) # 10621 边连接是否相同的 True=1 False=0
    is_same_indices = torch.nonzero(is_same_attr).squeeze() # 得到属性相同的边的序号 1x7686

    is_diff_attr = (~is_same_attr)
    is_diff_indices = torch.nonzero(is_diff_attr).squeeze()
    # is_diff_indices = torch.nonzero(torch.eq(is_same_attr, False)).squeeze()
    # print('same attribute edge num:',is_same_attr.shape[1])

    print('same attribute edge num:',is_same_indices.shape[0])
    print('diff attribute edge num:',is_diff_indices.shape[0])

    is_same_rand_arg = torch.randperm(is_same_indices.size(0)) # 得到扰乱后的索引 
    keep_same_indices=None
    if keepratio<1.0:
        keep_same_indices = is_same_indices[is_same_rand_arg[:int(keepratio * is_same_indices.size(0))]] # 随机保留ratio的同属性边索引
    else:
        keep_same_indices = is_same_indices
    print('keep same edge num:', len(keep_same_indices))
    # keep_direct_edge_index = None
    final_incices = torch.cat([keep_same_indices,is_diff_indices])
    # keep_direct_edge_index = edge_index_direct[:,keep_same_indices]
    keep_direct_edge_index = edge_index_direct[:,final_incices]

    new_graph=dgl.to_bidirected(dgl.graph((keep_direct_edge_index[0],keep_direct_edge_index[1]), num_nodes=adj.shape[0]))
    print('number of total edges:',new_graph.number_of_edges()//2)

    new_adj = new_graph.adj().to_dense()
    return new_adj

# 保留现有敏感属性相同边和部分敏感属性相异的边
def get_all_homo_and_partial_hetero_edges(adj, sens, keepratio, seed=22):
# set_seed(seed)
    # keepratio = 0.5
    edge_index = adj.nonzero().T # [2e, 2] --> [2, 2e] # 双向图 21242

    # edge_index_direct = edge_index[(edge_index[0]-edge_index[1])>=0] # 转向单向图增减边 # 10621 
    edge_index_direct = edge_index[:,(edge_index[0]-edge_index[1])<=0] # 转向单向图增减边 # 10621  [2, 2e]-->[2, e]

    print('total edge num:',edge_index_direct.shape[1])
    # is_same_attr = (sens[edge_index[0]]==sens[edge_index[1]]) # 边连接是否相同的 True=1 False=0
    is_same_attr = (sens[edge_index_direct[0]]==sens[edge_index_direct[1]]) # 10621 边连接是否相同的 True=1 False=0
    is_same_indices = torch.nonzero(is_same_attr).squeeze() # 得到属性相同的边的序号 1x7686

    is_diff_attr = (~is_same_attr)
    is_diff_indices = torch.nonzero(is_diff_attr).squeeze()
    # is_diff_indices = torch.nonzero(torch.eq(is_same_attr, False)).squeeze()
    # print('same attribute edge num:',is_same_attr.shape[1])

    print('same attribute edge num:',is_same_indices.shape[0])
    print('diff attribute edge num:',is_diff_indices.shape[0])

    is_diff_rand_arg = torch.randperm(is_diff_indices.size(0)) # 得到扰乱后的索引 
    keep_diff_indices=None
    if keepratio<1.0:
        keep_diff_indices = is_diff_indices[is_diff_rand_arg[:int(keepratio * is_diff_indices.size(0))]] # 随机保留ratio的同属性边索引
    else:
        keep_diff_indices = is_diff_indices
    print('keep diff edge num:', len(keep_diff_indices))
    # keep_direct_edge_index = None
    final_incices = torch.cat([keep_diff_indices,is_same_indices])
    # keep_direct_edge_index = edge_index_direct[:,keep_same_indices]
    keep_direct_edge_index = edge_index_direct[:,final_incices]

    new_graph=dgl.to_bidirected(dgl.graph((keep_direct_edge_index[0],keep_direct_edge_index[1]), num_nodes=adj.shape[0]))
    print('number of total edges:',new_graph.number_of_edges()//2)

    new_adj = new_graph.adj().to_dense()
    return new_adj

# 得到同样类别标签节点数+同敏感属性数配比:00 01 10 11 = 1:1:1:1
def get_same_label_and_sens_num_nodeid(labels,sens,choosed_labels=[0,1]):
    # unique_labels, label_counts = torch.unique(labels, return_counts=True) 
    # label_groups = [torch.where(labels == label)[0] for label in unique_labels[1:]] # node index list
    label_groups = [torch.where(labels == label)[0] for label in choosed_labels]
    selected_nodes = [] 
    min_num = 99999
    for group in label_groups: # 每类
        mask_A0 = torch.where(sens[group] == 0)[0]  # 属性 A 为0的节点索引
        mask_A1 = torch.where(sens[group] == 1)[0]  # 属性 A 为1的节点索引
        min_num_group = min(len(mask_A0),len(mask_A1))
        
        min_num = min(min_num,min_num_group)
    print('min_num_label_group:',min_num)
        
    for group in label_groups: # 标签类的最小值
        group_size = min_num  # 每个标签下的节点数
        mask_A0 = torch.where(sens == 0)[0] # sens的标签为0 idx是全局的
        mask_A1 = torch.where(sens == 1)[0] # sens的标签为1 idx是全局的

        selected_mask_A0 = np.intersect1d(mask_A0, group)
        selected_mask_A1 = np.intersect1d(mask_A1, group)

        selected_A0 = torch.from_numpy(np.random.choice(selected_mask_A0, size=group_size, replace=False))
        selected_A1 = torch.from_numpy(np.random.choice(selected_mask_A1, size=group_size, replace=False))
        # 将选择的节点加入最终的节点子集
        selected_nodes.append(torch.cat([selected_A0, selected_A1]))
    selected_nodes_ids = torch.cat(selected_nodes)
    return selected_nodes_ids


# 得到同样每个类别下敏感属性类型之比相同:0:1 = 1:1
def get_same_sens_num_nodeid(labels,sens,choosed_labels=[0,1]):
    # unique_labels, label_counts = torch.unique(labels, return_counts=True) 
    # label_groups = [torch.where(labels == label)[0] for label in unique_labels[1:]] # node index list
    label_groups = [torch.where(labels == label)[0] for label in choosed_labels]
    selected_nodes = [] 
        
    for group in label_groups: # 标签类的最小值
          # 每个标签下的节点数
        mask_A0 = torch.where(sens == 0)[0] # sens的标签为0 idx是全局的
        mask_A1 = torch.where(sens == 1)[0] # sens的标签为1 idx是全局的

        selected_mask_A0 = np.intersect1d(mask_A0, group)
        selected_mask_A1 = np.intersect1d(mask_A1, group)
        group_size = min(len(selected_mask_A0), len(selected_mask_A1))

        selected_A0 = torch.from_numpy(np.random.choice(selected_mask_A0, size=group_size, replace=False))
        selected_A1 = torch.from_numpy(np.random.choice(selected_mask_A1, size=group_size, replace=False))
        # 将选择的节点加入最终的节点子集
        selected_nodes.append(torch.cat([selected_A0, selected_A1]))
    selected_nodes_ids = torch.cat(selected_nodes)
    
    # return selected_nodes_ids 
    return torch.sort(selected_nodes_ids)

# 根据adj和节点id列表，返回子图的adj
def get_subgraph_adj_by_nodes(adj, nodes): 
    graph = dgl.from_scipy(adj) 
    subgraph = graph.subgraph(nodes) 
    return subgraph.adjacency_matrix_scipy(return_edge_ids=False).astype(float)