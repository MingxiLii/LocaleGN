import torch
import numpy as np
from torch_geometric.data import Data
from torch_scatter import scatter_add
import pandas as pd
import networkx as nx
import geopy.distance


def generate_dataset_adj(dataset_name,device, data_used_subset= False, subset_percent = 0.2):
    if dataset_name == 'LA':
        speed_matrix = pd.read_csv("~/deep_learning_implementation/Data/los_data/los_speed.csv")
        adj_matrix = pd.read_csv("~/deep_learning_implementation/Data/los_data/los_adj.csv", index_col=None,header=None)
        coordinates = pd.read_csv("~/deep_learning_implementation/Data/data_5_min/LA/los_locations.csv", header=None)
    if dataset_name == 'HK':
        speed_matrix = pd.read_csv("~/deep_learning_implementation/Data/data_5_min/one_month_data/data_hk_March_April.csv")
        adj_matrix = pd.read_csv("~/deep_learning_implementation/Data/hk_data/hk_adj.csv", index_col=None,header=None)
        coordinates = pd.read_csv("~/deep_learning_implementation/Data/data_5_min/HK/HKcoords.csv", header=None)
    if dataset_name == 'ST':
        speed_matrix = pd.read_csv("~/deep_learning_implementation/Data/data_5_min/one_month_data/data_st_March_April.csv")
        adj_matrix = pd.read_csv("~/deep_learning_implementation/Data/st_data/seattle_adj.csv", index_col=None, header=None)
        coordinates = pd.read_csv("~/deep_learning_implementation/Data/data_5_min/ST/st_locations.csv")
    if data_used_subset == True:
        start_index = np.random.randint(0, (1 - subset_percent) * speed_matrix.shape[0])
        end_index = int(start_index + subset_percent * speed_matrix.shape[0])
        speed_matrix = speed_matrix.iloc[start_index, end_index]

    speed_matrix = speed_matrix.clip(0, 100)
    speed_matrix = torch.tensor(speed_matrix.values)
    # mean, std = speed_matrix.mean(), speed_matrix.std()
    # speed_matrix = (speed_matrix - mean) / std
    max_value = speed_matrix.max()
    speed_matrix = speed_matrix / max_value
    X = torch.unsqueeze(speed_matrix, 2)

    num_nodes = X.shape[1]

    adj_matrix = adj_matrix.values
    adj = nx.convert_matrix.from_numpy_matrix(adj_matrix,parallel_edges=True,create_using=nx.MultiGraph)
    sp_L = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(adj)
    rows, cols = sp_L .nonzero()
    data = sp_L[rows, cols]
    indicies = []
    indicies.append(rows)
    indicies.append(cols)
    indicies= torch.tensor(indicies)
    data = torch.tensor(data,  dtype=torch.float32).squeeze(0)
    sp_L = torch.sparse_coo_tensor(indicies, data).to(device)

    edgelist = [(u, v) for (u, v) in adj.edges()]
    edge_index = torch.tensor(edgelist)
    edge_index = edge_index.transpose(0,1)


    edge_attr = []
    for edge in edgelist:
        origin_= coordinates.iloc[edge[0]].values
        origin = (origin_[1], origin_[0])
        des_ = coordinates.iloc[edge[1]].values
        des = (des_[1], des_[0])
        edge_attr.append(geopy.distance.geodesic(origin, des).km)

    edge_attr = [float(i)/max(edge_attr) for i in edge_attr]
    edge_attr = torch.tensor(edge_attr).unsqueeze(1)

    return X, sp_L, edgelist, edge_index, edge_attr, num_nodes, max_value

def loss_multi_step(output_tensors, X,t, output_size, num_nodes, his_added, device):
    #output(nodes_num, output_size)
    if his_added == True:
        multi_step_loss = []
        for step in range(output_size):
            loss_sup_seq = [torch.sum((output[:, step] - torch.tensor(X[t + step_t + step +1 , :, 0:], dtype=torch.float32,
                                                 device=device)) ** 2).item()
                for step_t, output in enumerate(output_tensors)]
            multi_step_loss.append(np.mean(loss_sup_seq)/num_nodes)
        print(multi_step_loss[0], multi_step_loss[1], multi_step_loss[2])
        return multi_step_loss

def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    source_target_distance = {}
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        #dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
        source_target_distance[(sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]])]= row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx


class Evaluation(object):
    def __init__(self):
        pass

    @staticmethod
    def mae_(target, output):
        return np.mean(np.abs(target - output))

    @staticmethod
    def mape_(target, output):
        return np.mean(np.abs((target - output) / (target+1)))*100

    @staticmethod
    def rmse_(target, output):
        return np.sqrt(np.mean(np.power(target - output, 2)))

    @staticmethod
    def total(target, output):
        mae = Evaluation.mae_(target, output)
        mape = Evaluation.mape_(target, output)
        rmse = Evaluation.rmse_(target, output)

        return mae, mape, rmse

def decompose_graph(graph):
    # graph: torch_geometric.data.data.Data
    # TODO: make it more robust
    x, edge_index, edge_attr, global_attr = None, None, None, None
    for key in graph.keys:
        if key=="x":
            x = graph.x
        elif key=="edge_index":
            edge_index = graph.edge_index
        elif key=="edge_attr":
            edge_attr = graph.edge_attr
        elif key=="global_attr":
            global_attr = graph.global_attr
        else:
            pass
    return (x, edge_index, edge_attr, global_attr)

def graph_concat(graph1, graph2,
                 node_cat=True, edge_cat=True, global_cat=False):
    """
    Args:
        graph1: torch_geometric.data.data.Data
        graph2: torch_geometric.data.data.Data
        node_cat: True if concat node_attr
        edge_cat: True if concat edge_attr
        global_cat: True if concat global_attr
    Return:
        new graph: concat(graph1, graph2)
    """
    # graph2 attr is used for attr that is not concated.
    _x = graph2.x
    _edge_attr = graph2.edge_attr
    _global_attr = graph2.global_attr
    _edge_index = graph2.edge_index

    if node_cat:
        try:
            _x = torch.cat([graph1.x, graph2.x], dim=-1)
        except:
            raise ValueError("Both graph1 and graph2 should have 'x' key.")

    if edge_cat:
        try:
            _edge_attr = torch.cat([graph1.edge_attr, graph2.edge_attr], dim=-1)
        except:
            raise ValueError("Both graph1 and graph2 should have 'edge_attr' key.")

    if global_cat:
        try:
            _global_attr = torch.cat([graph1.global_attr, graph2.global_attr], dim=-1)
        except:
            raise ValueError("Both graph1 and graph2 should have 'global_attr' key.")

    ret = Data(x=_x, edge_attr=_edge_attr, edge_index=_edge_index)
    ret.global_attr = _global_attr

    return ret


def copy_geometric_data(graph):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    node_attr, edge_index, edge_attr, global_attr = decompose_graph(graph)

    ret = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
    ret.global_attr = global_attr

    return ret

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def get_adj(edge_index, weight=None):
    """return adjacency matrix"""
    if not weight:
        weight = torch.ones(edge_index.shape[1])

    row, col = edge_index
    return torch.sparse.FloatTensor(edge_index, weight)


def get_laplacian(edge_index, weight=None, type='norm', sparse=True):
    """return Laplacian (sparse tensor)
    type: 'comb' or 'norm' for combinatorial or normalized one.
    """
    adj = get_adj(edge_index, weight=weight)  # torch.sparse.FloatTensor
    num_nodes = adj.shape[1]
    senders, receivers = edge_index
    num_edges = edge_index.shape[1]

    deg = scatter_add(torch.ones(num_edges), senders)
    sp_deg = torch.sparse.FloatTensor(torch.tensor([range(num_nodes), range(num_nodes)]), deg)
    Laplacian = sp_deg - adj  # L = D-A

    deg = deg.pow(-0.5)
    deg[deg == float('inf')] = 0
    sp_deg = torch.sparse.FloatTensor(torch.tensor([range(num_nodes), range(num_nodes)]), deg)
    Laplacian_norm = sp_deg.mm(Laplacian.mm(sp_deg.to_dense()))  # Lsym = (D^-1/2)L(D^-1/2)

    if type == "comb":
        return Laplacian if sparse else Laplacian.to_dense()
    elif type == "norm":
        return to_sparse(Laplacian_norm) if sparse else Laplacian_norm
    else:
        raise ValueError("type should be one of ['comb', 'norm']")

