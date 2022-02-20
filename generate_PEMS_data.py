
import torch
from torch_geometric_temporal.dataset import METRLADatasetLoader, PemsBayDatasetLoader

import pandas as pd
import networkx as nx
import numpy as np


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA

def generate_PEMS_dataset_adj(dataset_name,device, data_used_subset= False, subset_percent = 0.2):
    file_path = "/home/mingxi/deep_learning_implementation/flow_data/data/"+ dataset_name +"/" + dataset_name
    if dataset_name == "PEMS04_speed":
        file_path = "/home/mingxi/deep_learning_implementation/flow_data/data/" + 'PEMS04' + "/" + 'PEMS04'
    if dataset_name == "PEMS08_speed":
        file_path = "/home/mingxi/deep_learning_implementation/flow_data/data/" + 'PEMS08' + "/" + 'PEMS08'
    if dataset_name == 'PEMSD7':
        # PEMSD7: Time(5/1/2012 - 6/30/2012, weekdays), Nodes(228)
        speed_matrix = pd.read_csv("~/deep_learning_implementation/flow_data/PEMSD7/V_228.csv")
        speed_matrix = speed_matrix.values
        adj_matrix = pd.read_csv("~/deep_learning_implementation/flow_data/PEMSD7/weighted_adj.csv", index_col=None,header=None)
        adj_matrix = adj_matrix.values

    if dataset_name != 'PEMSD7':
        ##dataset_name
        # PEMS04: Time(1/1/2018 - 2/28/2018), Nodes(307)
        # PEMS07: Time(5/1/2017 - 8/31/2017), Nodes(883)
        # PEMS08: Time(7/1/2016 - 8/31/2016), Nodes(170)

        speed_matrix = np.load( file_path + ".npz")['data']
        if dataset_name != "PEMS04_speed" or dataset_name != "PEMS08_speed":
            speed_matrix = speed_matrix[:, :, 0]
            print(speed_matrix)
        else:
            speed_matrix = speed_matrix[:, :, 1]
            speed_matrix = speed_matrix.clip(0,100)
        max_dp = speed_matrix.max()
        num_nodes = speed_matrix.shape[1]
        _ , adj_matrix = get_adjacency_matrix(file_path + ".csv", num_nodes)
    if data_used_subset == True:
        start_index = np.random.randint(0, (1 - subset_percent) * speed_matrix.shape[0])
        end_index = int(start_index + subset_percent * speed_matrix.shape[0])
        speed_matrix = speed_matrix.iloc[start_index, end_index]

    #speed_matrix = torch.tensor(speed_matrix)
    # mean, std = speed_matrix.mean(), speed_matrix.std()
    # speed_matrix = (speed_matrix - mean)/std
    max = np.max(speed_matrix)
    speed_matrix = torch.tensor(speed_matrix) / max
    X = torch.unsqueeze(speed_matrix, 2)

    num_nodes = X.shape[1]


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
        edge_attr.append(adj_matrix[edge[0]][edge[1]])

    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(1)

    return X, sp_L, edgelist, edge_index, edge_attr, num_nodes, max

# X, sp_L, edgelist, edge_index, edge_attr, num_nodes, mean, std = generate_flow_dataset_adj('PEMSD7', device)
#
# X_, sp_L_, edgelist_, edge_index_, edge_attr_, num_nodes_, mean_, std_ = generate_flow_dataset_adj('PEMS04', device)
# X_1, _, _, _, _, _, mean__, std__ = generate_flow_dataset_adj('PEMS07', device)
# X_2, _, _, _, _, _, _mean, _std= generate_flow_dataset_adj('PEMS08', device)
# print(X.shape)
# print(X_.shape)
# print(X_1.shape)
# print(X_2.shape)