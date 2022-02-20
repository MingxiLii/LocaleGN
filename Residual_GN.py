from GN_Model import P_GN
from blocks import EdgeBlock, NodeBlock, GlobalBlock
from utils_tool import decompose_graph

from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter_min
from torch_geometric.data import Data

import torch.nn as nn
import torch

def generate_GN_input(node_attrs_pre, output_graph_pre, device):
    input_graphs_next = []
    node_attrs_next = []
    for step_t, node_attr_original in enumerate(node_attrs_pre):
        # node_attr (nodes_num,hidden_num)
        node_attr, edge_index, edge_attr, global_attr = decompose_graph(output_graph_pre[step_t])
        node_attr_res = node_attr + node_attr_original
        node_attrs_next.append(node_attr_res)


        input_graph = Data(x=node_attr_res, edge_index=edge_index, edge_attr=edge_attr)
        if step_t == 0:
            input_graph.global_attr = global_attr

        input_graphs_next.append(input_graph)

    init_graph_next = input_graphs_next[0]
    # h_init is zero tensor
    h_init_next = Data(x=torch.zeros(init_graph_next.x.size(), dtype=torch.float32, device=device),
                    edge_index=init_graph_next.edge_index,
                    edge_attr=torch.zeros(init_graph_next.edge_attr.size(), dtype=torch.float32, device=device))

    h_init_next.global_attr = init_graph_next.global_attr

    return h_init_next, input_graphs_next, node_attrs_next



class Res_GN(nn.Module):
    def __init__(self, node_attr_size, edge_num_embeddings, out_size,num_layers, device, edge_hidden_size , node_hidden_size ,
                 global_hidden_size, add_residual =True):
        super(Res_GN, self).__init__()

        self.input_size = node_attr_size
        self.edge_h_dim = edge_hidden_size
        self.node_h_dim = node_hidden_size
        self.node_half_h_dim = int(self.node_h_dim) / 2
        self.global_h_dim = global_hidden_size
        self.global_half_h_dim = int(self.global_h_dim) / 2
        self.num_layers = num_layers
        self.device = device
        self.add_residual = add_residual

        ## Encoder

        self.edge_enc = nn.Sequential(nn.Linear(1, self.edge_h_dim), nn.ReLU())
        self.node_enc = nn.Sequential(nn.Linear(self.input_size, self.node_h_dim), nn.ReLU())

        ## GN

        self.eb_custom_func = nn.Sequential(nn.Linear((self.edge_h_dim + self.node_h_dim * 2) * 2 + self.global_h_dim,
                                                      self.edge_h_dim),
                                            nn.ReLU(),
                                            )
        self.nb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim * 2 + self.edge_h_dim * 2 + self.global_h_dim,
                                                      self.node_h_dim),
                                            nn.ReLU(),
                                            )
        self.gb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                                      self.global_h_dim),
                                            nn.ReLU(),
                                            )
        self.eb_module = EdgeBlock((self.edge_h_dim + self.node_h_dim * 2) * 2 + self.global_h_dim,
                                   self.edge_h_dim,
                                   use_edges=True,
                                   use_sender_nodes=True,
                                   use_receiver_nodes=True,
                                   use_globals=True,
                                   custom_func=self.eb_custom_func)

        self.nb_module = NodeBlock(self.node_h_dim * 2 + self.edge_h_dim * 2 + self.global_h_dim,
                                   self.node_h_dim,
                                   use_nodes=True,
                                   use_sent_edges=True,
                                   use_received_edges=True,
                                   use_globals=True,
                                   sent_edges_reducer=scatter_add,
                                   received_edges_reducer=scatter_add,
                                   custom_func=self.nb_custom_func)

        self.gb_module = GlobalBlock(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                     self.global_h_dim,
                                     edge_reducer=scatter_mean,
                                     node_reducer=scatter_mean,
                                     custom_func=self.gb_custom_func,
                                     device=device)

        self.gn = P_GN(self.eb_module,
                        self.nb_module,
                        self.gb_module,
                        use_edge_block=True,
                        use_node_block=True,
                        use_global_block=True)


        ##Decoder
        self.node_dec = nn.Sequential(nn.Linear(self.node_h_dim , self.node_h_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.node_h_dim , out_size)
                                      )

        self.node_dec_for_input = nn.Sequential(nn.Linear(self.node_h_dim , self.node_h_dim ),
                                                nn.ReLU(),
                                                nn.Linear(self.node_h_dim, self.input_size ))

    def forward(self, data, sp_L, t, num_processing_steps, coeff, pde='diff'):
        from utils_tool import decompose_graph

        input_graphs_1 = []
        node_attrs_1 = []
        edge_indexs_1 = []
        edge_attrs_1 = []

        for step_t in range(num_processing_steps):
            # node_attr (nodes_num,hidden_num)
            node_attr, edge_index, edge_attr, global_attr = decompose_graph(data[step_t])
            # edge_indexs_1.append(edge_index)
            # edge_attrs_1.append(edge_attr)

            #### Encoder(edge, node)
            encoded_edge = self.edge_enc(edge_attr)  # Use embedding

            encoded_node = self.node_enc(node_attr)
            node_attrs_1.append(encoded_node)

            #### GN-1
            input_graph = Data(x=encoded_node, edge_index=edge_index, edge_attr=encoded_edge)
            if step_t == 0:
                input_graph.global_attr = global_attr

            input_graphs_1.append(input_graph)

        init_graph = input_graphs_1[0]
        # h_init is zero tensor
        h_init_1 = Data(x=torch.zeros(init_graph.x.size(), dtype=torch.float32, device=self.device),
                        edge_index=init_graph.edge_index,
                        edge_attr=torch.zeros(init_graph.edge_attr.size(), dtype=torch.float32, device=self.device))
        h_init_1.global_attr = init_graph.global_attr

        output_graphs_1, time_derivatives, spatial_derivatives = self.gn(input_graphs_1, sp_L, h_init_1, coeff, pde)

        ##muti-gn
        gn_graphs = []
        node_attrs_pre = node_attrs_1
        output_graphs_pre = output_graphs_1
        if self.num_layers != 1:
            for layer in range(self.num_layers - 1):
                h_init_next, input_graphs_next, node_attrs_next = generate_GN_input(node_attrs_pre, output_graphs_pre, self.device)

                output_graphs_pre, time_derivatives, spatial_derivatives = self.gn(input_graphs_next, sp_L, h_init_next, coeff, pde)
                gn_graphs.append(output_graphs_pre)

                node_attrs_pre = node_attrs_next

        output_nodes, pred_inputs = [], []
        if self.add_residual == True:
            ### residual connection for decoder
            input_graphs_next = []
            node_attrs_next = []
            for step_t, node_attr_original in enumerate(node_attrs_pre):
                # node_attr (nodes_num,hidden_num)
                node_attr, edge_index, edge_attr, global_attr = decompose_graph(output_graphs_pre[step_t])
                node_attr_res = node_attr + node_attr_original
                node_attrs_next.append(node_attr_res)
                input_graph = Data(x=node_attr_res, edge_index=edge_index, edge_attr=edge_attr)
                if step_t == 0:
                    input_graph.global_attr = global_attr
                input_graphs_next.append(input_graph)

            #### Decoder
            for output_graph in input_graphs_next:
                output_nodes.append(self.node_dec(output_graph.x).unsqueeze(0))
                pred_inputs.append(self.node_dec_for_input(output_graph.x))

        if self.add_residual == False:
            if self.num_layers==1:
                gn_graphs = output_graphs_pre
            for gn_graph in gn_graphs:
                output_nodes.append(self.node_dec(gn_graph.x).unsqueeze(0))
                pred_inputs.append(self.node_dec_for_input(gn_graph.x))

        return output_nodes, time_derivatives, spatial_derivatives









