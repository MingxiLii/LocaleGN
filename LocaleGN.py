import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter_min
from torch_geometric.data import Data

from GN_Model import P_GN
from blocks import EdgeBlock, NodeBlock, GlobalBlock

class Net(nn.Module):

    def __init__(self, node_attr_size, edge_num_embeddings, out_size,  gru_layer, device, edge_hidden_size = 64, node_hidden_size = 64,
                 global_hidden_size= 64):
        super(Net, self).__init__()

        self.input_size = node_attr_size
        self.edge_h_dim = edge_hidden_size
        self.node_h_dim = node_hidden_size
        self.node_half_h_dim = int(self.node_h_dim)/2
        self.global_h_dim = global_hidden_size
        self.global_half_h_dim = int(self.global_h_dim) / 2
        self.device = device

        self.gru_for_node = nn.GRU(node_attr_size, node_hidden_size, gru_layer)
        self.l_out = nn.Linear(in_features=node_hidden_size,
                               out_features=out_size)

        ## Encoder

        self.edge_enc = nn.Sequential(nn.Linear(1 , self.edge_h_dim), nn.ReLU())
        self.node_enc = nn.Sequential(nn.Linear(self.input_size, self.node_h_dim), nn.ReLU())


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
        self.eb_module = EdgeBlock((self.edge_h_dim+self.node_h_dim*2)*2+self.global_h_dim,
                                           self.edge_h_dim,
                                           use_edges=True,
                                           use_sender_nodes=True,
                                           use_receiver_nodes=True,
                                           use_globals=True,
                                           custom_func=self.eb_custom_func)

        self.nb_module = NodeBlock(self.node_h_dim*2+self.edge_h_dim*2+self.global_h_dim,
                                   self.node_h_dim,
                                   use_nodes=True,
                                   use_sent_edges=True,
                                   use_received_edges=True,
                                   use_globals=True,
                                   sent_edges_reducer=scatter_add,
                                   received_edges_reducer=scatter_add,
                                   custom_func=self.nb_custom_func)

        self.gb_module = GlobalBlock(self.node_h_dim+self.edge_h_dim+self.global_h_dim,
                                     self.global_h_dim,
                                     edge_reducer=scatter_mean,
                                     node_reducer=scatter_mean,
                                     custom_func=self.gb_custom_func,
                                     device=device)


        self.eb_custom_func_2 = nn.Sequential(nn.Linear((self.edge_h_dim + self.node_h_dim * 2) * 4 + self.global_h_dim * 2,
                                                      self.edge_h_dim*2),
                                            nn.ReLU(),
                                            )
        self.nb_custom_func_2 = nn.Sequential(nn.Linear(self.node_h_dim * 4 + self.edge_h_dim * 4 + self.global_h_dim * 2,
                                                      self.node_h_dim * 2),
                                            nn.ReLU(),
                                            )
        self.gb_custom_func_2 = nn.Sequential(nn.Linear(self.node_h_dim*2 + self.edge_h_dim*2 + self.global_h_dim*2,
                                                      self.global_h_dim*2),
                                            nn.ReLU(),
                                            )
        self.nb_module_2 = NodeBlock(self.node_h_dim * 4 + self.edge_h_dim * 4 + self.global_h_dim*2,
                                     self.node_h_dim*2,
                                     use_nodes=True,
                                     use_sent_edges=True,
                                     use_received_edges=True,
                                     use_globals=True,
                                     sent_edges_reducer=scatter_add,
                                     received_edges_reducer=scatter_add,
                                     custom_func=self.nb_custom_func_2)
        self.eb_module_2 = EdgeBlock((self.edge_h_dim + self.node_h_dim * 2) * 4 + self.global_h_dim*2,
                                       self.edge_h_dim*2,
                                       use_edges=True,
                                       use_sender_nodes=True,
                                       use_receiver_nodes=True,
                                       use_globals=True,
                                       custom_func=self.eb_custom_func_2)
        self.gb_module_2 = GlobalBlock(self.node_h_dim*2 +self.edge_h_dim*2+self.global_h_dim*2,
                                     self.global_h_dim*2,
                                     edge_reducer=scatter_mean,
                                     node_reducer=scatter_mean,
                                     custom_func=self.gb_custom_func_2,
                                     device=device)
        self.gn1 = P_GN(self.eb_module,
                       self.nb_module,
                       self.gb_module,
                       use_edge_block=True,
                       use_node_block=True,
                       use_global_block=True)

        self.gn2 = P_GN(self.eb_module_2,
                       self.nb_module_2,
                       self.gb_module_2,
                       use_edge_block=True,
                       use_node_block=True,
                       use_global_block=True)
        ##Decoder
        self.node_dec = nn.Sequential(nn.Linear(self.node_h_dim  , self.node_h_dim ),
                                      nn.ReLU(),
                                      nn.Linear(self.node_h_dim  , out_size)
                                      )
        self.node_dec_for_gru = nn.Sequential(nn.Linear(self.node_h_dim*gru_layer, self.node_h_dim*gru_layer),
                                      nn.ReLU(),
                                      nn.Linear(self.node_h_dim*gru_layer, out_size)
                                      )

        self.node_dec_for_input = nn.Sequential(nn.Linear(self.node_h_dim  , self.node_h_dim ),
                                                nn.ReLU(),
                                                nn.Linear(self.node_h_dim , self.input_size ))

    def forward(self, data, sp_L, t, num_processing_steps, coeff, pde='diff', skip = True):
        from utils_tool import decompose_graph

        input_graphs = []
        node_attrs = []
        edge_indexs = []
        edge_attrs = []
        nodes_num = sp_L.shape[0]

        for step_t in range(num_processing_steps):
            # node_attr (nodes_num,hidden_num)
            node_attr, edge_index, edge_attr, global_attr = decompose_graph(data[step_t])
            node_attrs.append(node_attr)
            edge_indexs.append(edge_index)
            edge_attrs.append(edge_attr)

            #### Encoder
            encoded_edge = self.edge_enc(edge_attr)

            encoded_node = self.node_enc(node_attr)


            #### GN
            input_graph = Data(x=encoded_node, edge_index=edge_index, edge_attr=encoded_edge)
            if step_t == 0:
                input_graph.global_attr = global_attr

            input_graphs.append(input_graph)

        ## gru for node
        nodes_attr_after_gru = []

        nodes_attr_for_gru = torch.cat(node_attrs, dim=1).unsqueeze(2)
        for node_attr_for_gru in nodes_attr_for_gru:
            node_attr_for_gru_ = node_attr_for_gru.unsqueeze(2)
            output, hn = self.gru_for_node(node_attr_for_gru_)
            out = self.l_out(hn).squeeze(1)
            nodes_attr_after_gru.append(out)
        nodes_attr_based_on_his= torch.cat(nodes_attr_after_gru, dim=0)




        init_graph = input_graphs[0]
        # h_init is zero tensor
        h_init = Data(x=torch.zeros(init_graph.x.size(), dtype=torch.float32, device=self.device),
                      edge_index=init_graph.edge_index,
                      edge_attr=torch.zeros(init_graph.edge_attr.size(), dtype=torch.float32, device=self.device))
        h_init.global_attr = init_graph.global_attr

        output_graphs_1, time_derivatives, spatial_derivatives = self.gn1(input_graphs, sp_L, h_init, coeff, pde)

        output_nodes, pred_inputs = [], []
        for output_graph in output_graphs_1:
            output_nodes.append(self.node_dec(output_graph.x).unsqueeze(0))
            pred_inputs.append(self.node_dec_for_input(output_graph.x))

        pred_result = output_nodes[num_processing_steps-1] + nodes_attr_based_on_his.unsqueeze(0)
        if skip == True:
            pred_result =  nodes_attr_based_on_his.unsqueeze(0)

        return pred_result, time_derivatives, spatial_derivatives