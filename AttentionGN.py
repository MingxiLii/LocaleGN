import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter_min
from torch_geometric.data import Data

from GN_Model import P_GN
from blocks import EdgeBlock, NodeBlock, GlobalBlock

class TSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, device):
        N, T, C = query.shape

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, T, self.heads, self.head_dim)  # embed_size维拆成 heads×head_dim
        keys = keys.reshape(N, T, self.heads, self.head_dim)
        query = query.reshape(N, T, self.heads, self.head_dim)

        values = self.values(values).to(device) # (N, T, heads, head_dim)
        keys = self.keys(keys).to(device)  # (N, T, heads, head_dim)
        queries = self.queries(query).to(device)  # (N, T, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        energy = torch.einsum("nqhd,nkhd->nqkh", [queries, keys]).to(device)  # 时间self-attention
        # queries shape: (N, T, heads, heads_dim),
        # keys shape: (N, T, heads, heads_dim)
        # energy: (N, T, T, heads)

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2).to(device) # 在K维做softmax，和为1
        # attention shape: (N, query_len, key_len, heads)

        out = torch.einsum("nqkh,nkhd->nqhd", [attention, values]).reshape(
            N, T, self.heads * self.head_dim
        ).to(device)
        # attention shape: (N, T, T, heads)
        # values shape: (N, T, heads, heads_dim)
        # out after matrix multiply: (N, T, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out).to(device)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, T, embed_size)

        return out

class attention_gn(nn.Module):
    def __init__(self, in_channel, node_attr_size, edge_num_embeddings, out_size,  att_layer, device, edge_hidden_size , node_hidden_size ,
                 global_hidden_size, heads):
        super(attention_gn, self).__init__()
        self.in_channel = in_channel
        self.input_size = node_attr_size
        self.edge_h_dim = edge_hidden_size
        self.node_h_dim = node_hidden_size
        self.node_half_h_dim = int(self.node_h_dim) / 2
        self.global_h_dim = global_hidden_size
        self.global_half_h_dim = int(self.global_h_dim) / 2
        self.device = device

        # Encoder
        self.edge_enc = nn.Sequential(nn.Linear(1, self.edge_h_dim), nn.ReLU())
        self.node_enc = nn.Sequential(nn.Linear(self.input_size, self.node_h_dim), nn.ReLU())
        self.node_enc_for_att = nn.Sequential(nn.Linear(self.in_channel, self.node_h_dim), nn.ReLU())

        # self-attention
        self.attention = TSelfAttention(node_hidden_size, heads)

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
                                     edge_reducer = scatter_mean,
                                     node_reducer = scatter_mean,
                                     custom_func=self.gb_custom_func,
                                     device=device)
        self.gn = P_GN(self.eb_module,
                        self.nb_module,
                        self.gb_module,
                        use_edge_block=True,
                        use_node_block=True,
                        use_global_block=True)
        ##Decoder
        self.node_dec = nn.Sequential(nn.Linear(self.node_h_dim, self.node_h_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.node_h_dim, out_size)
                                      )
        self.feed_forward_att = nn.Sequential(nn.Linear(self.node_h_dim * att_layer, self.node_h_dim * att_layer),
                                              nn.ReLU(),
                                              nn.Linear(self.node_h_dim * att_layer, in_channel)
                                              )

        self.node_dec_for_input = nn.Sequential(nn.Linear(self.node_h_dim, self.node_h_dim),
                                                nn.ReLU(),
                                                nn.Linear(self.node_h_dim, self.input_size))

    def forward(self, data, sp_L, t, num_processing_steps, coeff, pde='diff'):
        from utils_tool import decompose_graph

        input_graphs = []
        node_attrs = []
        edge_indexs = []
        edge_attrs = []
        nodes_num = sp_L.shape[0]
        self_attention_nodes = []

        for step_t in range(num_processing_steps):
            # node_attr (nodes_num,hidden_num)
            node_attr, edge_index, edge_attr, global_attr = decompose_graph(data[step_t])
            node_attrs.append(node_attr)
            edge_indexs.append(edge_index)
            edge_attrs.append(edge_attr)

            #### Encoder for attention
            encoded_node_for_att = self.node_enc_for_att(node_attr.unsqueeze(2))

            ### Self-Attention
            node_attrs_attention = self.attention(encoded_node_for_att, encoded_node_for_att, encoded_node_for_att, self.device) #(N,T,embed)
            node_attrs_attention = self.feed_forward_att(node_attrs_attention) #(N,T,C)

            self_attention_nodes.append(node_attrs_attention)
            ### residual_connection
            node_attr = node_attrs_attention.squeeze(2) + node_attr


            #### Input for GN
            encoded_edge = self.edge_enc(edge_attr)

            encoded_node = self.node_enc(node_attr)


            input_graph = Data(x= encoded_node, edge_index=edge_index, edge_attr=encoded_edge)
            if step_t == 0:
                input_graph.global_attr = global_attr
            input_graphs.append(input_graph)

        init_graph = input_graphs[0]
        # h_init is zero tensor
        h_init = Data(x=torch.zeros(init_graph.x.size(), dtype=torch.float32, device=self.device),
                          edge_index=init_graph.edge_index,
                          edge_attr=torch.zeros(init_graph.edge_attr.size(), dtype=torch.float32, device=self.device))
        h_init.global_attr = init_graph.global_attr

        ### GN
        output_graphs, time_derivatives, spatial_derivatives = self.gn(input_graphs, sp_L, h_init, coeff, pde)

        output_nodes, pred_inputs = [], []
        for output_graph in output_graphs:
            output_nodes.append(self.node_dec(output_graph.x))
            pred_inputs.append(self.node_dec_for_input(output_graph.x))

        return output_nodes, time_derivatives, spatial_derivatives
