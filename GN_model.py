import random

import torch.nn as nn
import torch
from utils_tool import graph_concat, copy_geometric_data
class GN(nn.Module):
    def __init__(self, edge_block, node_block, global_block, use_edge= True, use_node= True, use_global = True,
                 update_graph = False):
        super(GN, self).__init__()
        self.edge_block = edge_block
        self.node_block = node_block
        self.global_block = global_block
        self._use_edge = use_edge
        self._use_node = use_node
        self._use_global = use_global
        self._update_graph = update_graph


    def reset_parameters(self):
        for m in self.edge_block.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for m in self.node_block.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for m in self.global_block.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
    def forward(self,graph):
        if self._use_edge:
            graph = self.edge_block(graph)
        if self._use_node:
            graph = self.node_block(graph)
        if self._use_global:
            graph= self.global_block(graph)
        return graph

class P_GN(nn.Module):
    def __init__(self, edge_block_model, node_block_model, global_block_model,
                 use_edge_block=True, use_node_block= True,use_global_block = False):

        super(P_GN, self).__init__()

        # random coefficients
        self.a = (5 * random.random() - 2.5)
        self.b = (5 * random.random() - 2.5)

        self.eb_module = edge_block_model
        self.nb_module = node_block_model
        self.gb_module = global_block_model

        self._gn_module = GN(self.eb_module, self.nb_module, self.gb_module,
                             use_edge = use_edge_block, use_node= use_node_block, use_global= use_global_block, update_graph = False)
    def forward(self, input_graphs, laplacian, h_init, coeff = torch.tensor(0.1), pde = 'diff'):
        num_steps = len(input_graphs)

        output_tensor = []
        time_derivatives = []
        spatial_derivatives = []

        h_prev = None
        h_curr = h_init

        for input_graph in input_graphs:
            check = input_graph

            h_curr_concat = graph_concat(input_graph, h_curr, node_cat= True, edge_cat= True, global_cat= False)

            h_next = self._gn_module(h_curr_concat)

            if h_prev and pde == "wave":
                time_derivatives.append(h_next.x - 2*h_curr.x + h_prev.x)
            elif pde == "diff":
                time_derivatives.append(h_next.x - h_curr.x)
            elif h_prev and pde == "both":
                time_derivatives.append(h_next.x - 2*h_curr.x + h_prev.x + h_next.x - h_curr.x)
            elif h_prev and pde == "random":
                time_derivatives.append(h_next.x + self.a*h_curr.x + self.b*h_prev.x)


            spatial_derivatives.append(-coeff*laplacian.mm(h_curr.x))

            h_prev = h_curr
            h_curr = h_next

            output_tensor.append(copy_geometric_data(h_curr))

        return output_tensor, time_derivatives, spatial_derivatives



