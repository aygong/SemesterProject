import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from node_encoder import NodeEncoder
import time

"""
    Graph Transformer with edge features
    
"""
from graph_transformer_edge_layer import GraphTransformerLayer

class GraphTransformerNet(nn.Module):
    def __init__(self, 
                 device,
                 #num_users,
                 #num_vehicles,
                 #target_seq_len,
                 num_nodes,
                 num_node_feat,
                 num_edge_feat,
                 d_model=512,
                 num_layers=6,
                 num_heads=8,
                 d_k=64,
                 d_v=64,
                 #d_ff=2048,
                 d_ff=1024,
                 dropout=0.1,
                 #in_feat_dropout=0.0,
                 layer_norm=False,
                 batch_norm=True,
                 lap_pos_enc=False,
                 residual=True
                 ):
        
        super().__init__()

        self.num_nodes = num_nodes
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.device = device
        self.lap_pos_enc = lap_pos_enc
        self.residual = residual
        self.d_model = d_model
        
        if self.lap_pos_enc:
            #pos_enc_dim = net_params['pos_enc_dim']
            #self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
            raise NotImplementedError()
        
        
        self.embedding_h = nn.Linear(num_node_feat, d_model)

        self.embedding_e = nn.Linear(num_edge_feat, d_model)
        
        #self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.node_encoder = NodeEncoder(
            device,
            input_seq_len=10,
            d_model=d_model,
            num_layers=2,
            num_heads=num_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=2*d_model,
            dropout=dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(d_model, d_model, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(num_layers) ]) 
        #self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.last_policy_transform = GraphTransformerLayer(d_model, d_model, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual)
        #self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem 
        self.MLP_policy_layer = nn.Sequential(
            nn.Linear(2 * d_model, d_ff), 
            nn.ReLU(),
            nn.Linear(d_ff, 1) 
        )   

        #self.MLP_value_layer = nn.Sequential(
        #    nn.Linear(num_nodes * d_model, d_last_ff), 
        #    nn.ReLU(),
        #    nn.Linear(d_last_ff, 1) 
        #)

        self.last_value_transform = GraphTransformerLayer(d_model, d_model, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual)
        self.MLP_value_layer = nn.Sequential(
            nn.Linear(d_model+1, d_ff), 
            nn.ReLU(),
            nn.Linear(d_ff, 1) 
        )
        
    def forward(self, g, h, e, vehicle_node_id, num_nodes, h_lap_pos_enc=None, masking=False):
        # input embedding
        #h = self.node_encoder(h)
        h = self.embedding_h(h)
        #h = self.in_feat_dropout(h)
        if self.lap_pos_enc:

            raise NotImplementedError()
        
        e = self.embedding_e(e) 
        
        # transformer layers
        for conv in self.layers:
            h, e = conv(g, h, e)
        
        h_policy, _ = self.last_policy_transform(g,h,e)
        
        batch_size = len(vehicle_node_id)
        vehicle_node_id = torch.tensor([i*num_nodes + k for i,k in enumerate(vehicle_node_id)], device=self.device)
        ks = vehicle_node_id.repeat_interleave(num_nodes).long()
        pairs = torch.cat([h_policy[ks], h_policy], dim=1)
        policy = torch.squeeze(self.MLP_policy_layer(pairs))

        if masking:
            neighbors = [g.successors(k) for k in vehicle_node_id]
            mask = torch.tensor([False if i in neighbors[i//num_nodes] else True for i in range(batch_size * num_nodes)], device=self.device)
            for i in range(batch_size):
                # mask value nodes
                mask[(i+1)*num_nodes-1] = True
            policy = policy.masked_fill(mask, -1e6)
        
        policy = torch.reshape(policy, (batch_size, num_nodes))

        # also return value
        h_value, _ = self.last_value_transform(g,h,e)
        value_nodes = torch.zeros((batch_size, self.d_model+1))
        i=0
        for v_node in range(num_nodes-1, batch_size*num_nodes, num_nodes):
            n_neighbors = len(g.successors(v_node))
            value_nodes[i,:-1] = h_value[v_node]
            value_nodes[i,-1] = n_neighbors
            i+=1
        if i != batch_size:
            raise RuntimeError(f'i should be equal to batch_size: {i}, {batch_size}')
        #value = self.MLP_value_layer(torch.reshape(h, (batch_size, self.d_model * self.num_nodes)))
        value = self.MLP_value_layer(value_nodes)

        return policy, torch.squeeze(value)
        