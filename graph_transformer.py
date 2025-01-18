import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from node_encoder import NodeEncoder

"""
    Graph Transformer with edge features
    
"""
from graph_transformer_edge_layer import GraphTransformerLayer

class GraphTransformerNet(nn.Module):
    def __init__(self, 
                 device,
                 num_nodes,
                 num_node_feat,
                 num_edge_feat,
                 d_model=512,
                 num_layers=6,
                 num_heads=8,
                 d_k=64,
                 d_v=64,
                 d_ff=1024,
                 dropout=0.1,
                 layer_norm=False,
                 batch_norm=True,
                 lap_pos_enc=False,
                 residual=True,
                 pe_dim = 0
                 ):
        
        super().__init__()

        self.num_nodes = num_nodes
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.device = device
        self.lap_pos_enc = lap_pos_enc
        self.residual = residual
        self.pe_dim = pe_dim
        
        if self.lap_pos_enc:
            raise NotImplementedError()
        
        
        self.embedding_h = nn.Linear(num_node_feat, d_model)

        self.embedding_e = nn.Linear(num_edge_feat, d_model)
        
        #self.node_encoder = NodeEncoder(
        #    device,
        #    input_seq_len=10,
        #    d_model=d_model,
        #    num_layers=2,
        #    num_heads=num_heads,
        #    d_k=d_k,
        #    d_v=d_v,
        #    d_ff=2*d_model,
        #    dropout=dropout)
        
        if self.pe_dim >0:
            self.embedding_lap_pe = nn.Linear(pe_dim, d_model)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(d_model, d_model, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(num_layers) ]) 
        
        self.MLP_layer = nn.Sequential(
            nn.Linear(2 * d_model, d_ff), 
            nn.ReLU(),
            nn.Linear(d_ff, d_ff), 
            nn.ReLU(),
            nn.Linear(d_ff, 1) 
        )      
        
    def forward(self, g, h, e, vehicle_node_id, num_nodes, h_lap_pe=None, masking=False):

        # input embedding
        #h = self.node_encoder(h) # To set to try more complex node encoder
        h = self.embedding_h(h)

        if self.pe_dim > 0 and h_lap_pe:
            h_lap_pe = self.embedding_lap_pe(h_lap_pe.float())
            h = h + h_lap_pe
        
        e = self.embedding_e(e)   
        
        # transformer layers
        for conv in self.layers:
            h, e = conv(g, h, e)
   
        # Concatenate vehicle node with each other node, separately
        batch_size = len(vehicle_node_id)
        vehicle_node_id = torch.tensor([i*num_nodes + k for i,k in enumerate(vehicle_node_id)], device=self.device)
        ks = vehicle_node_id.repeat_interleave(num_nodes).long()
        pairs = torch.cat([h[ks], h], dim=1)
        policy = torch.squeeze(self.MLP_layer(pairs))
        
        if masking:
            # Mask non neighbors
            neighbors = [g.successors(k) for k in vehicle_node_id]
            mask = torch.tensor([False if i in neighbors[i//num_nodes] else True for i in range(batch_size * num_nodes)], device=self.device)
            policy = policy.masked_fill(mask, -1e6)
        
        policy = torch.reshape(policy, (batch_size, num_nodes))

        # also return value
        value=0 # TODO
        return policy, value
        
