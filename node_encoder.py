import torch
from torch import nn
import torch.nn.functional as f
import time


class Head(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64):
        super().__init__()

        self.Linear_q = nn.Linear(d_model, d_k)
        self.Linear_k = nn.Linear(d_model, d_k)
        self.Linear_v = nn.Linear(d_model, d_v)

    def forward(self, q, k, v, mask=None):

        # Linear.
        q = self.Linear_q(q)  # Shape: (batch_size, input_seq_len, d_k).
        k = self.Linear_k(k)  # Shape: (batch_size, input_seq_len, d_k).
        v = self.Linear_v(v)  # Shape: (batch_size, input_seq_len, d_v).

        # Scaled dot-product attention.
        scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)  # Shape: (batch_size, input_seq_len, seq_len).
        if mask is not None:
            mask = mask.unsqueeze(-2)
            scores = scores.masked_fill(mask == 0, -1e10)
        scores = f.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)  # Shape: (batch_size, input_seq_len, d_v).

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_k=64, d_v=64):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(d_model, d_k, d_v) for _ in range(num_heads)]
        )

        self.head = Head(d_model, d_k, d_v)

        self.linear = nn.Linear(num_heads * d_v, d_model)

    def forward(self, q, k, v, mask=None):

        # Linear and scaled dot-product attention.
        x = [head(q, k, v, mask) for head in self.heads]

        # Concatenate.
        x = torch.cat(x, dim=-1)  # Shape: (batch_size, input_seq_len, h * d_v).

        # Linear.
        x = self.linear(x)  # Shape: (batch_size, input_seq_len, d_model).

        return x


def point_wise_feed_forward_network(d_model=512, d_ff=2048):

    return nn.Sequential(
        nn.Linear(d_model, d_ff),  # Shape: (batch_size, input_seq_len, d_ff).
        nn.ReLU(),
        nn.Linear(d_ff, d_model)   # Shape: (batch_size, input_seq_len, d_model).
    )


class EncoderLayer(nn.Module):
    def __init__(self,
                 input_seq_len,
                 d_model=512,
                 num_heads=8,
                 d_k=64,
                 d_v=64,
                 d_ff=2048,
                 dropout=0.1):
        super().__init__()

        # Multi-head attention.
        self.mha = MultiHeadAttention(d_model, num_heads, d_k, d_v)

        # Point-wise feed-forward network.
        self.pff = point_wise_feed_forward_network(d_model, d_ff)

        # Layer normalization.
        self.layernorm1 = nn.LayerNorm([input_seq_len, d_model], 1e-6)
        self.layernorm2 = nn.LayerNorm([input_seq_len, d_model], 1e-6)

        # Dropout.
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        # Multi-head attention.
        mha_out = self.mha(x, x, x, mask)  # Shape: (batch_size, input_seq_len, d_model).

        # Residual connection and layer normalization.
        x = self.layernorm1(x + self.dropout1(mha_out))  # Shape: (batch_size, input_seq_len, d_model).

        # Point-wise feed-forward network.
        pff_out = self.pff(x)  # Shape: (batch_size, input_seq_len, d_model).

        # Residual connection and layer normalization.
        x = self.layernorm2(x + self.dropout2(pff_out))  # Shape: (batch_size, input_seq_len, d_model).

        return x


class Encoder(nn.Module):
    def __init__(self,
                 input_seq_len,
                 num_layers=6,
                 d_model=512,
                 num_heads=8,
                 d_k=64,
                 d_v=64,
                 d_ff=2048,
                 dropout=0.1):
        super().__init__()

        self.num_layers = num_layers

        # N encoder layers.
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(input_seq_len=input_seq_len,
                          d_model=d_model,
                          num_heads=num_heads,
                          d_k=d_k,
                          d_v=d_v,
                          d_ff=d_ff,
                          dropout=dropout) for _ in range(num_layers)]
        )

        # Dropout.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):

        # Dropout.
        x = self.dropout(x)  # Shape: (batch_size, input_seq_len, d_model).

        # N encoder layers.
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, src_mask)  # Shape: (batch_size, input_seq_len, d_model).

        return x


class NodeEncoder(nn.Module):
    def __init__(self,
                 device,
                 input_seq_len=10,
                 d_model=512,
                 num_layers=2,
                 num_heads=8,
                 d_k=64,
                 d_v=64,
                 d_ff=2048,
                 dropout=0.1):
        super().__init__()

        self.device = device

        # User encoder.
        self.embed_type = nn.Linear(5, d_model)
        self.linear_window = nn.Linear(2, d_model)
        self.linear_servicetime = nn.Linear(1, d_model)
        self.linear_ridetime = nn.Linear(1, d_model)
        self.linear_load = nn.Linear(1, d_model)
        self.embed_vehiclepres = nn.Embedding(2, d_model)
        self.linear_capacity = nn.Linear(1, d_model)
        self.linear_nextavailable = nn.Linear(1, d_model)
        self.linear_routedur = nn.Linear(1, d_model)
        self.embed_isnext = nn.Embedding(2, d_model)


        self.node_encoder = Encoder(
            input_seq_len=input_seq_len,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            dropout=dropout
        )

        self.node_linear = nn.Linear(10 * d_model, d_model)


    # noinspection PyListCreation
    def forward(self, nodes, user_mask=None, src_mask=None):
        
        # Embedding.
        nodes = nodes.to(self.device)
        node_seq = []
        node_seq.append(self.embed_type(nodes[:,:5]))
        node_seq.append(self.linear_window(nodes[:,5:7]))
        node_seq.append(self.linear_ridetime(nodes[:,7:8])) # slice of 1 element to keep dimensions of size 1 in tensor
        node_seq.append(self.linear_ridetime(nodes[:,8:9]))
        node_seq.append(self.linear_load(nodes[:,9:10]))
        node_seq.append(self.embed_vehiclepres(nodes[:,10].long()))
        node_seq.append(self.linear_capacity(nodes[:,11:12]))
        node_seq.append(self.linear_nextavailable(nodes[:,12:13]))
        node_seq.append(self.linear_routedur(nodes[:,13:14]))
        node_seq.append(self.embed_isnext(nodes[:,14].long()))

        #for n in node_seq:
        #    print(n.size())

        x = torch.stack(node_seq, dim=1)
        x = self.node_encoder(x)
        #print(x.flatten(start_dim=2).size())
        x = self.node_linear(x.flatten(start_dim=1))

        return x