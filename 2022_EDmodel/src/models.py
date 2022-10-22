# 220609

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Encoder(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dims,output_dim,dropout=0):
        """
        vocab_size: int, the number of input words
        embedding_dim: int, embedding dimention
        hidden_dims: list of int, the size of GRU hidden units
        output_dim: int, the unit size of bottleneck layer
        dropout: float [0,1], Dropout ratio
        """
        super().__init__()
        self.hidden_dims = hidden_dims
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        dims = self.hidden_dims.copy()
        dims.insert(0,embedding_dim)
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.gru = nn.ModuleList([nn.GRU(dims[i],dims[i+1],1) for i in range(len(self.hidden_dims))])
        self.linear = nn.Linear(sum(self.hidden_dims),output_dim)
        self.linear2 = nn.Linear(output_dim,sum(self.hidden_dims))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x,inference=False):
        # x: [T, B]
        embedding = self.dropout(self.embedding(x)) # [T, B, E]
        #norm = nn.BatchNorm1d(embedding.size(1)).to(DEVICE)
        #embedding = norm(embedding)
        states = []
        for v in self.gru:
            embedding, s = v(embedding) # [T, B, H]
            states.append(s.squeeze(0)) # [B, H]
        states = torch.cat(states,axis=1) # [B, Hsum]
        states = self.linear(states) # [B, Hout]
        if inference == False:
            states = states + torch.normal(0,0.05,size=states.shape).to(DEVICE)
        states = torch.tanh(states)
        states_to_dec = self.linear2(states)
        
        return embedding, states, states_to_dec # [T, B, H], [B, Hout], [B, Hsum]


class Decoder(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dims,output_dim,dropout=0):
        """
        vocab_size: int, the number of input words
        embedding_dim: int, embedding dimention
        hidden_dims: list of int, the size of GRU hidden units
        bottleneck_dim: int, the unit size of bottleneck layer
        dropout: float [0,1], Dropout ratio
        """
        super().__init__()
        self.hidden_dims = hidden_dims
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        dims = self.hidden_dims.copy()
        dims.insert(0,embedding_dim)
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.gru = nn.ModuleList([nn.GRU(dims[i],dims[i+1],1) for i in range(len(self.hidden_dims))])
        self.linear_out = nn.Linear(hidden_dims[-1],vocab_size,bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x,state):
        # x: [T, B]
        # state: [B, Hsum]
        embedding = self.dropout(self.embedding(x)) # [T, B, E]
        state = state.unsqueeze(0) # [1, B, Hsum]
        states = []
        cur_state = 0
        for v,w in zip(self.gru,self.hidden_dims):
            embedding, s = v(embedding,state[:,:,cur_state:cur_state+w].contiguous())
            cur_state += w
            states.append(s.squeeze(0))
        output = self.linear_out(embedding) # [T, B, Hvoc]

        return output, states


class Predictor(nn.Module):
    def __init__(self,input_dim,hidden_dims,output_dim,dropout=0,classify=True):
        """
        vocab_size: int, the number of input words
        input_dim: int, dimention of input tensor
        hidden_dims: list of int, the size of hidden layer units
        dropout: float [0,1], Dropout ratio
        classify: bool, whether classification task of not
        """
        super().__init__()
        dims = hidden_dims.copy()
        dims.insert(0,input_dim)
        self.linear = nn.ModuleList([nn.Linear(dims[i],dims[i+1]) for i in range(len(dims)-1)])
        self.out = nn.Linear(dims[-1],output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.classify = classify

    def forward(self,u):
        # u: tensor with length of input_dim
        for lin in self.linear:
            u = self.dropout(mish(lin(u)))
        u = self.dropout(self.out(u))
        if self.classify:
            sig = nn.Sigmoid()
            u = sig(u)
        return u


class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x,y):
        # x,y: [T, B]
        _, _, enc_state = self.encoder(x)
        output, dec_state = self.decoder(y,enc_state)

        return output, dec_state # [T, B, Hvoc], [B, Hsum]


class Seq2SeqwithPredictor(nn.Module):
    def __init__(self,encoder,decoder,predictor):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor

    def forward(self,x,y,prop=True):
        _, desc, enc_state = self.encoder(x)
        output, dec_state = self.decoder(y,enc_state)
        prop_out = self.predictor(desc) if prop else None

        return desc, output, dec_state, prop_out


def mish(x):
    return x*torch.tanh(F.softmax(x,dim=-1))