import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, h):
    super(MultiHeadAttention, self).__init__()
    self.d_model = d_model
    self.h = h

    self.W_q = nn.Linear(d_model, d_model, bias=False)
    self.W_k = nn.Linear(d_model, d_model, bias=False)
    self.W_v = nn.Linear(d_model, d_model, bias=False)
    self.W_o = nn.Linear(d_model, d_model, bias=False)

    self.d_k = d_model // h

  def forward(self, x):
    batch_size, seq_len = x.shape[0], x.shape[1]

    ##
    q = self.W_q(x)
    k = self.W_k(x)
    v = self.W_v(x)
    ## each of these is (batch, seq_len, d_model), we must make them now (batch, num_heads, seq_len, d_k)

    q = q.view(batch_size, seq_len, self.h, self.d_k) ## (batch, seq_len, num_heads, d_k)
    k = k.view(batch_size, seq_len, self.h, self.d_k) ## (batch, seq_len, num_heads, d_k)
    v = v.view(batch_size, seq_len, self.h, self.d_k) ## (batch, seq_len, num_heads, d_k)

    q = torch.transpose(q, 1, 2)
    k = torch.transpose(k, 1, 2)
    k = torch.transpose(k, 2, 3)
    v = torch.transpose(v, 1, 2)

    attention = torch.softmax(torch.divide(q@k, math.sqrt(self.d_k)), dim = -1) @ v ## (batch, num_heads, seq_len, d_k)

    ## Concat
    attention = torch.transpose(attention, 1, 2)
    attention = self.W_o(attention.reshape(batch_size, seq_len, self.d_model))
    return attention