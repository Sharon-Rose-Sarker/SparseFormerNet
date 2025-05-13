import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import math
import time
import torch.profiler as profiler


# CBAM (Convolutional Block Attention Module)
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, seq_len = x.size(0), x.size(1), x.size(2)

        avg_pool = torch.mean(x, dim=2)

        channel_att_sum = self.mlp(avg_pool)

        channel_att_sum = channel_att_sum.unsqueeze(2).unsqueeze(3)


        return x * channel_att_sum

class SpatialGate(nn.Module):
    def __init__(self, gate_channels):
        super(SpatialGate, self).__init__()
        self.gate_channels = gate_channels
        self.conv = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = x.squeeze(2)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x_out = self.conv(avg_out + max_out)

        return x * self.sigmoid(x_out).unsqueeze(1)

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(gate_channels)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


# Sparse Attention
def sparse_dot_product_attention(query, key, value, mask=None, sparse_type='local', block_size=8):
    d_k = query.size(-1)
    if len(query.size()) == 4:
      batch_size, n_heads, seq_len, _ = query.size()
    else:
      batch_size, seq_len, _ = query.size()
      n_heads = 1

    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -65504)

    if sparse_type == 'local':
        if len(query.size()) == 4:
          attn_mask = torch.full_like(scores, -65504)
          for i in range(0, seq_len, block_size):
              start = i
              end = min(i + block_size, seq_len)
              attn_mask[:, :, start:end, start:end] = 0

          scores = scores + attn_mask
        else:
          attn_mask = torch.full_like(scores, -65504)
          for i in range(0, seq_len, block_size):
              start = i
              end = min(i + block_size, seq_len)
              attn_mask[:, start:end, start:end] = 0

          scores = scores + attn_mask

    attention = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(attention, value)

    return output, attention


# Encoder-Decoder Transformer
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, sparse_type='local', block_size=8):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.sparse_type = sparse_type
        self.block_size = block_size
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        def transform(x, linear):
            if len(x.size()) == 3:
              return linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            else:
              x = x.unsqueeze(1)
              return linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        query = transform(query, self.linear_q)
        key = transform(key, self.linear_k)
        value = transform(value, self.linear_v)

        x, attention = sparse_dot_product_attention(query, key, value, mask, self.sparse_type, self.block_size)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linear_out(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout, use_cbam=False):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.cbam = CBAM(gate_channels=d_model) if use_cbam else None
        self.sublayer = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = x + self.dropout[0](self.self_attn(self.sublayer[0](x), self.sublayer[0](x), self.sublayer[0](x), mask))
        x_ff = self.feed_forward(self.sublayer[1](x))
        return x + self.dropout[1](x_ff)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, cross_attn, feed_forward, dropout, use_cbam=False):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.cbam = CBAM(gate_channels=d_model) if use_cbam else None
        self.sublayer = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        x = x + self.dropout[0](self.self_attn(self.sublayer[0](x), self.sublayer[0](x), self.sublayer[0](x), tgt_mask))

        x = x + self.dropout[1](self.cross_attn(self.sublayer[1](x), memory, memory, src_mask))

        x_ff = self.feed_forward(self.sublayer[2](x))

        if self.cbam is not None:
            x_ff = x_ff.unsqueeze(1)
            x_ff = self.cbam(x_ff).squeeze(1)

        return x + self.dropout[2](x_ff)


class Encoder(nn.Module):
    def __init__(self, layers, norm):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layers, norm):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, d_model, positional_encoding, use_cbam=False):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.d_model = d_model
        self.positional_encoding = positional_encoding
        self.input_projection = nn.Linear(19, d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.input_projection(src)
        tgt = self.input_projection(tgt)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        memory = self.encoder(src, src_mask)

        memory_seq_len = memory.size(1)
        tgt_seq_len = tgt.size(1)

        if tgt_seq_len > memory_seq_len:
            tgt = tgt[:, :memory_seq_len, :]
        elif tgt_seq_len < memory_seq_len:
            padding = torch.zeros(tgt.size(0), memory_seq_len - tgt_seq_len, tgt.size(2), device=tgt.device)
            tgt = torch.cat((tgt, padding), dim=1)

        out = self.decoder(tgt, memory, src_mask, tgt_mask)
        out = self.out(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.pe = torch.zeros(1, max_len, d_model)
        self.pe[:, :, 0::2] = torch.sin(position * div_term)
        self.pe[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        if len(x.size()) == 2:
          x = x.unsqueeze(1)
          seq_len = x.size(1)
          x = x + self.pe[:, :seq_len, :].to(x.device).detach()
          return self.dropout(x)
        else:
          seq_len = x.size(1)
          x = x + self.pe[:, :seq_len, :].to(x.device).detach()
          return self.dropout(x)

