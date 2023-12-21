# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BART model, ported from the fairseq repo."""
import logging
import math
import random
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .activations import ACT2FN
from .configuration_bart import BartConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_utils import PreTrainedModel, create_position_ids_from_input_ids


from collections import OrderedDict
from typing import Any, BinaryIO, ContextManager, Dict, List, Optional, Tuple
import torch.nn as nn

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_scatter
from torch_geometric.data import Data

from torch_geometric.nn.conv import MessagePassing
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
)
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor
)
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
)
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

"""class GATsmol(torch.nn.Module):
  def __init__(self, dim_in, dim_h, dim_out, heads=8, **kwargs):
    #super().__init__()
    print("smollll")
    super(GATsmol, self).__init__(**kwargs)
    #self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
    #self.gat2 = GATv2Conv(dim_h * heads, dim_out, heads=1)
    self.gat = nn.ModuleList()
    print("Inside GAT")
    self.gat.append(GATv2Conv(dim_in, dim_h, heads=heads))
    self.gat.append(GATv2Conv(dim_h * heads, dim_out, heads=1))

  def forward(self, h, edge_index):
    h = torch.clone(h.detach())
    h = F.dropout(h, p=0.6, training=self.training)
    h = self.gat[0](h, edge_index)
    h = F.elu(h)
    h = F.dropout(h, p=0.6, training=self.training)
    h = self.gat[1](h, edge_index)
    return h"""
import json
class infusion_KG_QA_3(nn.Module):
    def __init__(self, dim: int = 1024):
        super(infusion_KG_QA_3, self).__init__()
        #print("Infusion kg qa 2")
        self.att_weight_cq = nn.Linear(1024, 1)
        self.att_weight_c = nn.Linear(1024, 1)
        self.att_weight_q = nn.Linear(1024, 1)
        self.l_layer_attn_flow = nn.Linear(1024 * 4, dim)
    def forward(self, c, q):
        c_len = c.size(1)
        # print("c_len : ",c_len)  #512
        q_len = q.size(1)
        cq = []
        for i in range(q_len):
            # (batch, 1, hidden_size * 2)
            qi = q.select(1, i).unsqueeze(1)
            # (batch, c_len, 1)
            ci = self.att_weight_cq(c * qi).squeeze()
            cq.append(ci)
        # (batch, c_len, q_len)
        cq = torch.stack(cq, dim=-1)
        # print("cq shape: ",cq.shape)  #torch.Size([4, 512, 512])
        # (batch, c_len, q_len)
        s = self.att_weight_c(c).expand(-1, -1, q_len) + \
            self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
            cq
        # print("s shape: ",s.shape)   #torch.Size([4, 512, 512])
        # (batch, c_len, q_len)
        a = F.softmax(s, dim=2)
        #print(c.size())
        #print(q.size())
        # print("a shape: ",a.shape)   #torch.Size([4, 512, 512])
        # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
        c2q_att = torch.bmm(a, q)
        # print("c2q_att shape: ",c2q_att.shape)  #torch.Size([4, 512, 1024])
        # (batch, 1, c_len)
        b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)

        # print("b shape: ",b.shape)  # torch.Size([4, 1, 1024])
        # # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
        q2c_att = torch.bmm(b, c).squeeze()
        # print("q2c_att shape: ",q2c_att.shape)   #torch.Size([4, 1024])
        # # (batch, c_len, hidden_size * 2) (tiled)
        q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
        # print("q2c_att shape: ",q2c_att.shape)   #torch.Size([4, 512, 1024])
        # # q2c_att = torch.stack([q2c_att] * c_len, dim=1)
        # # (batch, c_len, hidden_size * 8)
        #print(q2c_att.size())
        #print(c2q_att.size())
        x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
        #print("shape x:", x.shape)
        z= self.l_layer_attn_flow(x)
        #print("z shape: ",z.shape)   #torch.Size([4, 512, 1024])
        return z


class infusion_KG_QA_1(nn.Module):
    def __init__(self, dim: int = 1024):
        super(infusion_KG_QA_1, self).__init__()
        print("Infusion kg qa 1!!!!!!!!!")
        self.att_weight_cq = nn.Linear(1024, 1)
        self.att_weight_c = nn.Linear(1024, 1)
        self.att_weight_q = nn.Linear(1024, 1)
        self.modeling_lstm = nn.LSTM(dim*3, dim, bidirectional=True, num_layers=2, batch_first=True, dropout=0.2)
        self.l_layer_attn_flow = nn.Linear(1024 * 5, dim)
    def forward(self, c, q):
        c_len = c.size(1)
        # print("c_len : ",c_len)  #512
        q_len = q.size(1)
        cq = []
        for i in range(q_len):
            # (batch, 1, hidden_size * 2)
            qi = q.select(1, i).unsqueeze(1)
            # (batch, c_len, 1)
            ci = self.att_weight_cq(c * qi).squeeze()
            cq.append(ci)
        # (batch, c_len, q_len)
        cq = torch.stack(cq, dim=-1)
        # print("cq shape: ",cq.shape)  #torch.Size([4, 512, 512])
        # (batch, c_len, q_len)
        s = self.att_weight_c(c).expand(-1, -1, q_len) + \
            self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
            cq
        # print("s shape: ",s.shape)   #torch.Size([4, 512, 512])
        # (batch, c_len, q_len)
        a = F.softmax(s, dim=2)
        # print("a shape: ",a.shape)   #torch.Size([4, 512, 512])
        # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
        c2q_att = torch.bmm(a, q)
        # print("c2q_att shape: ",c2q_att.shape)  #torch.Size([4, 512, 1024])
        # (batch, 1, c_len)
        # attn_output, attn_output_weights = self.multihead_attn(c2q_att, c2q_att, c2q_att)
        # b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
        # # print("b shape: ",b.shape)  # torch.Size([4, 1, 1024])
        # # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
        # q2c_att = torch.bmm(b, c).squeeze()
        # # print("q2c_att shape: ",q2c_att.shape)   #torch.Size([4, 1024])
        # # (batch, c_len, hidden_size * 2) (tiled)
        # q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
        # # print("q2c_att shape: ",q2c_att.shape)   #torch.Size([4, 512, 1024])
        # # q2c_att = torch.stack([q2c_att] * c_len, dim=1)
        # # (batch, c_len, hidden_size * 8)
        x = torch.cat([c, c2q_att, c * c2q_att], dim=-1)
        # print("x shape: ",x.shape)   #torch.Size([4, 512, 3072])
        m, _ = self.modeling_lstm(x)
        # print("m shape: ",m.shape)   #torch.Size([4, 512, 2048])
        m2 =  torch.cat([x, m], dim=-1)
        # print("m2 shape: ",m2.shape)   #torch.Size([4, 512, 3072])
        z= self.l_layer_attn_flow(m2)
        # print("z shape: ",z.shape)   #torch.Size([4, 512, 1024])
        return z

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count=8, model_dim=1024, dropout=0.1):
        print("multihead latest!!!!!!!!!!")
        print("asdasd")
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                device = key.device
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"].to(device), key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"].to(device), value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif type == "src":
                query = self.linear_query(query)
                if layer_cache["src_memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["src_memory_keys"],\
                               layer_cache["src_memory_values"]
                layer_cache["src_memory_keys"] = key
                layer_cache["src_memory_values"] = value
            elif type == "knl":
                query = self.linear_query(query)
                if layer_cache["knl_memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["knl_memory_keys"],\
                               layer_cache["knl_memory_values"]
                layer_cache["knl_memory_keys"] = key
                layer_cache["knl_memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = unshape(torch.matmul(drop_attn, value))

        output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()

        return output


class infusion_KG_QA_5(nn.Module):
    def __init__(self, dim: int = 1024):
        super(infusion_KG_QA_5, self).__init__()
        #print("Infusion kg qa 3")
        self.att_weight_cq = nn.Linear(1024, 1)
        self.att_weight_c = nn.Linear(1024, 1)
        self.att_weight_q = nn.Linear(1024, 1)
        self.feed_forward = nn.Sequential(
            nn.Linear(1024 * 4, 1024 * 2), nn.GELU(), nn.Linear(1024 * 2, 1024),
        )
        self.layer_norm1 = nn.LayerNorm(1024 * 4)
        self.l_layer_attn_flow = nn.Linear(1024 * 5, dim)
    def forward(self, c, q):
        c_len = c.size(1)
        # print("c_len : ",c_len)  #512
        q_len = q.size(1)
        cq = []
        for i in range(q_len):
            # (batch, 1, hidden_size * 2)
            qi = q.select(1, i).unsqueeze(1)
            # (batch, c_len, 1)
            ci = self.att_weight_cq(c * qi).squeeze()
            cq.append(ci)
        # (batch, c_len, q_len)
        cq = torch.stack(cq, dim=-1)
        # print("cq shape: ",cq.shape)  #torch.Size([4, 512, 512])
        # (batch, c_len, q_len)
        s = self.att_weight_c(c).expand(-1, -1, q_len) + \
            self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
            cq
        # print("s shape: ",s.shape)   #torch.Size([4, 512, 512])
        # (batch, c_len, q_len)
        a = F.softmax(s, dim=2)
        # print("a shape: ",a.shape)   #torch.Size([4, 512, 512])
        # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
        c2q_att = torch.bmm(a, q)
        # print("c2q_att shape: ",c2q_att.shape)  #torch.Size([4, 512, 1024])
        # (batch, 1, c_len)
        b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
        # print("b shape: ",b.shape)  # torch.Size([4, 1, 1024])
        # # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
        q2c_att = torch.bmm(b, c).squeeze()
        # print("q2c_att shape: ",q2c_att.shape)   #torch.Size([4, 1024])
        # # (batch, c_len, hidden_size * 2) (tiled)
        q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
        # print("q2c_att shape: ",q2c_att.shape)   #torch.Size([4, 512, 1024])
        # # q2c_att = torch.stack([q2c_att] * c_len, dim=1)
        # # (batch, c_len, hidden_size * 8)
        x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
        #print("shape x:", x.shape)
        x= self.layer_norm1(x)
        #print("x shape after layer_norm: ",x.shape)   #torch.Size([4, 512, 4096])
        z = self.feed_forward(x)
        #print("z shape after feed_forward: ",z.shape)   #torch.Size([4, 512, 1024])
        z = torch.cat([x,z], dim=-1)
        #print("z shape after concat: ",z.shape)   #torch.Size([4, 512, 1024])
        z= self.l_layer_attn_flow(z)
        #print("z shape after concat: ",z.shape)   #torch.Size([4, 512, 1024])
        return z


class KG_QA(nn.Module):
    def __init__(self, dim: int = 1024):
        super(KG_QA, self).__init__()
        self.att_weight_cq = nn.Linear(1024, 1)
        self.att_weight_c = nn.Linear(1024, 1)
        self.att_weight_q = nn.Linear(1024, 1)
        self.l_layer_attn_flow = nn.Linear(1024 * 3, dim)
    def forward(self, c, q):
        c_len = c.size(1)
        # print("c_len : ",c_len)  #512
        q_len = q.size(1)
        cq = []
        for i in range(q_len):
            # (batch, 1, hidden_size * 2)
            qi = q.select(1, i).unsqueeze(1)
            # (batch, c_len, 1)
            ci = self.att_weight_cq(c * qi).squeeze()
            cq.append(ci)
        # (batch, c_len, q_len)
        cq = torch.stack(cq, dim=-1)
        # print("cq shape: ",cq.shape)  #torch.Size([4, 512, 512])

        # (batch, c_len, q_len)
        s = self.att_weight_c(c).expand(-1, -1, q_len) + \
            self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
            cq
        # print("s shape: ",s.shape)   #torch.Size([4, 512, 512])

        # (batch, c_len, q_len)

        a = F.softmax(s, dim=2)
        # print("a shape: ",a.shape)   #torch.Size([4, 512, 512])
        # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)

        c2q_att = torch.bmm(a, q)
        # print("c2q_att shape: ",c2q_att.shape)  #torch.Size([4, 512, 1024])
        # (batch, 1, c_len)

        # attn_output, attn_output_weights = self.multihead_attn(c2q_att, c2q_att, c2q_att)

        # b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
        # # print("b shape: ",b.shape)  # torch.Size([4, 1, 1024])
        # # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)

        # q2c_att = torch.bmm(b, c).squeeze()
        # # print("q2c_att shape: ",q2c_att.shape)   #torch.Size([4, 1024])
        # # (batch, c_len, hidden_size * 2) (tiled)

        # q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
        # # print("q2c_att shape: ",q2c_att.shape)   #torch.Size([4, 512, 1024])
        # # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

        # # (batch, c_len, hidden_size * 8)
        x = torch.cat([c, c2q_att, c * c2q_att], dim=-1)
        # print("x shape: ",x.shape)   #torch.Size([4, 512, 3072])
        x = self.l_layer_attn_flow(x)
        return x

class MHA(nn.Module):
    def __init__(self, d_model: int = 1024, num_heads: int = 8):
        super(MHA, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.heads = nn.ModuleList([KG_QA() for i in range(num_heads)])
        self.layer = nn.Linear(1024*num_heads, 1024)

    def forward(
            self,
            query: Tensor,
            key: Tensor
    ) -> Tuple[Tensor, Tensor]:
        outt = []
        for head in self.heads:
            outt.append(head(query, key))
        outt = torch.concat(outt, dim=2)
        context = self.layer(outt)
        return context


class GATsmol(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATsmol, self).__init__()
        print("hehehheh")
        # use our gat message passing
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=4)
        self.conv2 = GATv2Conv(4 * hidden_dim, hidden_dim, heads=4)

        self.post_mp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.Dropout(0.6),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x, edge_index, adj=None):
        x = torch.clone(x.detach())
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.dropout(F.relu(x), p=0.6, training=self.training)
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.dropout(F.relu(x), p=0.6, training=self.training)
        # MLP output
        x = self.post_mp(x)
        return x


class myGATv2(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1,
                 negative_slope=0.2, dropout=0., **kwargs):
        super(myGATv2, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None
        self._alpha = None
        # self.lin_l is the linear transformation that you apply to embeddings
        # BEFORE message passing.
        self.lin_l = nn.Linear(in_channels, heads * out_channels)
        self.lin_r = self.lin_l

        self.att = Parameter(torch.Tensor(1, heads, out_channels))
        self.reset_parameters()

    # initialize parameters with xavier uniform
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, size=None):
        H, C = self.heads, self.out_channels  # DIM：H, outC
        # Linearly transform node feature matrix.
        #print(x.size())
        #print(x.is_cuda)
        x_source = self.lin_l(x)  # DIM: [Nodex x In] [in x H * outC] => [nodes x H * outC] => [nodes, H, outC]
        #print(type(x_source))
        x_source = x_source.view(-1, H, C)
        #print(type(x_source))
        x_target = self.lin_r(x).view(-1, H, C)  # DIM: [Nodex x In] [in x H * outC] => [nodes x H * outC] => [nodes, H, outC]

        #  Start propagating messages (runs message and aggregate)
        out = self.propagate(edge_index, x=(x_source, x_target), size=size)  # DIM: [nodes, H, outC]
        out = out.view(-1, self.heads * self.out_channels).to(torch.device("cuda"))  # DIM: [nodes, H * outC]
        alpha = self._alpha
        self._alpha = None
        return out

    # Process a message passing
    def message(self, x_j, x_i, index, ptr, size_i):
        # computation using previous equationss
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)  # See Equation above: Apply the non-linearty function
        alpha = (x * self.att).sum(dim=-1)  # Apply attnention "a" layer after the non-linearity
        alpha = softmax(alpha, index, ptr, size_i)  # This softmax only calculates it over all neighbourhood nodes
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # Multiple attention with node features for all edges
        out = x_j * alpha.unsqueeze(-1)
        out = out
        return out

    # Aggregation of messages
    def aggregate(self, inputs, index, dim_size=None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim,
                                    dim_size=dim_size, reduce='sum')
        out = out
        return out

class GATv2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATv2, self).__init__()
        # use our gat message passing
        self.conv1 = myGATv2(input_dim, hidden_dim, heads=4)
        self.conv2 = myGATv2(4 * hidden_dim, hidden_dim, heads=4)
        #self.conv1.to(torch.device("cuda"))
        #self.conv2.to(torch.device("cuda"))
        self.post_mp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.Dropout(0.6),
            nn.Linear(hidden_dim, output_dim))
        #self.post_mp = nn.Linear(4 * hidden_dim, output_dim)

    def forward(self, x, edge_index,  adj=None):
        # Layer 1
        x = torch.clone(x.detach())

        x = self.conv1(x, edge_index)
        x = F.dropout(F.relu(x), p=0.6, training=self.training)
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.dropout(F.relu(x), p=0.6, training=self.training)
        # MLP output
        x = self.post_mp(x)
        #print(x.size())
        return x
        #return F.sigmoid(x)

class GCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True,
                 directed=False, self_loop=True, **kwargs):
        """
        Initialize a GCN layer.
        Args:
            in_channels      In-channel dimension of node embeddings
            out_channels     Out-channel dimension of node embeddings
            bias             A boolean value determining whether we add a
                                learnable bias term in linear transformation
            directed         A boolean value determining whether we use directed
                                message passing D^{-1}A or use symmetric normalized
                                adjacency matrix D^{-1/2}AD^{-1/2}
            self_loop        A boolean value determining whether we add a self-
                                loop for each node
        """
        super(GCNLayer, self).__init__(**kwargs, aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.directed = directed
        self.self_loop = self_loop

        # Define the layers needed for the message and update functions below.
        # self.lin is the linear transformation that we apply to the embedding.
        self.lin = nn.Linear(self.in_channels, self.out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset all learnable parameters in the linear transformation.
        """
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        """
        Produce a forward propagation of GCN layer.

        Args:
            x             The node embedding
            edge_index    The (2, |E|) adjacency list of the graph
            edge_weight   The (|E|) vector specifying the edge weights in the graph
                            (for unweighted graph, edge weight is 1)

        Returns:
            An updated node embedding
        """
        # Add self-loops to the adjacency matrix.
        if self.self_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            edge_weight = torch.cat((edge_weight, torch.ones(x.size(0))), dim=-1)

        # Apply linear transformation on node features.
        x = self.lin(x)

        # Compute normalization by updated node degree.
        if self.directed:
            row, _ = edge_index
            deg = degree(row, x.size(0), dtype=x.dtype)  # only out-degree
            deg_inv = deg.pow(-1)
            deg_inv[deg_inv == float('inf')] = 0
            norm = deg_inv[row]
        else:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=(x, x), norm=norm, edge_weight=edge_weight)

    def message(self, x_j, edge_weight, norm):
        """
        Send the message of the neighboring node (i.e., x_j) to the source node (i.e., x_i).

        Args:
            x_j           The embedding of the neighboring node of source node x_i
            edge_weight   The edge weight of certain edge
            norm          Normalization constant determined by self.directed

        Returns:
            A message sending from the neighboring node to the source node
        """
        a = norm.view(-1, 1).to(torch.device("cuda"))
        b = edge_weight.view(-1, 1).to(torch.device("cuda"))
        w = a * x_j * b
        z = w.to(torch.device("cuda"))
        return z


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        """
        Initialize a GCN model.
        Args:
            input_dim       Input dimension of node embeddings
            hidden_dim      Hidden dimension of node embeddings
            output_dim      Output dimension of node embeddings
            num_layers      The number of GCN layers
            dropout         The dropout ratio in (0, 1]
                              (dropout: the probability of an element getting zeroed)
            return_embeds   A boolean value determining whether we skip the
                              classification layer and return node embeddings
        """

        super(GCN, self).__init__()

        # Construct all convs
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, directed=False)
                                          for i in range(self.num_layers - 1)])

        # Construct batch normalization
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim)
                                        for i in range(self.num_layers - 1)])
        # First GCN layer
        self.convs[0] = GCNLayer(input_dim, hidden_dim, directed=False)
        # Last GCN layer
        self.last_conv = GCNLayer(hidden_dim, output_dim, directed=False)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.dropout = dropout
        self.return_embeds = return_embeds

    def reset_parameters(self):
        """
        Reset all learnable parameters in GCN layers and Batch Normalization
        Layers.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        """
        Produce a forward propagation of GCN model. Before the last GCN layer,
        we transform the embedding (x) in the following sequence:
          x -> GCN_Layer -> Batch_Norm -> ReLU -> Dropout.
        At the last GCN layer, the following sequence is applied:
          x -> GCN Layer -> Softmax -> output.

        Args:
            x             The node embedding
            edge_index    The adjacency list of the graph

        Returns:
            out           The predictions of labels / the updated node embedding
        """
        x = torch.clone(x.detach())
        for l in range(self.num_layers - 1):
            # Unweighted graph has weight 1.
            x = self.convs[l](x, edge_index, torch.ones(edge_index.shape[1]))
            x = self.bns[l](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.last_conv(x, edge_index, torch.ones(edge_index.shape[1]))
        if self.return_embeds:
            out = x
        else:
            out = self.softmax(x)

        return out




"""class StaticAttention(nn.Module):
    def __init__(self, hidden_size, entity_vocab, relation_vocab, t_embed):
        super().__init__()
        self.t_embed = t_embed
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.hidden_size = hidden_size
        self.entity_embedding = nn.Embedding(self.entity_vocab, t_embed)
        # self.entity_embedding.weight = nn.Parameter(embedding_matrix_entity, requires_grad=True)
        self.rel_embedding = nn.Embedding(self.relation_vocab, t_embed)
        # self.rel_embedding.weight = nn.Parameter(embedding_matrix_rel, requires_grad=True)
        self.MLP = nn.Linear(3 * self.t_embed, 3 * self.t_embed)
        self.lii = nn.Linear(3*self.t_embed, self.hidden_size, bias=False)

    def forward(self, kg_enc_input):
        # print("kg_enc_input size: ",kg_enc_input.size()) #torch.Size([8, 512, 3])
        batch_size, _, _ = kg_enc_input.size()
        # print("batch_size :",batch_size)
        head, rel, tail = torch.split(kg_enc_input, 1, 2)  # (bsz, pl, tl)
        # print("head shape: ",head.shape) #torch.Size([bsz, 512, 1])
        # print("rel shape: ",rel.shape) #torch.Size([bsz, 512, 1])
        # print("tail shape: ",tail.shape) #torch.Size([bsz, 512, 1])
        head_emb =  self.entity_embedding(head.squeeze(-1))  # (bsz, pl, tl, t_embed)
        # print("head_emb shape: ",head_emb.shape) #torch.Size([bsz, 512, 300])
        rel_emb = self.rel_embedding(rel.squeeze(-1)) # (bsz, pl, tl, t_embed)
        # print("rel_emb shape: ",rel_emb.shape) #torch.Size([bsz, 512, 300])
        tail_emb = (self.entity_embedding(tail.squeeze(-1)))  # (bsz, pl, tl, t_embed)
        # print("tail_emb shape: ",tail_emb.shape) #torch.Size([bsz, 512, 300])
        triple_cat =torch.cat([head_emb, rel_emb, tail_emb], 2)
        # print("triple_cat shape: ",triple_cat.shape)
        triple_emb = self.MLP(triple_cat)  # (bsz, pl, 3 * t_embed)
        triple_emb = self.lii(triple_emb)
        # print("triple_emb shape: ",triple_emb.shape)

        # triple_emb = torch.mean(triple_emb,2)
        return triple_emb"""

"""class StaticAttention(nn.Module):
    def __init__(self, hidden_size, entity_vocab, relation_vocab, t_embed):
        super().__init__()
        self.t_embed = t_embed
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.hidden_size = hidden_size
        self.entity_embedding = nn.Embedding(self.entity_vocab, t_embed)
        # self.entity_embedding.weight = nn.Parameter(embedding_matrix_entity, requires_grad=True)
        self.rel_embedding = nn.Embedding(self.relation_vocab, t_embed)
        # self.rel_embedding.weight = nn.Parameter(embedding_matrix_rel, requires_grad=True)
        self.MLP = nn.Linear(3 * self.t_embed, 3 * self.t_embed)
        # self.lii = nn.Linear(3*self.t_embed, self.hidden_size, bias=False)
        self.lii = nn.Linear(self.t_embed, self.hidden_size, bias=False)
        self.linear = nn.Linear(self.t_embed,self.t_embed)

    def forward(self, kg_enc_input):
        # print("kg_enc_input size: ",kg_enc_input.size()) #torch.Size([8, 512, 3])
        batch_size, _, _ = kg_enc_input.size()
        # print("batch_size :",batch_size)
        head, rel, tail = torch.split(kg_enc_input, 1, 2)  # (bsz, pl, tl)
        # print("head shape: ",head.shape) #torch.Size([bsz, 512, 1])
        # print("rel shape: ",rel.shape) #torch.Size([bsz, 512, 1])
        # print("tail shape: ",tail.shape) #torch.Size([bsz, 512, 1])
        head_emb =  self.entity_embedding(head.squeeze(-1))  # (bsz, pl, tl, t_embed)
        # print("head_emb shape: ",head_emb.shape) #torch.Size([bsz, 512, 300])
        rel_emb = self.rel_embedding(rel.squeeze(-1)) # (bsz, pl, tl, t_embed)
        # print("rel_emb shape: ",rel_emb.shape) #torch.Size([bsz, 512, 300])
        tail_emb = (self.entity_embedding(tail.squeeze(-1)))  # (bsz, pl, tl, t_embed)
        # print("tail_emb shape: ",tail_emb.shape) #torch.Size([bsz, 512, 300])

        # triple_cat =torch.cat([head_emb, rel_emb, tail_emb], 2)
        # # print("triple_cat shape: ",triple_cat.shape)
        # triple_emb = self.MLP(triple_cat)  # (bsz, pl, 3 * t_embed)
        # # print("triple_emb shape: ",triple_emb.shape) #torch.Size([bsz, 512, 900])
        # triple_emb = self.lii(triple_emb)
        # # print("triple_emb shape after linear layer: ",triple_emb.shape) #torch.Size([bsz, 512, 768])
        # return triple_emb

        tail_dash= self.linear(head_emb) - self.linear(rel_emb)
        # print("tail_dash shape: ",tail_dash.shape)
        characters = torch.max(tail_emb, 1)[1]
        # print("characters shape: ",characters.shape)
        loss_emb = nn.CrossEntropyLoss()
        emb_loss=loss_emb(tail_dash,characters)
        triple_emb = self.lii(tail_dash)
        # print("triple_emb shape after linear layer: ",triple_emb.shape) #torch.Size([bsz, 512, 768])
        return triple_emb"""

class StaticAttention(nn.Module):
    def __init__(self, hidden_size, entity_vocab, relation_vocab, t_embed):
        super().__init__()
        self.t_embed = t_embed
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.hidden_size = hidden_size
        self.entity_embedding = nn.Embedding(self.entity_vocab + self.relation_vocab, self.t_embed)

        #liss = []
        #for i in range(entity_vocab + relation_vocab):
        #    liss.append(i)
        #temp_lis = torch.tensor(liss, dtype=torch.long)
        #self.embeddings = self.entity_embedding(temp_lis).to(torch.device("cuda"))
        x = json.load(open("/home/Abstractive_Qa/embeddings.json"))
        listt = []
        for i in range(len(x.keys())):
            listt.append([])

        for k in x.keys():
            listt[int(k)] = x[k]
        # print(list)

        self.embeddings = torch.tensor(listt).to(torch.device("cuda"))
        #self.gcn = GCN(input_dim=t_embed, hidden_dim=t_embed, output_dim=t_embed, num_layers=2, dropout=0.5,
        #          return_embeds=True).to(torch.device("cuda"))
        hpl = [8, 1]
        nfl = [t_embed, t_embed, t_embed]
        #self.gcn = GAT(num_of_layers=2, num_heads_per_layer=hpl, num_features_per_layer=nfl).to(torch.device("cuda"))

        #self.gcn = GATv2(t_embed, t_embed, t_embed)
        self.gcn = GATsmol(t_embed, t_embed, t_embed)
        #print(next(self.gcn.parameters()).is_cuda)
        #self.MLP = nn.Linear(3 * self.t_embed, 3 * self.t_embed)
        # self.lii = nn.Linear(3*self.t_embed, self.hidden_size, bias=False)
        self.lii = nn.Linear(self.t_embed*3, self.hidden_size, bias=False)
        #self.linear = nn.Linear(self.t_embed,self.t_embed)

    def forward(self, kg_enc_input):
        kg = torch.split(kg_enc_input, dim=0, split_size_or_sections=1)
        output = []
        for kgi in kg:
            kg_inp = kgi.squeeze(0)
            head, rel, tail = torch.split(kg_inp, dim=1, split_size_or_sections=1)
            head = head.squeeze(-1)
            rel = rel.squeeze(-1)
            tail = tail.squeeze(-1)
            edge_list_1 = torch.stack([head, rel])
            edge_list_2 = torch.stack([rel, tail])

            edge_list = torch.cat([edge_list_1, edge_list_2], 1)

            w = self.gcn.forward(x=self.embeddings, edge_index=edge_list)
            #w = self.embeddings
            #w = self.gcn.forward((self.embeddings,edge_list))[0]

            # GCN

            triples_emb = []
            for row in kg_inp:
                triples_emb.append(torch.cat([w[row[0]], w[row[1]], w[row[2]]], 0))
            output.append(torch.stack(triples_emb))
        final = torch.stack(output)
        triple_emb = self.lii(final)
        return triple_emb


logger = logging.getLogger(__name__)


BART_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bart-large": "https://cdn.huggingface.co/facebook/bart-large/pytorch_model.bin",
    "bart-large-mnli": "https://cdn.huggingface.co/facebook/bart-large-mnli/pytorch_model.bin",
    "bart-large-cnn": "https://cdn.huggingface.co/facebook/bart-large-cnn/pytorch_model.bin",
    "bart-large-xsum": "https://cdn.huggingface.co/facebook/bart-large-xsum/pytorch_model.bin",
    "mbart-large-en-ro": "https://cdn.huggingface.co/facebook/mbart-large-en-ro/pytorch_model.bin",
}

BART_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matters related to general usage and behavior.

    Parameters:
        config (:class:`~transformers.BartConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.

"""
BART_GENERATION_EXAMPLE = r"""
    Examples::

        from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
        # see ``examples/summarization/bart/evaluate_cnn.py`` for a longer example
        model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
        tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
        ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
        inputs = tokenizer.batch_encode_plus([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
        # Generate Summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

"""

BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
               Indices of input sequence tokens in the vocabulary. Use BartTokenizer.encode to produce them.
            Padding will be ignored by default should you provide it.
            Indices can be obtained using :class:`transformers.BartTokenizer.encode(text)`.
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices in input_ids.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`, defaults to :obj:`None`):
            Tuple consists of (`last_hidden_state`, `optional`: `hidden_states`, `optional`: `attentions`)
            `last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`) is a sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention of the decoder.
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
            Provide for translation and summarization training. By default, the model will create this tensor by shifting the input_ids right, following the paper.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`, defaults to :obj:`None`):
            Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
            If you want to change padding behavior, you should read :func:`~transformers.modeling_bart._prepare_decoder_inputs` and modify.
            See diagram 1 in the paper for more info on the default strategy
"""


def invert_mask(attention_mask):
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)


def _prepare_bart_decoder_inputs(
    config, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32
):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=decoder_input_ids.device
    )
    return decoder_input_ids, decoder_padding_mask, causal_mask


class PretrainedBartModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"
    pretrained_model_archive_map = BART_PRETRAINED_MODEL_ARCHIVE_MAP

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


# Helper Functions, mostly for making masks
def _check_shapes(shape_1, shape2):
    if shape_1 != shape2:
        raise AssertionError("shape mismatch: {} != {}".format(shape_1, shape2))


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


# Helper Modules


class EncoderLayer(nn.Module):
    def __init__(self, config: BartConfig, state="off"):
        super().__init__()
        self.embed_dim = config.d_model
        self.output_attentions = config.output_attentions
        self.self_attn = SelfAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout,
        )
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        #self.block = KG_QA()
        #print(state)
        #if state == "on":
        #    self.block = KG_QA()
        #self.l_layer = torch.nn.Linear(552, 512)
        #print("archi11111")

    def forward(self, x, encoder_padding_mask, kg_input, infusion=0):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(
            query=x, key=x, key_padding_mask=encoder_padding_mask, need_weights=self.output_attentions
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if infusion == 1: # attention
            if self.normalize_before:
                x = self.self_attn_layer_norm(x)
            residual = x
            x = x.transpose(0, 1)
            x = self.block(x, kg_input)
            x = x.transpose(0, 1)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if not self.normalize_before:
                x = self.cross_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if infusion == 2: # linear
            res = x
            x = x.transpose(0, 1)
            z = torch.concat([x, kg_input], dim=1)
            z = torch.permute(z, (0, 2, 1))
            z = self.l_layer(z)
            z = torch.permute(z, (0, 2, 1))
            z = z.transpose(0, 1)
            x = res + z
        return x, attn_weights


class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens, hidden_size=None, entity_vocab=None, relation_vocab=None, t_embed=None):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx,
            )
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None

        # Linear concat
        self.entity_vocab = len(json.load(open("/home/Abstractive_Qa/KG_Bart/bart/vocab_entities.json", "r")))
        self.relation_vocab = len(json.load(open("/home/Abstractive_Qa/KG_Bart/bart/vocab_relation.json", "r")))
        self.t_embed = 768
        #print("biobertttttttttt!!!")
        self.StaticAttention = StaticAttention(self.hidden_size, self.entity_vocab, self.relation_vocab, self.t_embed)
        #print("static attention", next(self.StaticAttention.parameters()).is_cuda)
        self.l_layer = torch.nn.Linear(512+512, 512)

        # attention
        self.att_weight_c = nn.Linear(1024, 1)
        self.att_weight_q = nn.Linear(1024, 1)
        self.att_weight_cq = nn.Linear(1024, 1)
        self.l_layer_attn_flow = nn.Linear(1024*3, 1024)
        print("atfl3333333333333333 added!!!xexe")
        self.att_flow_layer_2 = infusion_KG_QA_3()

    def att_flow_layer(self, c, q):
        """
        :param c: (batch, c_len, hidden_size )
        :param q: (batch, q_len, hidden_size )
        :return: (batch, c_len, q_len)
        """
        # print("c_len : ",c.shape)  #torch.Size([4, 512, 1024])
        # print("q_len : ",q.shape)  #torch.Size([4, 512, 1024])
        c_len = c.size(1)
        # print("c_len : ",c_len)  #512
        q_len = q.size(1)
        # print("q_len : ",q_len)  #512

        cq = []
        for i in range(q_len):
            # (batch, 1, hidden_size * 2)
            qi = q.select(1, i).unsqueeze(1)
            # (batch, c_len, 1)
            ci = self.att_weight_cq(c * qi).squeeze()
            cq.append(ci)
        # (batch, c_len, q_len)
        cq = torch.stack(cq, dim=-1)
        # print("cq shape: ",cq.shape)  #torch.Size([4, 512, 512])

        # (batch, c_len, q_len)
        s = self.att_weight_c(c).expand(-1, -1, q_len) + \
            self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
            cq
        # print("s shape: ",s.shape)   #torch.Size([4, 512, 512])

        # (batch, c_len, q_len)

        a = F.softmax(s, dim=2)
        # print("a shape: ",a.shape)   #torch.Size([4, 512, 512])
        # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)

        c2q_att = torch.bmm(a, q)
        # print("c2q_att shape: ",c2q_att.shape)  #torch.Size([4, 512, 1024])
        # (batch, 1, c_len)

        # attn_output, attn_output_weights = self.multihead_attn(c2q_att, c2q_att, c2q_att)

        # b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
        # # print("b shape: ",b.shape)  # torch.Size([4, 1, 1024])
        # # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)

        # q2c_att = torch.bmm(b, c).squeeze()
        # # print("q2c_att shape: ",q2c_att.shape)   #torch.Size([4, 1024])
        # # (batch, c_len, hidden_size * 2) (tiled)

        # q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
        # # print("q2c_att shape: ",q2c_att.shape)   #torch.Size([4, 512, 1024])
        # # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

        # # (batch, c_len, hidden_size * 8)
        x = torch.cat([c, c2q_att, c * c2q_att], dim=-1)
        # print("x shape: ",x.shape)   #torch.Size([4, 512, 3072])
        x = self.l_layer_attn_flow(x)
        return x


    def forward(
        self, input_ids, attention_mask=None, kg_input=None
    ):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *self.output_hidden_states:* is True.
                - **all_attentions** (List[Tensor]): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        triple_emb = self.StaticAttention(kg_input)

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        coun = 0
        for encoder_layer in self.layers:
            inf = 0
            coun += 1
            if coun == len(self.layers):
                inf = 1
            if self.output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask, triple_emb) # inf

            if self.output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if self.output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)

        #print("hereee")
        #print("Bart output", x.size())
        #print("GNN output", triple_emb.size())
        _method = "att"

        if _method == "att":
            x = x + self.att_flow_layer_2(x, triple_emb)
            #x = x + self.att_flow_layer(x, triple_emb)
        else:
            x = torch.cat((x,triple_emb),1)
            x = torch.permute(x, (0, 2, 1))
            x = self.l_layer(x)
            x = torch.permute(x, (0, 2, 1))

        return x, encoder_states, all_attentions


class DecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.output_attentions = config.output_attentions
        self.self_attn = SelfAttention(
            embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
    ):
        residual = x

        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            need_weights=self.output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


class BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.d_model, self.padding_idx,
            )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_cached_states=None,
        use_cache=False,
        **unused
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_cached_states (dict or None): dictionary used for storing state during generation

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - hidden states
                - attentions
        """
        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if self.output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if self.output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        return x, next_cache, all_hidden_states, list(all_self_attns)


def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache


class SelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def _shape(self, tensor, dim_0, bsz):
        return tensor.contiguous().view(dim_0, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute key and value if they are static
                if static_kv:
                    key = None
        else:
            saved_state = None
            layer_state = {}

        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)

        # Update cache
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len,)

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training,)

        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)
        key_padding_mask = self._cat_prev_key_padding_mask(
            key_padding_mask, prev_key_padding_mask, bsz, k.size(1), static_kv
        )
        return k, v, key_padding_mask

    @staticmethod
    def _cat_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)

        elif key_padding_mask is not None:
            filler = torch.zeros(
                batch_size,
                src_len - key_padding_mask.size(1),
                dtype=key_padding_mask.dtype,
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    # This can trivially be shared with RobertaClassificationHead

    def __init__(
        self, input_dim, inner_dim, num_classes, pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int,
    ):
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        assert padding_idx is not None
        num_embeddings += padding_idx + 1  # WHY?
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        if use_cache:  # the position is our current step in the decoded sequence
            pos = int(self.padding_idx + input.size(1))
            positions = input.data.new(1, 1).fill_(pos)
        else:
            positions = create_position_ids_from_input_ids(input, self.padding_idx)
        return super().forward(positions)


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _filter_out_falsey_values(tup) -> Tuple:
    """Remove entries that are None or [] from an iterable."""
    return tuple(x for x in tup if isinstance(x, torch.Tensor) or x)


# Public API
def _get_shape(t):
    return getattr(t, "shape", None)


@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.", BART_START_DOCSTRING,
)
class BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        kg_input=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        use_cache=False,
    ):

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None
        if encoder_outputs is None:
            if kg_input is not None:
                encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, kg_input=kg_input)
            else:
                encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        assert isinstance(encoder_outputs, tuple)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        # Attention and hidden_states will be [] or None if they aren't needed
        decoder_outputs: Tuple = _filter_out_falsey_values(decoder_outputs)
        assert isinstance(decoder_outputs[0], torch.Tensor)
        encoder_outputs: Tuple = _filter_out_falsey_values(encoder_outputs)
        return decoder_outputs + encoder_outputs

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly


@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.",
    BART_START_DOCSTRING + BART_GENERATION_EXAMPLE,
)
class BartForConditionalGeneration(PretrainedBartModel):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = BartModel(config)
        self.model = base_model
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens))
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        kg_input=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        lm_labels=None,
        use_cache=False,
        **unused
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
            with labels
            in ``[0, ..., config.vocab_size]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

            # Mask filling only works for bart-large
            from transformers import BartTokenizer, BartForConditionalGeneration
            tokenizer = BartTokenizer.from_pretrained('bart-large')
            TXT = "My friends are <mask> but they eat too many carbs."
            model = BartForConditionalGeneration.from_pretrained('bart-large')
            input_ids = tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']
            logits = model(input_ids)[0]
            masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            probs = logits[0, masked_index].softmax(dim=0)
            values, predictions = probs.topk(5)
            tokenizer.decode(predictions).split()
            # ['good', 'great', 'all', 'really', 'very']
        """
        outputs = self.model(
            input_ids,
            kg_input=kg_input,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # TODO(SS): do we need to ignore pad tokens in lm_labels?
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step, decoder_cached_states are empty
        if not past[1]:
            encoder_outputs, decoder_cached_states = past, None
        else:
            encoder_outputs, decoder_cached_states = past
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_cached_states": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_scores_for_generation(self, scores, cur_len, max_length):
        if cur_len == 1:
            self._force_token_ids_generation(scores, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(scores, self.config.eos_token_id)
        return scores

    def _force_token_ids_generation(self, scores, token_ids) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.config.vocab_size) if x not in token_ids],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        assert len(scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        ((enc_out, enc_mask), decoder_cached_states) = past
        reordered_past = []
        for layer_past in decoder_cached_states:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        new_enc_out = enc_out if enc_out is None else enc_out.index_select(0, beam_idx)
        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)

        past = ((new_enc_out, new_enc_mask), reordered_past)
        return past

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.model.shared)  # make it on the fly


@add_start_docstrings(
    """Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks. """,
    BART_START_DOCSTRING,
)
class BartForSequenceClassification(PretrainedBartModel):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.classification_head = BartClassificationHead(
            config.d_model, config.d_model, config.num_labels, config.classif_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BartConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
                Classification loss (cross entropy)
            logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the
                self-attention
                heads.

    Examples::

        from transformers import BartTokenizer, BartForSequenceClassification
        import torch

        tokenizer = BartTokenizer.from_pretrained('bart-large')
        model = BartForSequenceClassification.from_pretrained('bart-large')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute",
        add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
        )
        x = outputs[0]  # last hidden state
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
        logits = self.classification_head(sentence_representation)
        # Prepend logits
        outputs = (logits,) + outputs[1:]  # Add hidden states and attention if they are here
        if labels is not None:  # prepend loss to output,
            loss = F.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """Identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
            The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out[:, 0 : dim // 2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))  # This line breaks for odd n_pos
        out[:, dim // 2 :] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False
        return out

    @torch.no_grad()
    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)
