import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import to_2tuple
import numpy as np

torch.set_printoptions(threshold=150, edgeitems=10)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class AdapterSuper(nn.Module):
    def __init__(self,
                 embed_dims_in,
                 embed_dims_out,
                 reduction_dims,
                 drop_rate_adapter=0,
                 add_adapter_gate=False,
                 parallel_adapter = False,
                 sequential_adapter = False
                        ):
        super(AdapterSuper, self).__init__()
    
        self.embed_dims_in = embed_dims_in
        self.embed_dims_out = embed_dims_out

        # Follow visual prompt
        # self.super_reductuion_dim = int(self.embed_dims/8)

        # Follow towards unified
        self.super_reductuion_dim = reduction_dims

        self.dropout = nn.Dropout(p=drop_rate_adapter)

        # gate setting
        self.add_adapter_gate = add_adapter_gate

        # adapter type
        self.parallel_adapter = parallel_adapter
        self.sequential_adapter = sequential_adapter

        print('adapter',self.super_reductuion_dim)
        if self.super_reductuion_dim > 0:
            self.ln1 = nn.Linear(self.embed_dims_in, self.super_reductuion_dim)
            self.activate = QuickGELU()
            self.ln2 = nn.Linear(self.super_reductuion_dim, self.embed_dims_out)

            self.init_weights()
            
            if self.add_adapter_gate:
                self.adapter_gate = nn.Linear(self.embed_dims_in, 1)
                # initialize gate with mean 0 and std 0.02
                self.adapter_gate.weight.data.normal_(mean=0, std=0.02)
        
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)


        self.apply(_init_weights)


    def set_sample_config(self, sample_embed_dim):
        self.identity = False
        self.sample_embed_dim = sample_embed_dim
        if self.sample_embed_dim == 0:
            self.identity = True
        else:
            # import pdb;pdb.set_trace()
            self.sampled_weight_0 = self.ln1.weight[:self.sample_embed_dim,:]
            self.sampled_bias_0 =  self.ln1.bias[:self.sample_embed_dim]

            self.sampled_weight_1 = self.ln2.weight[:, :self.sample_embed_dim]
            self.sampled_bias_1 =  self.ln2.bias


    def forward(self, x, identity=None):
        if self.identity:
            # import pdb;pdb.set_trace()
            return x
            # return x + 0*self.sampled_weight_0 + 0*self.sampled_bias_0 + 0*self.sampled_weight_1 + 0*self.sampled_bias_1
        # import pdb;pdb.set_trace()
        out = F.linear(x, self.sampled_weight_0, self.sampled_bias_0)
        out = self.activate(out)
        out = self.dropout(out)
        out = F.linear(out, self.sampled_weight_1, self.sampled_bias_1)
        if self.add_adapter_gate:
            gate = torch.sigmoid(self.adapter_gate(x))
            gate = gate.mean(dim=1).unsqueeze(-1)
            print('adapter gate called', gate.shape, out.shape, gate.max().item(), gate.min().item(),gate.squeeze())
            out = gate * out
        if identity is None:
            identity = x

        if self.parallel_adapter:
            return out
        else:
            # TODO: try to remove this redundant identity for sequential adapter
            return identity + out
        # seq adapter
        # return identity + out
        # parallel adapter
        # return out

    def calc_sampled_param_num(self):
        if self.identity:
            return 0
        else:
            return  self.sampled_weight_0.numel() + self.sampled_bias_0.numel() + self.sampled_weight_1.numel() + self.sampled_bias_1.numel()

    # def get_complexity(self, sequence_length):
    #     total_flops = 0
    #     if self.sampled_bias is not None:
    #          total_flops += self.sampled_bias.size(0)
    #     total_flops += sequence_length * np.prod(self.sampled_weight.size())
    #     return total_flops