import torch.nn as nn
import torch.nn.functional as F


from utils import *
from torch import Tensor
from torch.nn.modules.module import Module
from torch_geometric.nn.inits import glorot
from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
from loss_func import sce_loss


def build_model(args):
    layer_hidden=args.layer_hidden
    num_heads = args.num_heads_mae  # 4
    num_out_heads = args.num_out_heads_mae  # 1
    num_hidden = args.num_hidden_mae  # 256
    num_layers = args.num_layers_mae  # 2
    residual = args.residual_mae  # False
    attn_drop = args.attn_drop_mae  # 0.1
    in_drop = args.in_drop_mae  # 0.2
    norm = args.norm_mae  # None
    negative_slope = args.negative_slope_mae  # 0.2
    encoder_type = args.encoder_mae  # "gat"
    decoder_type = args.decoder_mae  # "gat"
    activation = args.activation_mae  # "prelu"
    loss_fn = args.loss_fn_mae  # "sce"
    alpha_l = args.alpha_l_mae  # 2
    concat_hidden = args.concat_hidden_mae  # False
    in_dim = args.unifeat  # 8


    model = PreModel(
        layer_hidden=layer_hidden,
        in_dim=in_dim,
        num_hidden=num_hidden,
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        norm=norm,
        loss_fn=loss_fn,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
    )
    return model

class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(k)])
        self.relu = nn.ReLU()

    def forward(self, tokens):
        # tokens 是一个长度为 k 的列表，每个元素 shape 为 [N, D]
        for i in range(len(tokens)):
            tokens[i] = self.relu(self.fc[i](tokens[i]))  # [N, D_in] -> [N, D_out]
        return tokens

class PreModel(nn.Module):
    def __init__(
            self,
            layer_hidden: int,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            alpha_l: float = 2,
            concat_hidden: bool = False,
    ):
        super(PreModel, self).__init__()

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1
        self.linner_layer = LinearLayer(in_dim, in_dim, k=4)
        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden
        self.GRU=nn.GRU(in_dim, in_dim, bias=True)
        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        self.propagate_layers = 5
        self.prelu = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(in_dim)
        in_dim=in_dim*layer_hidden
        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            activation=activation,
            dropout=feat_drop,
            nhead_out=dec_num_hidden,
            residual=residual,
            norm=norm,
            attn_drop=attn_drop,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=2,
            nhead=enc_nhead,
            activation=activation,
            dropout=feat_drop,
            nhead_out=dec_num_hidden,
            residual=residual,
            norm=norm,
            attn_drop=attn_drop,
        )

    def compute_graph_smoothness_loss(self,middle_feat, adj):
        degree = torch.diag(adj.sum(dim=1))
        laplacian = degree - adj
        smoothness_loss = torch.trace(middle_feat.T @ laplacian @ middle_feat)
        smoothness_loss = smoothness_loss / middle_feat.shape[0]
        return smoothness_loss

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def forward(self,feat_list,adj):
        inputs = [x.clone() for x in feat_list]
        inputs=self.linner_layer(inputs)
        # ---- attribute reconstruction ----
        input1,input2,input3,input4=inputs
        num_dims = input1.size()[1]
        num_nodes = input1.size()[0]
        x1s = torch.zeros(num_nodes, num_dims).cuda()
        x2s = torch.zeros(num_nodes, num_dims).cuda()
        x3s = torch.zeros(num_nodes, num_dims).cuda()
        x4s = torch.zeros(num_nodes, num_dims).cuda()
        for passing_round in range(self.propagate_layers):

            attention1 = (input2+input3+input4)/3 # message passing with concat operation
            attention2 = (input1+input3+input4)/3
            attention3 = (input1+input2+input4)/3
            attention4 = (input1+input2+input3)/3
            X1 = torch.stack([ attention1,input1], dim=0)
            X2 = torch.stack([attention2, input2], dim=0)
            X3 = torch.stack([attention3, input3], dim=0)
            X4 = torch.stack([ attention4,input4], dim=0)

            out1,h_v1 = self.GRU(X1)
            out2,h_v2 = self.GRU(X2)
            out3,h_v3 = self.GRU(X3)
            out4,h_v4 = self.GRU(X4)

            input1 = h_v1.clone().squeeze(0)
            input2 = h_v2.clone().squeeze(0)
            input3 = h_v3.clone().squeeze(0)
            input4 = h_v4.clone().squeeze(0)


            # print('attention size:', attention3[None].contiguous().size(), exemplar.size())
            if passing_round == self.propagate_layers - 1:
                x1s = self.my_fcn(input1,feat_list[0])
                x2s = self.my_fcn(input2,feat_list[1])
                x3s = self.my_fcn(input3,feat_list[2])
                x4s = self.my_fcn(input4,feat_list[3])
        input1r, input2r, input3r, input4r = inputs
        x1s=(input1r+x1s)/2
        x2s=(input2r+x2s)/2
        x3s=(input3r+x3s)/2
        x4s=(input4r+x4s)/2
        out = torch.cat([x1s,x2s,x3s,x4s], dim=1)
        # out = torch.cat([input1, input2, input3, input4], dim=1)

        x=out.clone()

        x.to("cuda")
        middle_feat= self.encoder(x,adj)
        middle_feat = self.encoder_to_decoder(middle_feat)
        # print(middle_feat[:1][0])
        # ---- attribute reconstruction ----

        # if self._decoder_type not in ("mlp", "linear"):
        #     # * remask, re-mask
        #     middle_feat[mask_nodes] = 0
        # if self._decoder_type in ("mlp", "liear"):
        #     recon = self.decoder(middle_feat)
        # else:
        loss_smoo=self.compute_graph_smoothness_loss(middle_feat,adj)
        recon= self.decoder(middle_feat,adj)

        return recon,loss_smoo,out

    def my_fcn(self, input1_att, exemplar):
        input1 = input1_att + exemplar
        input1 = self.bn1(input1)
        input1 = self.prelu(input1)
        return input1