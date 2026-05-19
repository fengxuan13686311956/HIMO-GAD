import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model import build_model


from utils import *

def pretrain_mae(args , adj_train ,feat_list_train):

    # 打印预训练信息
    print("pretraining start")

    #model定义
    model = build_model(args)
    model.to(args.device)
    optimizer = create_optimizer(args.optimizer_mae, model, args.lr_mae, args.weight_decay_mae)
    print("model built")
    #model训练
    print("model pretrain:")
    model = train_mae(model,feat_list_train,adj_train, optimizer, args.max_epoch_mae)
    model = model.cpu()
    # model = model.to(args.device)
    model.eval()
    print("model pretrained")
    return model,optimizer

def train_mae(model, feat_list_train,adjs,optimizer, max_epoch_mae):
    model.train()
    max_epoch=max_epoch_mae
    epoch_iter = tqdm(range(max_epoch))
    a=0.04
    for epoch in epoch_iter:
        for feat_list,adj in zip(feat_list_train,adjs):


            recon,smoo_loss,out= model(feat_list,adj)
            loss=model.criterion(feat_list,recon)
            loss+=a*smoo_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_iter.set_description(f"Epoch {epoch} | train_loss: {loss:.4f}")

    return model
