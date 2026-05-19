from random import random
import matplotlib.cm as cm
import numpy as np
import networkx as nx
import scipy.sparse as sp
import pickle as pkl
import scipy.io as sio
import umap
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.decomposition import PCA

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

import dgl
import torch
import torch.nn as nn
from torch import optim as optim

from dot_gat import DotGAT
from gat import GAT
from gcn import GCN, create_norm
from gin import GIN

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = sparse_mx.shape
    return torch.sparse_coo_tensor(indices, values, shape)

def normalize_score(ano_score):
    ano_score = ((ano_score - np.min(ano_score)) / (np.max(ano_score) - np.min(ano_score)))
    return ano_score

def x_svd(data, out_dim):
    assert data.shape[-1] >= out_dim
    U, S, _ = torch.linalg.svd(data)
    newdata= torch.mm(U[:, :out_dim], torch.diag(S[:out_dim]))
    return newdata

def load_mat(dataset):

    data = sio.loadmat("./Datasets/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    if data in ['YelpChi', 'Facebook']:
        adj = normalize_adj(network)
    else:
        adj = normalize_adj(network + sp.eye(network.shape[0]))
    adj = sp.csr_matrix(adj)
    feat = sp.lil_matrix(attr)
    ano_labels = np.squeeze(np.array(label))

    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None
    return adj, feat, ano_labels, str_ano_labels, attr_ano_labels

def normalize_features(features, method='standard'):
    """
    归一化特征矩阵
    Args:
        features (torch.Tensor): 输入特征矩阵
        method (str): 归一化方法，可选值为 'standard'（标准归一化）或 'minmax'（归一化到 [0, 1] 范围）
    Returns:
        normalized_features (torch.Tensor): 归一化后的特征矩阵
    """
    if method == 'standard':
        # 标准归一化（均值为 0，方差为 1）
        mean = torch.mean(features, dim=0, keepdim=True)
        std = torch.std(features, dim=0, keepdim=True)
        normalized_features = (features - mean) / (std + 1e-8)  # 防止除零错误
    elif method == 'minmax':
        # Min-Max 归一化（缩放到 [0, 1] 范围）
        min_val = torch.min(features, dim=0, keepdim=True).values
        max_val = torch.max(features, dim=0, keepdim=True).values
        normalized_features = (features - min_val) / (max_val - min_val + 1e-8)  # 防止除零错误
    else:
        normalized_features=features

    return normalized_features

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def loaddata(dataset, args, device):

    # 原始数据加载

    adj, features, ano_label, str_ano_label, attr_ano_label = load_mat(dataset)

    # 邻接矩阵处理
    adj = adj.astype(np.float32)

    adj_dense = torch.FloatTensor(adj.todense())
    # 特征处理
    if dataset in ['Amazon', 'YelpChi', 'tolokers', 'tfinance']:
        features = preprocess_features(features)
    else:
        features = features.toarray()
    features = torch.FloatTensor(features)
    features = x_svd(features, args.unifeat)
    # print(features.shape)
    # print(adj_dense.shape)
    if dataset not in ['Amazon', 'tolokers', 'tfinance']:
        features = normalize_features(features, method='standard')

    # 数据类型设备处理
    adj_lil = lil_matrix(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj_lil.tocsr()).to_dense()
    features = features.to(device)
    adj = adj.to(device)

    feat_list = [features]
    for _ in range(args.layer_hidden-1):
        feat_list.append(torch.spmm(adj, feat_list[-1]))
    return adj, ano_label,feat_list

def completionsim(feature1, feature2,eps=1e-8):
    feature2 = torch.cat([feature2[0], feature2[1], feature2[2], feature2[3]], dim=1)
    feature1 = feature1 / (torch.norm(feature1, dim=-1, keepdim=True)+ eps)
    feature2 = feature2 / (torch.norm(feature2, dim=-1, keepdim=True)+ eps)
    dist = torch.sum(feature1*feature2, dim=1)
    dist = dist.detach().cpu().numpy()
    return dist

def evaluate(message, ano_label, str_ano_label=None, attr_ano_label=None):
    score = 1-normalize_score(message)
    auc = roc_auc_score(ano_label, score)
    AP = average_precision_score(ano_label, score, average='macro', pos_label=1, sample_weight=None)

    if str_ano_label is not None:
        sa_auc = roc_auc_score(str_ano_label, score)
        sa_AP = average_precision_score(str_ano_label, score, average='macro', pos_label=1, sample_weight=None)
        print('Structural: AUC: {:.4f} AP:{:.4f}'.format(sa_auc, sa_AP))
    if attr_ano_label is not None:
        aa_auc = roc_auc_score(attr_ano_label, score)
        aa_AP = average_precision_score(attr_ano_label, score, average='macro', pos_label=1, sample_weight=None)
        print('Context: AUC:{:.4f} AP:{:.4f}'.format(aa_auc, aa_AP))
    return auc, AP

def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod

def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


from matplotlib.patches import Circle, Ellipse


def few_shot(features, ano_label, num_shot, edge_build=False):
    """
    features: list of [views] -> each view is a list of tensors [N_i, d]
    ano_label: list of tensors [N_i] (B graphs)
    num_shot: int
    edge_build: whether to create fully-connected graph for shots
    """

    num_views = len(features)
    num_graphs = len(features[0])
    normal_indices = []

    for i in range(num_graphs):
        # 获取标签为0的节点索引
        for node_idx in np.where(ano_label[i].cpu().numpy() == 0)[0]:
            normal_indices.append((i, node_idx))

    if len(normal_indices) < num_shot:
        raise ValueError(f"正常节点总数不足（{len(normal_indices)}），无法选取 {num_shot} 个样本")

    # 随机选择 num_shot 个 (graph_idx, node_idx)
    selected_indices = torch.randperm(len(normal_indices))[:num_shot]
    selected = [normal_indices[i] for i in selected_indices]

    # 逐视图提取 shot 特征
    shot_features = []
    for view_idx in range(num_views):
        view = features[view_idx]
        view_shot = torch.stack(
            [view[graph_idx][node_idx] for graph_idx, node_idx in selected],
            dim=0
        )  # [num_shot, d]
        shot_features.append(view_shot)

    # 构建邻接矩阵
    if edge_build:
        shot_adj = torch.ones(num_shot, num_shot) - torch.eye(num_shot)
    else:
        shot_adj = torch.zeros(num_shot, num_shot)

    shot_adj = shot_adj.to(shot_features[0].device)

    return shot_features, shot_adj

def compute_misclassifications(message, ano_label, threshold=None):

    scores = np.array(1-normalize_score(message))
    labels = np.array(ano_label)

    if threshold is None:
        threshold = np.median(scores)

    # 根据阈值判断是否为异常
    predicted = (scores > threshold).astype(int)

    # 正常（0）被误判为异常（1）
    false_positive = np.sum((labels == 0) & (predicted == 1))

    # 异常（1）被误判为正常（0）
    false_negative = np.sum((labels == 1) & (predicted == 0))

    return false_positive, false_negative



def compute_f_low(adj, feat_list, lambda_smooth=0.01):
    """
    计算图结构的下层目标函数值 f_low，用于跨域适应度量
    参数:
        adj: 邻接矩阵 (N x N torch.Tensor)
        feat_list: 特征列表 [H⁽⁰⁾, H⁽¹⁾, ..., H⁽ᵏ⁾] (每项为 torch.Tensor)
        lambda_smooth: 平滑项权重 (默认0.01)
    返回:
        f_low_value: 计算得到的f_low值 (标量)
    """
    # 1. 计算重构损失项: Σ||H⁽ˡ⁾ - adj @ H⁽ˡ⁻¹⁾||²
    recon_loss = 0.0
    for l in range(1, len(feat_list)):
        h_pred = adj @ feat_list[l - 1]  # 邻接矩阵传播
        recon_loss += torch.norm(feat_list[l] - h_pred, p='fro').item() ** 2

    # 2. 计算平滑项: Tr(H⁽ᵏ⁾ᵀ L H⁽ᵏ⁾)
    L = compute_laplacian(adj)  # 计算归一化拉普拉斯矩阵
    last_rep = feat_list[-1]
    smooth_term = torch.trace(last_rep.T @ L @ last_rep).item()

    # 3. 综合f_low值
    return recon_loss + lambda_smooth * smooth_term


def compute_laplacian(adj):
    """
    计算归一化拉普拉斯矩阵 L = I - D^{-1/2} A D^{-1/2}
    参数:
        adj: 邻接矩阵 (N x N)
    返回:
        L: 归一化拉普拉斯矩阵
    """
    # 计算度矩阵
    deg = torch.diag(torch.sum(adj, dim=1))
    # 处理孤立节点 (避免除零)
    deg_inv_sqrt = torch.zeros_like(deg)
    non_zero = torch.diag(deg) != 0
    deg_inv_sqrt[non_zero, non_zero] = torch.sqrt(1.0 / torch.diag(deg)[non_zero])

    return torch.eye(adj.size(0), device=adj.device) - deg_inv_sqrt @ adj @ deg_inv_sqrt


def compute_cross_domain_metric(src_adj, src_feat_list, tgt_adj, tgt_feat_list):
    """
    计算跨域适应度量指标
    参数:
        src_adj, src_feat_list: 源域邻接矩阵和特征列表
        tgt_adj, tgt_feat_list: 目标域邻接矩阵和特征列表
    返回:
        metric_dict: 包含跨域度量的字典
    """
    # 计算各域f_low
    f_src = compute_f_low(src_adj, src_feat_list)
    f_tgt = compute_f_low(tgt_adj, tgt_feat_list)

    # 计算域间差异指标
    domain_gap = f_tgt - f_src
    relative_gap = domain_gap / f_src

    return {
        'f_low_source': f_src,
        'f_low_target': f_tgt,
        'domain_gap_abs': domain_gap,
        'domain_gap_rel': relative_gap,
        'is_transferable': relative_gap < 0.5  # 可迁移性阈值
    }

import torch
#
# def graph_perturb(A, X, mode="both", level=1, seed=None):
#     """
#     对图数据 (邻接矩阵 A, 节点特征 X) 进行加噪 (PyTorch 版本)
#     - A: 邻接矩阵 (N, N)，torch.Tensor
#     - X: 节点特征矩阵 (N, d)，torch.Tensor
#     - mode: {"adj", "feat", "both"}
#     - level: 噪声等级 {1,2,3,4,5}
#     - seed: 随机种子
#     """
#     if seed is not None:
#         torch.manual_seed(seed)
#
#     device = A.device
#     N, d = X.shape
#     A_noisy, X_noisy = A.clone().detach(), X.clone().detach().float()
#
#     # 定义不同等级的噪声参数
#     noise_params = {
#         1: {"add_ratio": 0.01, "remove_ratio": 0.01, "sigma": 0.05, "mask_ratio": 0.05},
#         2: {"add_ratio": 0.05, "remove_ratio": 0.05, "sigma": 0.1,  "mask_ratio": 0.1},
#         3: {"add_ratio": 0.1,  "remove_ratio": 0.1,  "sigma": 0.2,  "mask_ratio": 0.2},
#         4: {"add_ratio": 0.2,  "remove_ratio": 0.2,  "sigma": 0.3,  "mask_ratio": 0.3},
#         5: {"add_ratio": 0.3,  "remove_ratio": 0.3,  "sigma": 0.4,  "mask_ratio": 0.4},
#     }
#     params = noise_params.get(level, noise_params[max(noise_params.keys())])
#
#     # -------- 邻接矩阵扰动 --------
#     if mode in ("adj", "both"):
#         triu_indices = torch.triu_indices(N, N, offset=1, device=A_noisy.device)
#         edge_mask = A_noisy[triu_indices[0], triu_indices[1]] == 1
#         non_edge_mask = ~edge_mask
#
#         edges = triu_indices[:, edge_mask]
#         non_edges = triu_indices[:, non_edge_mask]
#
#         # 删除边
#         num_remove = int(edges.size(1) * params["remove_ratio"])
#         if num_remove > 0:
#             remove_idx = torch.randperm(edges.size(1), device=A_noisy.device)[:num_remove]
#             i, j = edges[:, remove_idx]
#             A_noisy[i, j] = 0
#             A_noisy[j, i] = 0
#
#         # 添加边
#         num_add = int(non_edges.size(1) * params["add_ratio"])
#         if num_add > 0:
#             add_idx = torch.randperm(non_edges.size(1), device=A_noisy.device)[:num_add]
#             i, j = non_edges[:, add_idx]
#             A_noisy[i, j] = 1
#             A_noisy[j, i] = 1
#
#     # -------- 特征矩阵扰动 --------
#     if mode in ("feat", "both"):
#         # 高斯噪声
#         noise = torch.randn_like(X_noisy) * params["sigma"]
#         X_noisy = X_noisy + noise
#
#         # 随机掩码
#         if params["mask_ratio"] > 0:
#             mask = torch.rand_like(X_noisy) < params["mask_ratio"]
#             X_noisy = X_noisy.masked_fill(mask, 0.0)
#
#     return A_noisy.to(device), X_noisy.to(device)
#
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import mannwhitneyu
#
# def compute_traj_metrics(node_feats):
#     """
#     输入: node_feats [steps, dim] (单个节点在每一步的特征向量)
#     输出: dict of metrics
#     """
#     steps = len(node_feats)
#     diffs = node_feats[1:] - node_feats[:-1]  # 每一步差分
#     step_norms = np.linalg.norm(diffs, axis=1)
#
#     # 总长度
#     L = np.sum(step_norms)
#
#     # 起点到终点位移
#     D = np.linalg.norm(node_feats[-1] - node_feats[0])
#
#     # 平均/最大单步变化
#     mean_step = np.mean(step_norms)
#     max_step = np.max(step_norms)
#
#     # 方向一致性
#     unit_vecs = diffs / (step_norms[:, None] + 1e-8)
#     mean_dir = np.mean(unit_vecs, axis=0)
#     mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-8)
#     coherence = np.mean([np.dot(u, mean_dir) for u in unit_vecs])
#
#     # 轨迹弯曲度
#     curvatures = []
#     for i in range(1, len(unit_vecs)):
#         curvatures.append(np.arccos(np.clip(np.dot(unit_vecs[i-1], unit_vecs[i]), -1, 1)))
#     curvature = np.mean(curvatures) if curvatures else 0.0
#
#     return {
#         "Length": L,
#         "Displacement": D,
#         "MeanStep": mean_step,
#         "MaxStep": max_step,
#         "Coherence": coherence,
#         "Curvature": curvature
#     }
#
# def analyze_node_trajectories(feat_list, ano_label, dataset_name,
#                               steps=4, num_samples=300, seed=42):
#     np.random.seed(seed)
#
#     ano_nodes = np.where(ano_label == 1)[0]
#     normal_nodes = np.where(ano_label == 0)[0]
#
#     num_ano = min(num_samples, len(ano_nodes))
#     num_normal = min(num_samples, len(normal_nodes))
#
#     sampled_ano = np.random.choice(ano_nodes, num_ano, replace=False)
#     sampled_normal = np.random.choice(normal_nodes, num_normal, replace=False)
#
#     metrics_ano, metrics_normal = [], []
#
#     for node in sampled_ano:
#         feats = np.stack([feat_list[s][node].detach().cpu().numpy() for s in range(steps)])
#         metrics_ano.append(compute_traj_metrics(feats))
#
#     for node in sampled_normal:
#         feats = np.stack([feat_list[s][node].detach().cpu().numpy() for s in range(steps)])
#         metrics_normal.append(compute_traj_metrics(feats))
#
#     # 转换成 numpy
#     metrics_ano = {k: [m[k] for m in metrics_ano] for k in metrics_ano[0].keys()}
#     metrics_normal = {k: [m[k] for m in metrics_normal] for k in metrics_normal[0].keys()}
#
#     # 绘图 + 统计检验
#     for metric in metrics_ano.keys():
#         data_ano = metrics_ano[metric]
#         data_normal = metrics_normal[metric]
#
#         # Mann-Whitney U test
#         stat, p = mannwhitneyu(data_ano, data_normal, alternative="two-sided")
#
#         plt.figure(figsize=(6, 4))
#         sns.boxplot(data=[data_normal, data_ano], showfliers=False)  # 🚫 不显示离群点
#         plt.xticks([0, 1], ["Normal", "Anomalous"])
#         plt.title(f"{metric} of {dataset_name} (p={p:.3e})")
#         plt.ylabel(metric)
#         plt.show()
#
#     return metrics_ano, metrics_normal
#
#
