import scipy.io as sio
import scipy.sparse as sp
import numpy as np
from sklearn.decomposition import TruncatedSVD
import os
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
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
def svd_reduce_features(mat_file, out_file, dim=8):
    # 加载原始mat数据
    data = sio.loadmat(mat_file)

    label = data['Label'] if 'Label' in data else data['gnd']
    attr = data['Attributes'] if 'Attributes' in data else data['X']
    network = data['Network'] if 'Network' in data else data['A']

    # 处理异常标签
    str_ano_label = data['str_anomaly_label'] if 'str_anomaly_label' in data else None
    attr_ano_label = data['attr_anomaly_label'] if 'attr_anomaly_label' in data else None

    # 特征矩阵转稠密并降维
    if sp.issparse(attr):
        attr = attr.toarray()
    svd = TruncatedSVD(n_components=dim, random_state=42)
    attr_svd = svd.fit_transform(attr)

    # 构建保存字典
    save_dict = {
        'Label': label,
        'Network': network,
        'Attributes': attr_svd
    }

    if str_ano_label is not None:
        save_dict['str_anomaly_label'] = str_ano_label
    if attr_ano_label is not None:
        save_dict['attr_anomaly_label'] = attr_ano_label

    # 保存到新mat文件
    sio.savemat(out_file, save_dict)
    print(f"Saved: {out_file}")


# 要处理的数据集名称
datasets = ['Flickr', 'tolokers', 'YelpChi', 'ACM', 'Facebook',
            'citeseer', 'pubmed', 'cs', 'cora', 'photo', 'questions', 'tfinance']

# 数据集路径（假设都放在 ./Datasets 目录）
in_dir = './Datasets'
out_dir = './Datasets_svd'  # 也可以换成别的保存目录

# 执行降维保存
for dataset in datasets:
    input_path = os.path.join(in_dir, f'{dataset}.mat')
    output_path = os.path.join(out_dir, f'{dataset}_svd.mat')
    svd_reduce_features(input_path, output_path, dim=8)