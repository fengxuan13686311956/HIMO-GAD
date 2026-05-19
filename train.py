import torch.nn as nn
import matplotlib.pyplot as plt
import umap

from sklearn.model_selection import train_test_split

from utils import *
import random
from pretrain import pretrain_mae, train_mae
import torch.nn.functional as F
import os
import argparse
from sklearn.preprocessing import QuantileTransformer

import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0]))
os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
device = torch.device("cuda")
parser = argparse.ArgumentParser(description='zero-shot by mae')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--unifeat', type=int, default=8)
parser.add_argument("--device", type=str, default="cuda")
#GAE
parser.add_argument("--max_epoch_mae", type=int, default=200)
parser.add_argument("--finetune_epoch", type=int, default=20)
parser.add_argument("--warmup_steps_mae", type=int, default=-1)
parser.add_argument("--num_heads_mae", type=int, default=4)
parser.add_argument("--num_out_heads_mae", type=int, default=1)
parser.add_argument("--num_layers_mae", type=int, default=3)
parser.add_argument("--num_hidden_mae", type=int, default=128)
parser.add_argument("--residual_mae", action="store_true", default=False)
parser.add_argument("--in_drop_mae", type=float, default=.2)
parser.add_argument("--attn_drop_mae", type=float, default=.1)
parser.add_argument("--norm_mae", type=str, default=None)
parser.add_argument("--lr_mae", type=float, default=0.001)
parser.add_argument("--weight_decay_mae", type=float, default=5e-4)
parser.add_argument("--negative_slope_mae", type=float, default=0.2)
parser.add_argument("--activation_mae", type=str, default="prelu")
parser.add_argument("--encoder_mae", type=str, default="gcn")
parser.add_argument("--decoder_mae", type=str, default="gcn")
parser.add_argument("--loss_fn_mae", type=str, default="sce")

parser.add_argument("--alpha_l_mae", type=float, default=3)
parser.add_argument("--beta", type=float, default=0.1)

parser.add_argument("--optimizer_mae", type=str, default="adam")
parser.add_argument("--max_epoch_f_mae", type=int, default=30)
parser.add_argument("--lr_f_mae", type=float, default=0.001)
parser.add_argument("--weight_decay_f_mae", type=float, default=0.0)
parser.add_argument("--linear_prob_mae", action="store_true", default=False)
parser.add_argument("--load_model_mae", action="store_true")
parser.add_argument("--save_model_mae", action="store_true")
parser.add_argument("--use_cfg_mae", action="store_true")
parser.add_argument("--in_dim_mae", type=int, default=8)
parser.add_argument("--concat_hidden_mae", action="store_true", default=False)
parser.add_argument("--deg4feat_mae", action="store_true", default=False, help="use node degree as input feature")
parser.add_argument("--layer_hidden", type=int, default=4)
parser.add_argument("--num_shot", type=int, default=10)
args = parser.parse_args()
args.device = device
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
#F、A、F、B都更好

# Load and preprocess data
#,,'questions''YelpChi','YelpHotel','YelpNYC','elliptic',,'tolokers','questions',,'YelpChi''ACM''Reddit','Flickr','BlogCatalog'
#'Facebook','citeseer','pubmed','cs','cora','photo','questions','tfinance'
traindatasets = ['Flickr','tolokers','YelpChi','ACM']
targdataset = ['cs_all']
adj_train = []
ano_label_train = []
feat_list_train = []
metrics_all = {}
for dataset in traindatasets:
    adj, ano_label,feat_list = loaddata(dataset, args, args.device)
    # metrics_ano, metrics_normal = analyze_node_trajectories(
    #     feat_list=feat_list,
    #     ano_label=ano_label,
    #     steps=4,
    #     num_samples=300,
    #     seed=args.seed,
    #     dataset_name=dataset,
    # )

    # train_adj, test_adj, train_features, test_features, train_ano_label, test_ano_label= split_dataset(adj, features, ano_label, test_size=.2, random_state=42)
    adj_train.append(adj)
    ano_label_train.append(ano_label)
    feat_list_train.append(feat_list)

for _ in range(1):
    #模型训练

    model,optimizer = pretrain_mae(args , adj_train ,feat_list_train)
    # #微调
    # print("tuneing")
    # features_shot = []
    # ano_label_shot = []
    # for dataset in targdataset:
    #     _, features, ano_label = loaddata(dataset, args, args.device)
    #     features = [x.cpu() for x in features]
    #     features_shot.append(features)
    #     ano_label_shot.append(ano_label)
    # shot_list = []
    # adj_shot_list = []
    # shot,adj_shot=few_shot(features_shot,ano_label_shot,args.num_shot,edge_build=True)
    # shot_list.append(shot)
    # adj_shot_list.append(adj_shot)
    #
    # model.train()
    # model = train_mae(model, shot_list, adj_shot_list, optimizer,args.finetune_epoch)
    #
    # model.eval()
    # print("tuned")

    # 推理
    print("inferencing")
    for dataset in targdataset:
        adj, ano_label,feat_list  = loaddata(dataset, args, args.device)
        adj=adj.cpu()
        features = [x.cpu() for x in feat_list]

        features_predicted,_,_ = model(features, adj)

        completion_message = completionsim(features_predicted, features)
        # false_positive, false_negative=compute_misclassifications(completion_message,ano_label)
        # print(false_positive, false_negative)
        completion_auc, completion_AP = evaluate(completion_message, ano_label)
        print(f'{dataset} AUC: {completion_auc:.4f} AP: {completion_AP:.4f}')
