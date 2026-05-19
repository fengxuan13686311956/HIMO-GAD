import torch
import torch.nn.functional as F


# def sce_loss(feat, recon,  k_pct: float = 0.4, shrink_grad_ratio: float = 0.2,alpha=3):
#     feat = torch.cat([feat[0], feat[1], feat[2], feat[3]], dim=1)
#     # feat = F.normalize(feat, p=2, dim=-1)
#     # recon = F.normalize(recon, p=2, dim=-1)
#     #
#     # loss = (1 - (feat * recon).sum(dim=-1)).pow_(alpha)
#     #
#     # loss = loss.mean()
#     N, D = feat.shape
#
#     feat_norm = F.normalize(feat, dim=-1)
#     recon_adj = recon.clone().requires_grad_()
#
#     recon_norm = F.normalize(recon_adj, dim=-1)
#     cos_dist = 1 - (feat_norm * recon_norm).sum(dim=-1)
#
#     k = max(1, int(N * k_pct))
#     th = torch.kthvalue(cos_dist, k).values
#     easy_mask = cos_dist <= th  # [N]
#
#     orig = recon_adj[easy_mask]
#     # 保留梯度：注意不要 detach 过多
#     recon_adj[easy_mask] = orig.detach() * shrink_grad_ratio + orig * (1 - shrink_grad_ratio)
#
#     feat_norm2 = F.normalize(feat, dim=-1)
#     recon_norm2 = F.normalize(recon_adj, dim=-1)
#     target = torch.ones(N, device=feat.device)
#     loss = F.cosine_embedding_loss(feat_norm2, recon_norm2, target, reduction='mean')
#     return loss
def sce_loss(feat, recon, k_pct: float = 0.1, shrink_grad_ratio: float = 0.0, alpha=3):
    feat = torch.cat([feat[0], feat[1], feat[2], feat[3]], dim=1)
    N, D = feat.shape

    feat_norm = F.normalize(feat, dim=-1)
    recon_adj = recon.clone().requires_grad_()

    recon_norm = F.normalize(recon_adj, dim=-1)
    cos_dist = 1 - (feat_norm * recon_norm).sum(dim=-1)  # cosine distance

    k = max(1, int(N * k_pct))
    th = torch.kthvalue(cos_dist, N - k + 1).values  # 取最大的 k% 中的最小值（即第 N - k + 1 小的值）
    hard_mask = cos_dist >= th  # 保留最难重建的 k%，其余样本参与梯度更新

    orig = recon_adj[hard_mask]
    recon_adj[hard_mask] = orig.detach() * shrink_grad_ratio + orig * (1 - shrink_grad_ratio)

    feat_norm2 = F.normalize(feat, dim=-1)
    recon_norm2 = F.normalize(recon_adj, dim=-1)
    target = torch.ones(N, device=feat.device)
    loss = F.cosine_embedding_loss(feat_norm2, recon_norm2, target, reduction='mean')

    return loss

def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss