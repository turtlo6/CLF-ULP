# _*_ coding : utf-8 _*_
# @Time : 2025/4/15 13:38
# @Author : wfr
# @file : utils
# @Project : MultilayerUAVLP

import numpy as np
import torch
import os
from sklearn.metrics import roc_auc_score, average_precision_score

import os
import numpy as np
import torch


def get_multilayer_snapshots(path, num_layers, layer_prefix):
    """
    读取多个层的edge_seq.npy文件，并转换为PyTorch的edge_index格式。

    参数:
        path (str): 包含多个 *_edge_seq.npy 文件的文件夹路径
        num_layers (int): 层数（例如3表示有 GM_A、GM_B、GM_C 三层）
        layer_prefix (str): 层前缀（例如 "GM" 或 "RW"）

    返回:
        edge_index_sequence (List[List[Tensor]]):
            edge_index_sequence[t][l] 表示第 t 个时间步第 l 层的边张量
    """
    edge_index_sequence = []

    # 根据传入的前缀动态生成文件名
    layer_names = [f'{layer_prefix}_A', f'{layer_prefix}_B', f'{layer_prefix}_C'][:num_layers]

    # 读取所有层的数据
    all_layers_edge_seq = []
    for name in layer_names:
        file_path = os.path.join(path, f'{name}_edge_seq.npy')
        edge_seq = np.load(file_path, allow_pickle=True)
        all_layers_edge_seq.append(edge_seq)

    num_timesteps = len(all_layers_edge_seq[0])
    for t in range(num_timesteps):
        snapshot_at_t = []
        for l in range(num_layers):
            edges = all_layers_edge_seq[l][t]
            if len(edges) == 0:
                # 若无边则构造空edge_index张量
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_array = np.array(edges).T  # 转置成 [2, num_edges]
                edge_index = torch.tensor(edge_array, dtype=torch.long)
            snapshot_at_t.append(edge_index)
        edge_index_sequence.append(snapshot_at_t)

    return edge_index_sequence


def edge_index_to_adj(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def link_prediction_evaluator(pred_score, true_label):
    N = int(len(pred_score) ** 0.5)
    mask = ~torch.eye(N, dtype=torch.bool, device=pred_score.device).view(-1)

    prob = torch.sigmoid(pred_score[mask]).detach().cpu().numpy()
    labels = true_label[mask].detach().cpu().numpy()

    try:
        auc = roc_auc_score(labels, prob)
        ap = average_precision_score(labels, prob)
    except ValueError:
        auc = float('nan')
        ap = float('nan')

    return auc, ap
