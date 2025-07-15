# _*_ coding : utf-8 _*_
# @Time : 2025/4/22 13:52
# @Author : wfr
# @file : loss
# @Project : MultilayerUAVLP
import torch


def get_reconstruct_loss(pred_adj_list, true_adj_list, beta):
    """
    :param pred_adj_list: list of 3 层预测的邻接矩阵
    :param true_adj_list: list of 3 层真实邻接矩阵
    :param beta: 控制存在的边的损失权重
    :return:
    """
    loss = 0
    for pred_adj, true_adj in zip(pred_adj_list, true_adj_list):
        weight = true_adj * (beta - 1) + 1
        loss += torch.mean(torch.sum(weight * torch.square(true_adj - pred_adj), dim=1), dim=-1)

    return loss


def get_embedding_consistency_loss(z_history, edge_index_seq_list, alpha, beta):
    """
    :param z_history: list of 3 lists, each是 T 个 (N, F) 的节点嵌入
    :param edge_index_seq_list: list of T 个时间步，每个是 [edge_index_l1, l2, l3]
    :param alpha: 层内邻居嵌入相似损失的权重
    :param beta: 层间节点嵌入相似损失的权重
    :return: scalar loss
    """
    T = len(edge_index_seq_list)
    num_layers = len(z_history)
    intra_loss = 0.0
    inter_loss = 0.0

    for t in range(T):
        # 层内邻居一致性损失
        for l in range(num_layers):
            z = z_history[l][t]  # (N, F)
            edge_index = edge_index_seq_list[t][l]  # (2, E)
            src, tgt = edge_index
            diff = z[src] - z[tgt]  # (E, F)
            intra_loss += torch.mean((diff ** 2))

        # 层间节点一致性损失（每对层）
        for i in range(num_layers):
            for j in range(i + 1, num_layers):
                zi = z_history[i][t]  # (N, F)
                zj = z_history[j][t]
                inter_loss += torch.mean((zi - zj) ** 2)

    intra_loss = intra_loss / (T * num_layers)
    inter_loss = inter_loss / (T * (num_layers * (num_layers - 1) / 2))
    return alpha * intra_loss + beta * inter_loss


def get_total_loss(pred_adj_list, true_adj_list, z_history, edge_index_seq_list,
                   recon_beta=5.0, intra_alpha=1.0, inter_beta=1.0):
    """
    总损失：邻接矩阵重构损失 + 层内邻居一致性 + 层间节点一致性
    :param pred_adj_list: list of 3 个 (N, N) 的预测邻接矩阵
    :param true_adj_list: list of 3 个 (N, N) 的真实邻接矩阵
    :param z_history: list of 3 个 list，每个包含 T 个 (N, F) 的嵌入
    :param edge_index_seq_list: list of T 个时间步，每个是 [edge_index_layer1, layer2, layer3]
    :param recon_beta: 存在边的重构损失加权
    :param intra_alpha: 层内邻居一致性损失的权重
    :param inter_beta: 层间节点一致性损失的权重
    :return: total_loss
    """
    # ==== 重构损失 ====
    recon_loss = 0.0
    for pred_adj, true_adj in zip(pred_adj_list, true_adj_list):
        weight = true_adj * (recon_beta - 1) + 1  # 对存在边加权
        recon_loss += torch.mean(torch.sum(weight * (true_adj - pred_adj) ** 2, dim=1), dim=0)
    recon_loss = recon_loss / len(pred_adj_list)

    # ==== 嵌入一致性损失 ====
    T = len(edge_index_seq_list)
    num_layers = len(z_history)
    intra_loss = 0.0
    inter_loss = 0.0

    for t in range(T):
        for l in range(num_layers):
            z = z_history[l][t]  # (N, F)
            edge_index = edge_index_seq_list[t][l]
            src, tgt = edge_index
            diff = z[src] - z[tgt]
            intra_loss += torch.mean((diff ** 2))

        for i in range(num_layers):
            for j in range(i + 1, num_layers):
                zi = z_history[i][t]
                zj = z_history[j][t]
                inter_loss += torch.mean((zi - zj) ** 2)

    intra_loss = intra_loss / (T * num_layers)
    inter_loss = inter_loss / (T * (num_layers * (num_layers - 1) / 2))

    # ==== 总损失 ====
    total_loss = recon_loss + intra_alpha * intra_loss + inter_beta * inter_loss

    return total_loss
