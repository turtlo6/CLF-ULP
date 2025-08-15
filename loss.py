import torch


def get_reconstruct_loss(pred_adj_list, true_adj_list, beta):
    loss = 0
    for pred_adj, true_adj in zip(pred_adj_list, true_adj_list):
        weight = true_adj * (beta - 1) + 1
        loss += torch.mean(torch.sum(weight * torch.square(true_adj - pred_adj), dim=1), dim=-1)

    return loss


def get_embedding_consistency_loss(z_history, edge_index_seq_list, alpha, beta):
    T = len(edge_index_seq_list)
    num_layers = len(z_history)
    intra_loss = 0.0
    inter_loss = 0.0

    for t in range(T):
        # Intra layer neighbor consistency loss
        for l in range(num_layers):
            z = z_history[l][t]  # (N, F)
            edge_index = edge_index_seq_list[t][l]  # (2, E)
            src, tgt = edge_index
            diff = z[src] - z[tgt]  # (E, F)
            intra_loss += torch.mean((diff ** 2))

        # Inter layer node consistency loss (per pair of layers)
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
    # ==== Reconstruction loss ====
    recon_loss = 0.0
    for pred_adj, true_adj in zip(pred_adj_list, true_adj_list):
        weight = true_adj * (recon_beta - 1) + 1
        recon_loss += torch.mean(torch.sum(weight * (true_adj - pred_adj) ** 2, dim=1), dim=0)
    recon_loss = recon_loss / len(pred_adj_list)

    # ==== Embedded consistency loss ====
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

    # ==== total loss ====
    total_loss = recon_loss + intra_alpha * intra_loss + inter_beta * inter_loss

    return total_loss
