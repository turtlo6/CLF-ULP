import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np

from model.DynMul import DynamicMultiLayerLP
from utils import *
from loss import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)  # 设置PyTorch随机种子
    torch.cuda.manual_seed_all(seed)  # 设置PyTorch的CUDA随机种子（如果GPU可用）
    np.random.seed(seed)  # 设置NumPy的随机种子
    random.seed(seed)  # 设置Python内置random库的随机种子
    torch.backends.cudnn.deterministic = True  # 设置使用CUDA加速时保证结果一致性


setup_seed(0)

data_name = 'MG_12'
num_nodes = 100
num_layers = 3  # 网络层数
history_len = 10
num_snaps = 80
num_train_snaps = 60
num_test_snaps = 20
num_epochs = 500
early_stop_epochs = 50
learning_rate = 1e-4
# ===================
in_dim = num_nodes
hidden_dim = 256
embed_dim = 128
gat_layer_num = 2
lstm_hidden_dim = 256
# ===================
recon_beta = 10
intra_alpha = 1
inter_beta = 5
# ===================
embedding_save_path = 'emb/emb_%s_%s' % (data_name, learning_rate)
num_node_embedding_save = 20

# Load multilayer dynamic snapshots
data_path = "data"
# edge_index_sequence = List[List[Tensor]]
# edge_index_sequence[t][l] = 第 t 个时间步，第 l 层图的边（edge_index）张量
edge_index_sequence = get_multilayer_snapshots(path=data_path, num_layers=num_layers, layer_prefix=data_name)

# Create train/test target links
# train_edges, test_edges = make_train_val_test_edges(edge_index_sequence, history_len)

# Initialize model
model = DynamicMultiLayerLP(in_dim, hidden_dim, embed_dim, gat_layer_num, lstm_hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train():
    model.train()
    total_loss = 0
    count = 0

    for t in tqdm(range(history_len, num_train_snaps)):
        optimizer.zero_grad()

        # 构造输入序列
        history_edge_index_seq = edge_index_sequence[t - history_len:t]
        history_edge_index_seq = [[e.to(device) for e in snapshot] for snapshot in history_edge_index_seq]

        # 模型输出：List of [N, N] 邻接矩阵
        pred_adj_list, embedding_list = model(history_edge_index_seq)

        # 构造真实邻接矩阵（每一层）
        true_edge_index_list = edge_index_sequence[t]
        true_adj_list = [edge_index_to_adj(edge_idx.to(device), num_nodes) for edge_idx in true_edge_index_list]

        loss = get_total_loss(pred_adj_list, true_adj_list, embedding_list, history_edge_index_seq,
                              recon_beta, intra_alpha, inter_beta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / count


# Step 5: Evaluation function
@torch.no_grad()
def test(save_embedding=False, save_path=embedding_save_path, num_save=num_node_embedding_save):
    model.eval()
    auc_list = []
    ap_list = []

    # 随机选择 num_save 个时间步用于保存嵌入
    save_timesteps = random.sample(range(num_snaps - num_test_snaps, num_snaps), num_save)
    saved_embeddings = {}

    for t in range(num_snaps - num_test_snaps, num_snaps):
        history_edge_index_seq = edge_index_sequence[t - history_len:t]
        history_edge_index_seq = [[e.to(device) for e in snapshot] for snapshot in history_edge_index_seq]

        pred_adj_list, embedding_list = model(history_edge_index_seq)
        true_edge_index_list = edge_index_sequence[t]
        true_adj_list = [edge_index_to_adj(edge_idx.to(device), num_nodes) for edge_idx in true_edge_index_list]

        for pred_adj, true_adj in zip(pred_adj_list, true_adj_list):
            pred_score = pred_adj.view(-1)
            true_label = true_adj.view(-1)

            auc, ap = link_prediction_evaluator(pred_score, true_label)
            auc_list.append(auc)
            ap_list.append(ap)

        if save_embedding and t in save_timesteps:
            # flatten embedding_list
            flat_embeddings = []
            for emb in embedding_list:
                if isinstance(emb, list):
                    flat_embeddings.extend(emb)
                else:
                    flat_embeddings.append(emb)

            emb_array = torch.cat(flat_embeddings, dim=0).cpu().numpy()  # shape: [num_layers * N, D]
            saved_embeddings[f"t{t}"] = emb_array

    if save_embedding and saved_embeddings:
        np.savez(save_path + '.npz', **saved_embeddings)

    return sum(auc_list) / len(auc_list), sum(ap_list) / len(ap_list)


# Step 6: Run training and validation
best_AUC = 0
best_AUPRC = 0
no_improve_epochs = 0
for epoch in range(1, num_epochs + 1):
    loss = train()
    cur_auc, cur_auprc = test()
    print(
        f"Epoch {epoch:02d}, Loss: {loss:.4f}, Test AUC: {cur_auc:.4f}, AP: {cur_auprc:.4f}, Best AUC: {best_AUC:.4f}, AP: {best_AUPRC:.4f}")

    if cur_auc <= best_AUC:
        no_improve_epochs += 1
        if no_improve_epochs >= early_stop_epochs:
            break
    else:
        best_AUC = cur_auc
        best_AUPRC = cur_auprc
        no_improve_epochs = 0

    f_input = open(
        'res/%s_lr%s.txt' % (data_name, learning_rate),
        'a+')
    f_input.write(
        f"Epoch {epoch:02d}, Loss: {loss:.4f}, Test AUC: {cur_auc:.4f}, AP: {cur_auprc:.4f}, Best AUC: {best_AUC:.4f}, AP: {best_AUPRC:.4f}\n")
    f_input.close()

# Step 7: Final test
test_auc, test_ap = test(True)
print(f"Final Test AUC: {test_auc:.4f}, AP: {test_ap:.4f}")