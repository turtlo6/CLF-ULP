import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# ===== CrossGAT3Layer（结构建模） =====
class CrossGAT3Layer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, heads=2, dropout=0.6):
        super(CrossGAT3Layer, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.gat_layers = nn.ModuleList()
        self.cross_attn = nn.ModuleList()

        # 输入层GAT
        self.gat_layers.append(nn.ModuleList([
            GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
            for _ in range(3)
        ]))

        # 中间层
        for _ in range(num_layers - 2):
            self.gat_layers.append(nn.ModuleList([
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout)
                for _ in range(3)
            ]))

        # 输出层GAT
        self.gat_layers.append(nn.ModuleList([
            GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)
            for _ in range(3)
        ]))

        # cross_attn 的输入维度设置
        attn_in_dims = [hidden_dim * heads] * (num_layers - 1) + [out_dim]
        for l in range(num_layers):
            self.cross_attn.append(nn.ModuleList([
                nn.Linear(attn_in_dims[l], 1) for _ in range(3)
            ]))

    def forward(self, xs, edge_indices):
        """
        :param xs: 节点特征, 长度为3的list; xs[i]: [num_nodes, in_dim]
        :param edge_indices: 长度为3的list; edge_indices[i]: [2, num_edges_i]
        :return: 长度为3的list
        """
        # 遍历每层神经网络
        for layer in range(self.num_layers):
            new_xs = []
            # 本层对三个图都跑 GATConv，存下来
            x_all = [self.gat_layers[layer][j](xs[j], edge_indices[j]) for j in range(3)]
            # 遍历每层多层图
            for i in range(3):
                x_self = x_all[i]
                j, k = (i + 1) % 3, (i + 2) % 3
                xj, xk = x_all[j], x_all[k]  # 注意这里是 x_all 不是 xs

                a_j = torch.sigmoid(self.cross_attn[layer][i](xj))
                a_k = torch.sigmoid(self.cross_attn[layer][i](xk))
                x_cross = a_j * xj + a_k * xk
                x_out = x_self + self.dropout(x_cross)
                new_xs.append(F.elu(x_out))
            xs = new_xs
        return xs  # list of [Z^1, Z^2, Z^3]


# ===== Dynamic Link Prediction Model =====
class DynamicMultiLayerLP(nn.Module):
    def __init__(self, num_nodes, hidden_dim, embed_dim, gat_layers=2, lstm_hidden=128, dropout=0.5):
        super(DynamicMultiLayerLP, self).__init__()

        self.node_embeddings = nn.ModuleList([
            nn.Embedding(num_nodes, hidden_dim) for _ in range(3)
        ])  # 每层一个 embedding

        self.crossgat = CrossGAT3Layer(
            in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=embed_dim,
            num_layers=gat_layers
        )
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(2 * lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, 1)
        )

    def forward(self, edge_index_seq_list):
        """
        :param edge_index_seq_list: 长度为T的list，每个元素是长度为3的 edge_index 列表
        :return: pred_adj_list: list of (N, N) predicted adjacency matrices
        """
        T = len(edge_index_seq_list)
        num_layers = 3
        N = self.node_embeddings[0].num_embeddings

        # z_history表示每层的每个时间步的节点嵌入矩阵
        # z_history[l][t]表示第l层在时间步t的节点嵌入矩阵Tensor shape (N, F)
        z_history = [[] for _ in range(num_layers)]

        for t in range(T):
            # 构造每层节点特征：直接查 embedding
            x_t_list = [self.node_embeddings[l].weight for l in range(num_layers)]
            edge_t_list = edge_index_seq_list[t]
            z_t_list = self.crossgat(x_t_list, edge_t_list)
            for l in range(num_layers):
                z_history[l].append(z_t_list[l])

        # Stack over time: (N, T, F)
        z_stacked = [torch.stack(z_history[l], dim=1) for l in range(num_layers)]

        # LSTM 时间建模
        h_final = []
        for l in range(num_layers):
            output, (h_n, _) = self.lstm(z_stacked[l])  # h_n: (1, N, lstm_hidden)
            h_final.append(h_n.squeeze(0))  # (N, lstm_hidden)

        # Link prediction for all pairs
        pred_adj_list = []
        for l in range(num_layers):
            h = h_final[l]  # (N, d)
            h_u = h.unsqueeze(1).repeat(1, N, 1)
            h_v = h.unsqueeze(0).repeat(N, 1, 1)
            edge_feat = torch.cat([h_u, h_v], dim=-1)
            score = self.decoder(edge_feat).squeeze(-1)  # (N, N)
            pred_adj_list.append(score)

        # return pred_adj_list
        return pred_adj_list, z_history


class DynamicMultiLayerLP_no_lstm(nn.Module):
    def __init__(self, num_nodes, hidden_dim, embed_dim, gat_layers=2, dropout=0.5):
        super(DynamicMultiLayerLP_no_lstm, self).__init__()

        self.node_embeddings = nn.ModuleList([
            nn.Embedding(num_nodes, hidden_dim) for _ in range(3)
        ])  # 每层一个 embedding

        self.crossgat = CrossGAT3Layer(
            in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=embed_dim,
            num_layers=gat_layers
        )

        self.decoder = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, edge_index_seq_list):
        """
        :param edge_index_seq_list: 长度为T的list，每个元素是长度为3的 edge_index 列表
        :return: pred_adj_list: list of (N, N) predicted adjacency matrices
        """
        T = len(edge_index_seq_list)
        num_layers = 3
        N = self.node_embeddings[0].num_embeddings

        # z_history[l][t] 表示第l层在时间t的节点嵌入 (N, F)
        z_history = [[] for _ in range(num_layers)]

        for t in range(T):
            # 每层输入初始节点特征
            x_t_list = [self.node_embeddings[l].weight for l in range(num_layers)]
            edge_t_list = edge_index_seq_list[t]
            z_t_list = self.crossgat(x_t_list, edge_t_list)
            for l in range(num_layers):
                z_history[l].append(z_t_list[l])

        # 使用最后一个时间步的嵌入进行预测
        h_final = [z_history[l][-1] for l in range(num_layers)]  # 每层一个 (N, embed_dim)

        pred_adj_list = []
        for l in range(num_layers):
            h = h_final[l]  # (N, embed_dim)
            h_u = h.unsqueeze(1).repeat(1, N, 1)
            h_v = h.unsqueeze(0).repeat(N, 1, 1)
            edge_feat = torch.cat([h_u, h_v], dim=-1)
            score = self.decoder(edge_feat).squeeze(-1)  # (N, N)
            pred_adj_list.append(score)

        return pred_adj_list, z_history