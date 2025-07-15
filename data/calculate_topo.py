# _*_ coding : utf-8 _*_
# @Time : 2025/4/15 19:43
# @Author : wfr
# @file : calculate_topo
# @Project : MultilayerUAVLP
import numpy as np

# A: control
# B: calculate
# C: communicate
data_name = 'MG_2_C'

edge_seq = np.load('%s_edge_seq.npy' % data_name, allow_pickle=True)
# 计算平均边密度
total_density = 0
max_edges = 0
min_edges = 9999999
total_edges = 0
for i in range(edge_seq.shape[0]):
    edge_list = edge_seq[i]

    # 计算实际存在的边数
    actual_edges = len(edge_list)
    total_edges += actual_edges
    max_edges = max(max_edges, actual_edges)
    min_edges = min(min_edges, actual_edges)

    # 计算节点数，假设节点编号从1开始
    nodes = set()
    for edge in edge_list:
        nodes.add(edge[0])
        nodes.add(edge[1])
    num_nodes = len(nodes)
    # print("图快照%d的节点数为%d" % (i, num_nodes))

    # 计算完全图的边数
    complete_edges = num_nodes * (num_nodes - 1) / 2

    # 计算图的边密度
    density = actual_edges / complete_edges
    total_density += density

av_density = total_density / edge_seq.shape[0]
av_edges = total_edges / edge_seq.shape[0]
print("最小边数为:", min_edges)
print("最大边数为:", max_edges)
print("平均边数为:", av_edges)
print("图的边密度为:", av_density)