import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import torch
from matplotlib.lines import Line2D
data_name = "MG"
learning_rate = 1e-4
file_path = "emb/emb_%s_%s.npz" % (data_name, learning_rate)
os.makedirs("vis", exist_ok=True)
data = np.load(file_path)


def visualize_embedding(emb_dict, num_layers=3, num_nodes=100, title_prefix='GM', focus_nodes=range(20)):
    for t_key, emb in emb_dict.items():
        print(f"{t_key}: emb.shape = {emb.shape}")

        selected_emb = emb[:num_layers * num_nodes]
        assert selected_emb.shape[0] == num_layers * num_nodes

        emb_2d = TSNE(n_components=2, random_state=42, perplexity=20).fit_transform(selected_emb)

        plt.figure(figsize=(10, 8))

        cmap = plt.cm.get_cmap('tab20', len(focus_nodes))
        markers = ['o', 's', '^']
        assert num_layers <= len(markers)

        for node_idx, node in enumerate(focus_nodes):
            node_color = cmap(node_idx)
            coords = []

            for l in range(num_layers):
                idx = l * num_nodes + node
                jitter = np.random.normal(scale=1.5, size=2)
                point = emb_2d[idx] + jitter
                coords.append(point)

                plt.scatter(point[0], point[1],
                            marker=markers[l],
                            color=node_color,
                            edgecolor='black',
                            s=400)

            coords = np.array(coords)
            plt.plot(coords[:, 0], coords[:, 1], linestyle='--', color=node_color, alpha=0.6)

        # plt.axis('off')
        ax = plt.gca()

        ax.set_xticks(np.arange(-50, 50, 6))
        ax.set_yticks(np.arange(-50, 50, 6))
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_axisbelow(True)
        ax.grid(True, linestyle='-', linewidth=1, alpha=0.4)

        x_min, x_max = emb_2d[:, 0].min(), emb_2d[:, 0].max()
        y_min, y_max = emb_2d[:, 1].min(), emb_2d[:, 1].max()
        margin = 2
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)

        plt.savefig(f'vis/{title_prefix}_{t_key}_embedding.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close()


visualize_embedding(data, title_prefix=data_name)
