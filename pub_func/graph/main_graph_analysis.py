import os
import time
import re
import networkx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import networkx as nx
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

#esol load
# def load_graph_data_with_gnn(data_dir, output_path, gnn_model_type='GCN', hidden_channels=64, num_epochs=30):
#     """
#     加载NetworkX图数据并使用GNN提取特征
#
#     参数:
#     data_dir: 图数据所在目录
#     output_path: 结果输出目录
#     gnn_model_type: GNN模型类型 ('GCN', 'GAT', 'GraphSAGE')
#     hidden_channels: 隐藏层维度
#     num_epochs: 训练轮数
#
#     返回:
#     features: 图特征数据
#     molecule_ids: 分子ID列表
#     """
#     import os
#     import pickle
#     import numpy as np
#     import torch
#     import torch.nn.functional as F
#     import networkx as nx
#
#     # 确保输出目录存在
#     os.makedirs(output_path, exist_ok=True)
#
#     # 检查是否已经存在提取好的特征
#     feature_file = os.path.join(output_path, f'{gnn_model_type}_graph_features.npy')
#     ids_file = os.path.join(output_path, 'molecule_ids.pkl')
#
#     if os.path.exists(feature_file) and os.path.exists(ids_file):
#         print(f"加载已有的GNN特征: {feature_file}")
#         features = np.load(feature_file)
#         with open(ids_file, 'rb') as f:
#             molecule_ids = pickle.load(f)
#         return features, molecule_ids
#
#     try:
#         # 导入必要的PyTorch Geometric库
#         from torch_geometric.data import Data, DataLoader
#         from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
#     except ImportError:
#         print("请安装必要的库: pip install torch torch-geometric torch-scatter torch-sparse")
#         raise
#
#     print(f"正在从 {data_dir} 加载图数据...")
#
#     # 获取目录中的所有pkl文件
#     all_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
#
#     if not all_files:
#         raise FileNotFoundError(f"目录 {data_dir} 中未找到pkl文件")
#
#     # 存储所有图数据和ID
#     graph_data_list = []
#     molecule_ids = []
#
#     # 加载图数据并转换为PyTorch Geometric格式
#     for file_name in all_files:
#         file_path = os.path.join(data_dir, file_name)
#         print(f"加载文件: {file_path}")
#
#         try:
#             with open(file_path, 'rb') as f:
#                 G = pickle.load(f)
#
#             # 提取分子ID
#             mol_id = os.path.splitext(file_name)[0]
#
#             # 确保是NetworkX图
#             if isinstance(G, nx.Graph):
#                 # 创建节点特征字典
#                 # 首先确定我们将使用哪些属性
#                 node_attrs = ['atomic_num', 'formal_charge', 'is_aromatic']
#                 node_features = []
#
#                 # 为每个节点创建特征向量
#                 for node in sorted(G.nodes()):
#                     attrs = G.nodes[node]
#                     features = []
#
#                     # 添加原子序数（必须有）
#                     atomic_num = attrs.get('atomic_num', 0)
#                     features.append(float(atomic_num))
#
#                     # 添加形式电荷
#                     formal_charge = attrs.get('formal_charge', 0)
#                     features.append(float(formal_charge))
#
#                     # 添加其他布尔特征
#                     is_aromatic = 1.0 if attrs.get('is_aromatic', False) else 0.0
#                     features.append(is_aromatic)
#
#                     # 添加元素的one-hot编码 (可选)
#                     # 这里简化为几个常见元素
#                     element = attrs.get('element', '')
#                     element_onehot = [0.0] * 5  # 假设我们只关心C, H, O, N, 其他
#                     if element == 'C':
#                         element_onehot[0] = 1.0
#                     elif element == 'H':
#                         element_onehot[1] = 1.0
#                     elif element == 'O':
#                         element_onehot[2] = 1.0
#                     elif element == 'N':
#                         element_onehot[3] = 1.0
#                     else:
#                         element_onehot[4] = 1.0
#
#                     features.extend(element_onehot)
#
#                     node_features.append(features)
#
#                 # 创建边索引
#                 edge_index = []
#                 for u, v in G.edges():
#                     edge_index.append([u, v])
#                     edge_index.append([v, u])  # 添加反向边使图成为无向图
#
#                 # 如果图没有边，添加自环
#                 if not edge_index:
#                     for i in range(len(node_features)):
#                         edge_index.append([i, i])
#
#                 # 转换为张量
#                 x = torch.tensor(node_features, dtype=torch.float)
#                 edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#
#                 # 检查edge_index是否为空
#                 if edge_index.size(1) == 0:
#                     print(f"警告: {file_name} 没有边！添加自环边...")
#                     edge_index = torch.tensor([[i, i] for i in range(x.size(0))],
#                                               dtype=torch.long).t().contiguous()
#
#                 # 创建Data对象
#                 graph_data = Data(x=x, edge_index=edge_index)
#                 graph_data_list.append(graph_data)
#                 molecule_ids.append(mol_id)
#             else:
#                 print(f"文件 {file_name} 不包含NetworkX图对象，跳过")
#
#         except Exception as e:
#             print(f"处理文件 {file_path} 时出错: {str(e)}")
#             continue
#
#     if not graph_data_list:
#         raise ValueError("未能从文件中提取有效的图数据")
#
#     print(f"成功加载了 {len(graph_data_list)} 个分子图")
#
#     # 打印前几个图的信息用于调试
#     for i, data in enumerate(graph_data_list[:3]):
#         print(f"图 {i} 信息:")
#         print(f"  节点数: {data.x.size(0)}")
#         print(f"  特征维度: {data.x.size(1)}")
#         print(f"  边索引形状: {data.edge_index.shape}")
#         print(f"  边数: {data.edge_index.shape[1]}")
#
#     # 确定输入特征维度
#     input_dim = graph_data_list[0].x.shape[1]
#     output_dim = hidden_channels  # 输出特征维度
#
#     # 定义GNN模型
#     class GNN(torch.nn.Module):
#         def __init__(self, input_dim, hidden_channels, output_dim, model_type='GCN'):
#             super(GNN, self).__init__()
#
#             if model_type == 'GCN':
#                 self.conv1 = GCNConv(input_dim, hidden_channels)
#                 self.conv2 = GCNConv(hidden_channels, hidden_channels)
#                 self.conv3 = GCNConv(hidden_channels, output_dim)
#             elif model_type == 'GAT':
#                 self.conv1 = GATConv(input_dim, hidden_channels)
#                 self.conv2 = GATConv(hidden_channels, hidden_channels)
#                 self.conv3 = GATConv(hidden_channels, output_dim)
#             elif model_type == 'GraphSAGE':
#                 self.conv1 = SAGEConv(input_dim, hidden_channels)
#                 self.conv2 = SAGEConv(hidden_channels, hidden_channels)
#                 self.conv3 = SAGEConv(hidden_channels, output_dim)
#             else:
#                 raise ValueError(f"不支持的GNN模型类型: {model_type}")
#
#         def forward(self, x, edge_index, batch=None):
#             # 应用GNN层
#             x = self.conv1(x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=0.2, training=self.training)
#
#             x = self.conv2(x, edge_index)
#             x = F.relu(x)
#
#             x = self.conv3(x, edge_index)
#
#             # 如果提供了batch信息，则进行图池化
#             if batch is not None:
#                 x = global_mean_pool(x, batch)
#
#             return x
#
#     # 创建数据加载器
#     loader = DataLoader(graph_data_list, batch_size=32, shuffle=True)
#
#     # 初始化模型
#     device = torch.device('cpu')  # 使用CPU
#     print(f"使用设备: {device}")
#
#     model = GNN(input_dim, hidden_channels, output_dim, model_type=gnn_model_type).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
#     # 训练模型（链接预测任务）
#     print(f"开始训练 {gnn_model_type} 模型...")
#
#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for data in loader:
#             data = data.to(device)
#             optimizer.zero_grad()
#
#             # 使用模型提取节点嵌入
#             node_embeddings = model(data.x, data.edge_index)
#
#             # 链接预测任务
#             src = data.edge_index[0]
#             dst = data.edge_index[1]
#
#             # 如果边太多，随机采样一部分
#             if src.size(0) > 5000:
#                 perm = torch.randperm(src.size(0))[:5000]
#                 src = src[perm]
#                 dst = dst[perm]
#
#             # 正样本边的嵌入相似度
#             pos_score = (node_embeddings[src] * node_embeddings[dst]).sum(dim=1)
#
#             # 生成负样本边
#             neg_dst = torch.randint(0, data.num_nodes, (src.size(0),), device=device)
#             neg_score = (node_embeddings[src] * node_embeddings[neg_dst]).sum(dim=1)
#
#             # 使用margin ranking loss
#             loss = F.margin_ranking_loss(
#                 pos_score, neg_score, torch.ones_like(pos_score), margin=0.1
#             )
#
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item() * data.num_graphs
#
#         # 每5个epoch打印一次损失
#         if (epoch + 1) % 5 == 0 or epoch == 0:
#             print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {total_loss / len(graph_data_list):.4f}')
#
#     # 提取图特征
#     print("提取图特征...")
#
#     model.eval()
#     all_features = []
#
#     with torch.no_grad():
#         for i, data in enumerate(graph_data_list):
#             try:
#                 # 将单个图转移到设备
#                 data = data.to(device)
#
#                 # 提取节点特征
#                 node_features = model(data.x, data.edge_index)
#
#                 # 图级别池化 - 对所有节点特征取平均
#                 graph_embedding = node_features.mean(dim=0, keepdim=True)
#
#                 all_features.append(graph_embedding.cpu().numpy())
#             except Exception as e:
#                 print(f"处理图 {i} (ID: {molecule_ids[i]}) 时出错: {str(e)}")
#                 # 创建零向量作为占位符
#                 zero_feat = np.zeros((1, output_dim))
#                 all_features.append(zero_feat)
#
#     # 将所有特征连接起来
#     features = np.vstack(all_features)
#
#     # 保存特征
#     np.save(feature_file, features)
#     with open(ids_file, 'wb') as f:
#         pickle.dump(molecule_ids, f)
#
#     print(f"成功提取了 {len(features)} 个分子的图特征，特征维度为 {features.shape[1]}")
#     return features, molecule_ids

#qm7b load

def load_graph_data_with_gnn(data_dir, output_path, gnn_model_type='GCN', hidden_channels=64, num_epochs=30):
    """
    加载NetworkX图数据并使用GNN提取特征

    参数:
    data_dir: 图数据所在目录
    output_path: 结果输出目录
    gnn_model_type: GNN模型类型 ('GCN', 'GAT', 'GraphSAGE')
    hidden_channels: 隐藏层维度
    num_epochs: 训练轮数

    返回:
    features: 图特征数据
    molecule_ids: 分子ID列表
    """
    import os
    import pickle
    import numpy as np
    import torch
    import torch.nn.functional as F
    import networkx as nx

    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)

    # 检查是否已经存在提取好的特征
    feature_file = os.path.join(output_path, f'{gnn_model_type}_graph_features.npy')
    ids_file = os.path.join(output_path, 'molecule_ids.pkl')

    if os.path.exists(feature_file) and os.path.exists(ids_file):
        print(f"加载已有的GNN特征: {feature_file}")
        features = np.load(feature_file)
        with open(ids_file, 'rb') as f:
            molecule_ids = pickle.load(f)
        return features, molecule_ids

    try:
        # 导入必要的PyTorch Geometric库
        from torch_geometric.data import Data, DataLoader
        from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
    except ImportError:
        print("请安装必要的库: pip install torch torch-geometric torch-scatter torch-sparse")
        raise

    print(f"正在从 {data_dir} 加载图数据...")

    # 获取目录中的所有pkl和gpickle文件
    all_files = [f for f in os.listdir(data_dir)
                 if f.endswith('.pkl') or f.endswith('.gpickle')]

    if not all_files:
        raise FileNotFoundError(f"目录 {data_dir} 中未找到pkl或gpickle文件")

    # 存储所有图数据和ID
    graph_data_list = []
    molecule_ids = []

    # 加载图数据并转换为PyTorch Geometric格式
    for file_name in all_files:
        file_path = os.path.join(data_dir, file_name)
        print(f"加载文件: {file_path}")

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # 提取分子ID
            mol_id = os.path.splitext(file_name)[0]

            # 处理NetworkX图对象
            if isinstance(data, nx.Graph):
                G = data
                # 创建节点特征列表
                node_features = []

                # 为每个节点创建特征向量
                for node in sorted(G.nodes()):
                    attrs = G.nodes[node]
                    features = []

                    # 元素类型的one-hot编码
                    element = attrs.get('element', '')
                    element_onehot = [0.0] * 5  # H, C, N, O, S
                    if element == 'H':
                        element_onehot[0] = 1.0
                    elif element == 'C':
                        element_onehot[1] = 1.0
                    elif element == 'N':
                        element_onehot[2] = 1.0
                    elif element == 'O':
                        element_onehot[3] = 1.0
                    elif element == 'S':
                        element_onehot[4] = 1.0

                    features.extend(element_onehot)

                    # 如果有xyz坐标，添加为特征
                    if 'xyz' in attrs:
                        xyz = attrs['xyz']
                        features.extend([float(xyz[0]), float(xyz[1]), float(xyz[2])])

                    node_features.append(features)

                # 创建边索引
                edge_index = []
                for u, v in G.edges():
                    edge_index.append([u, v])
                    edge_index.append([v, u])  # 添加反向边使图成为无向图

                # 如果图没有边，添加自环
                if not edge_index:
                    for i in range(len(node_features)):
                        edge_index.append([i, i])

                # 转换为张量
                x = torch.tensor(node_features, dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

                # 创建Data对象
                graph_data = Data(x=x, edge_index=edge_index)
                graph_data_list.append(graph_data)
                molecule_ids.append(mol_id)
            else:
                print(f"文件 {file_name} 不包含NetworkX图对象，尝试其他格式...")
                # 继续处理代码...

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            continue

    if not graph_data_list:
        raise ValueError("未能从文件中提取有效的图数据")

    print(f"成功加载了 {len(graph_data_list)} 个分子图")

    # 打印前几个图的信息用于调试
    for i, data in enumerate(graph_data_list[:3]):
        print(f"图 {i} 信息:")
        print(f"  节点数: {data.x.size(0)}")
        print(f"  特征维度: {data.x.size(1)}")
        print(f"  边索引形状: {data.edge_index.shape}")
        print(f"  边数: {data.edge_index.shape[1]}")

    # 确定输入特征维度
    input_dim = graph_data_list[0].x.shape[1]
    output_dim = hidden_channels  # 输出特征维度

    # 定义GNN模型
    class GNN(torch.nn.Module):
        def __init__(self, input_dim, hidden_channels, output_dim, model_type='GCN'):
            super(GNN, self).__init__()

            if model_type == 'GCN':
                self.conv1 = GCNConv(input_dim, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, hidden_channels)
                self.conv3 = GCNConv(hidden_channels, output_dim)
            elif model_type == 'GAT':
                self.conv1 = GATConv(input_dim, hidden_channels)
                self.conv2 = GATConv(hidden_channels, hidden_channels)
                self.conv3 = GATConv(hidden_channels, output_dim)
            elif model_type == 'GraphSAGE':
                self.conv1 = SAGEConv(input_dim, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, hidden_channels)
                self.conv3 = SAGEConv(hidden_channels, output_dim)
            else:
                raise ValueError(f"不支持的GNN模型类型: {model_type}")

        def forward(self, x, edge_index, batch=None):
            # 应用GNN层
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

            x = self.conv2(x, edge_index)
            x = F.relu(x)

            x = self.conv3(x, edge_index)

            # 如果提供了batch信息，则进行图池化
            if batch is not None:
                x = global_mean_pool(x, batch)

            return x

    # 创建数据加载器
    loader = DataLoader(graph_data_list, batch_size=32, shuffle=True)

    # 初始化模型
    device = torch.device('cpu')  # 使用CPU
    print(f"使用设备: {device}")

    model = GNN(input_dim, hidden_channels, output_dim, model_type=gnn_model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 训练模型（链接预测任务）
    print(f"开始训练 {gnn_model_type} 模型...")

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()

            # 使用模型提取节点嵌入
            node_embeddings = model(data.x, data.edge_index)

            # 链接预测任务
            src = data.edge_index[0]
            dst = data.edge_index[1]

            # 如果边太多，随机采样一部分
            if src.size(0) > 5000:
                perm = torch.randperm(src.size(0))[:5000]
                src = src[perm]
                dst = dst[perm]

            # 正样本边的嵌入相似度
            pos_score = (node_embeddings[src] * node_embeddings[dst]).sum(dim=1)

            # 生成负样本边
            neg_dst = torch.randint(0, data.num_nodes, (src.size(0),), device=device)
            neg_score = (node_embeddings[src] * node_embeddings[neg_dst]).sum(dim=1)

            # 使用margin ranking loss
            loss = F.margin_ranking_loss(
                pos_score, neg_score, torch.ones_like(pos_score), margin=0.1
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs

        # 每5个epoch打印一次损失
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {total_loss / len(graph_data_list):.4f}')

    # 提取图特征
    print("提取图特征...")

    model.eval()
    all_features = []

    with torch.no_grad():
        for i, data in enumerate(graph_data_list):
            try:
                # 将单个图转移到设备
                data = data.to(device)

                # 提取节点特征
                node_features = model(data.x, data.edge_index)

                # 图级别池化 - 对所有节点特征取平均
                graph_embedding = node_features.mean(dim=0, keepdim=True)

                all_features.append(graph_embedding.cpu().numpy())
            except Exception as e:
                print(f"处理图 {i} (ID: {molecule_ids[i]}) 时出错: {str(e)}")
                # 创建零向量作为占位符
                zero_feat = np.zeros((1, output_dim))
                all_features.append(zero_feat)

    # 将所有特征连接起来
    features = np.vstack(all_features)

    # 保存特征
    np.save(feature_file, features)
    with open(ids_file, 'wb') as f:
        pickle.dump(molecule_ids, f)

    print(f"成功提取了 {len(features)} 个分子的图特征，特征维度为 {features.shape[1]}")
    return features, molecule_ids

def run_dbscan_clustering(X, eps_range=(0.1, 0.5, 0.1), min_samples_range=(5, 20, 5)):
    """
    执行DBSCAN聚类算法，尝试不同的参数组合

    参数:
    X: 输入数据
    eps_range: epsilon参数范围，格式(min, max, step)
    min_samples_range: min_samples参数范围，格式(min, max, step)

    返回:
    best_labels: 最佳聚类标签
    best_silhouette: 最佳轮廓系数
    eps_results: 不同参数组合的结果
    """
    print("\n运行DBSCAN聚类分析...")

    best_silhouette = -1
    best_labels = None
    best_params = None
    eps_results = []

    eps_values = np.arange(eps_range[0], eps_range[1], eps_range[2])
    min_samples_values = range(min_samples_range[0], min_samples_range[1], min_samples_range[2])

    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                print(f"  DBSCAN参数: eps={eps:.2f}, min_samples={min_samples}")
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)

                # 检查是否有噪声点（标签为-1）
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)

                print(f"  找到 {n_clusters} 个簇, {n_noise} 个噪声点")

                if n_clusters <= 1:
                    print("  只找到1个簇或全是噪声，跳过")
                    continue

                # 计算轮廓系数（排除噪声点）
                if n_noise < len(X):
                    mask = labels != -1
                    if sum(mask) > 1:  # 确保有足够的非噪声点
                        silhouette = silhouette_score(X[mask], labels[mask])
                        print(f"  轮廓系数: {silhouette:.4f}")

                        eps_results.append({
                            'eps': eps,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'n_noise': n_noise,
                            'silhouette': silhouette
                        })

                        if silhouette > best_silhouette:
                            best_silhouette = silhouette
                            best_labels = labels
                            best_params = (eps, min_samples)

            except Exception as e:
                print(f"  DBSCAN (eps={eps}, min_samples={min_samples}) 出错: {str(e)}")

    if best_labels is not None:
        print(f"\nDBSCAN最佳参数: eps={best_params[0]:.2f}, min_samples={best_params[1]}")
        print(f"簇数量: {len(set(best_labels)) - (1 if -1 in best_labels else 0)}")
        print(f"轮廓系数: {best_silhouette:.4f}")
    else:
        print("\n未找到有效的DBSCAN聚类结果")

    return best_labels, best_silhouette, eps_results


def run_hierarchical_clustering(X, n_clusters_range=(2, 10)):
    """
    执行层次聚类算法，尝试不同的簇数量

    参数:
    X: 输入数据
    n_clusters_range: 簇数量范围，格式(min, max)

    返回:
    best_labels: 最佳聚类标签
    best_silhouette: 最佳轮廓系数
    cluster_results: 不同簇数量的结果
    """
    print("\n运行层次聚类分析...")

    best_silhouette = -1
    best_labels = None
    best_n_clusters = None
    cluster_results = []

    for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
        try:
            print(f"  层次聚类: n_clusters={n_clusters}")
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X)

            # 计算轮廓系数
            silhouette = silhouette_score(X, labels)
            print(f"  轮廓系数: {silhouette:.4f}")

            cluster_results.append({
                'n_clusters': n_clusters,
                'silhouette': silhouette
            })

            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_labels = labels
                best_n_clusters = n_clusters

        except Exception as e:
            print(f"  层次聚类 (n_clusters={n_clusters}) 出错: {str(e)}")

    if best_labels is not None:
        print(f"\n层次聚类最佳簇数量: {best_n_clusters}")
        print(f"轮廓系数: {best_silhouette:.4f}")
    else:
        print("\n未找到有效的层次聚类结果")

    return best_labels, best_silhouette, cluster_results


#kmeans
def run_kmeans_and_ils(X_embedded, k_range=(2, 10), max_time=300):
    """
    运行KMeans聚类算法，寻找最佳聚类数量

    参数:
    X_embedded: 降维后的数据
    k_range: 尝试的聚类数量范围，元组(最小值, 最大值)
    max_time: 最大运行时间(秒)

    返回:
    best_labels: 最佳聚类标签
    best_silhouette: 最佳轮廓系数
    history: 不同k值的轮廓系数历史
    """
    import time
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as np

    print(f"开始聚类分析，数据形状: {X_embedded.shape}")
    start_time = time.time()

    # 初始化最佳结果记录
    best_silhouette = -1
    best_labels = None
    best_k = None
    history = []

    # 检查数据中是否有NaN或Inf值
    if np.isnan(X_embedded).any() or np.isinf(X_embedded).any():
        print("警告: 输入数据包含NaN或Inf值，将尝试清理数据")
        mask = np.isfinite(X_embedded).all(axis=1)
        X_embedded = X_embedded[mask]
        if len(X_embedded) == 0:
            print("错误: 清理后无可用数据")
            return np.array([]), -1, []

    # 计算不同k值下KMeans的轮廓系数
    print("计算KMeans聚类轮廓系数...")

    try:
        for k in range(k_range[0], k_range[1] + 1):
            # 检查超时
            if time.time() - start_time > max_time:
                print(f"警告: 聚类分析已运行 {max_time} 秒，提前停止")
                break

            print(f"  计算 k={k} 的KMeans聚类...")
            try:
                km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X_embedded)
                score = silhouette_score(X_embedded, km.labels_)

                history.append({'n_clusters': k, 'silhouette': score})
                print(f"  k={k} 的轮廓系数: {score:.4f}")

                if score > best_silhouette:
                    best_silhouette = score
                    best_labels = km.labels_
                    best_k = k
            except Exception as e:
                print(f"  计算 k={k} 时出错: {str(e)}")
    except Exception as e:
        print(f"KMeans聚类阶段出错: {str(e)}")

    # 如果没有成功的KMeans结果，返回默认值
    if best_labels is None:
        print("警告: 未能成功完成KMeans聚类")
        return np.zeros(len(X_embedded)), -1, []

    total_time = time.time() - start_time
    print(f"聚类分析完成，耗时: {total_time:.2f}秒")
    print(f"最佳聚类数: {best_k}, 轮廓系数: {best_silhouette:.4f}")

    return best_labels, best_silhouette, history

def perform_pca(X, n_components):
    """
    执行PCA降维

    参数:
    X: 输入数据矩阵
    n_components: 降维后的维度

    返回:
    X_pca: 降维后的数据
    pca: PCA模型对象
    metrics: 模型特有的指标
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # 计算额外的指标
    X_reconstructed = pca.inverse_transform(X_pca)
    reconstruction_error = np.mean(np.square(X - X_reconstructed))
    variance_explained = np.sum(pca.explained_variance_ratio_)

    metrics = {
        'reconstruction_error': reconstruction_error,
        'variance_explained': variance_explained
    }

    print(f"PCA (n_components={n_components}):")
    print(f"  重构误差: {reconstruction_error:.4f}")
    print(f"  解释方差比例: {variance_explained:.4f}")

    return X_pca, pca, metrics


def perform_umap(X, n_components, n_neighbors, min_dist):
    """
    执行UMAP降维

    参数:
    X: 输入数据矩阵
    n_components: 降维后的维度
    n_neighbors: 局部邻居数量
    min_dist: 最小距离参数

    返回:
    X_umap: 降维后的数据
    umap_reducer: UMAP模型对象
    metrics: 模型特有的指标
    """
    umap_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                             n_components=n_components, random_state=42)
    X_umap = umap_reducer.fit_transform(X)

    metrics = {}

    print(f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}):")
    print(f"  降维完成")

    return X_umap, umap_reducer, metrics



def perform_autoencoder(X, encoding_dim, intermediate_dim=256, epochs=50, batch_size=32):
    """
    执行自编码器降维

    参数:
    X: 输入数据矩阵
    encoding_dim: 编码维度
    intermediate_dim: 中间层维度
    epochs: 训练轮数
    batch_size: 批处理大小

    返回:
    X_ae: 降维后的数据
    encoder: 编码器模型
    metrics: 模型特有的指标
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np

    # 转换为PyTorch张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义自编码器模型
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, encoding_dim, intermediate_dim):
            super(Autoencoder, self).__init__()
            # 编码器
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, intermediate_dim),
                nn.ReLU(),
                nn.Linear(intermediate_dim, encoding_dim)
            )
            # 解码器
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, intermediate_dim),
                nn.ReLU(),
                nn.Linear(intermediate_dim, input_dim)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

        def encode(self, x):
            return self.encoder(x)

    # 初始化模型
    input_dim = X.shape[1]
    model = Autoencoder(input_dim, encoding_dim, intermediate_dim)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    print(f"Autoencoder (encoding_dim={encoding_dim}, intermediate_dim={intermediate_dim}):")
    print(f"  开始训练...")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in dataloader:
            # 前向传播
            decoded = model(batch_x)
            loss = criterion(decoded, batch_x)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(dataset)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    # 生成降维结果
    model.eval()
    with torch.no_grad():
        X_encoded = model.encode(X_tensor).numpy()

    # 计算重构误差
    with torch.no_grad():
        X_decoded = model(X_tensor).numpy()
    reconstruction_error = np.mean(np.square(X - X_decoded))

    # 计算自编码器散度
    ae_divergence = calculate_autoencoder_divergence(X, X_encoded, X_decoded)

    metrics = {
        'reconstruction_error': reconstruction_error,
        'ae_divergence': ae_divergence
    }

    print(f"  训练完成，重构误差: {reconstruction_error:.4f}")
    print(f"  自编码器散度: {ae_divergence:.6f}")

    return X_encoded, model, metrics


def calculate_autoencoder_divergence(X_original, X_encoded, X_decoded):
    """
    计算自编码器散度（信息损失度量）

    参数:
    X_original: 原始高维数据
    X_encoded: 降维后的数据
    X_decoded: 重构后的数据

    返回:
    divergence: 散度值
    """
    import numpy as np

    # 计算重构误差
    reconstruction_error = np.mean(np.square(X_original - X_decoded))

    # 计算编码空间的方差比
    original_variance = np.var(X_original, axis=0).sum()
    encoded_variance = np.var(X_encoded, axis=0).sum()
    variance_ratio = encoded_variance / original_variance if original_variance > 0 else 0

    # 结合重构误差和方差比计算散度
    # 低重构误差和高方差保留表示低散度
    divergence = reconstruction_error / (variance_ratio + 1e-10)

    return divergence


def calculate_pca_divergence(X_original, X_pca, pca_model):
    """
    计算PCA散度

    参数:
    X_original: 原始高维数据
    X_pca: PCA降维后的数据
    pca_model: PCA模型对象

    返回:
    divergence: PCA散度值
    """
    import numpy as np

    # 计算重构误差
    X_reconstructed = pca_model.inverse_transform(X_pca)
    reconstruction_error = np.mean(np.square(X_original - X_reconstructed))

    # 计算未被解释的方差比例
    unexplained_variance_ratio = 1.0 - np.sum(pca_model.explained_variance_ratio_)

    # PCA散度 - 结合重构误差和未解释方差
    divergence = reconstruction_error * unexplained_variance_ratio

    return divergence


def calculate_umap_divergence(X_original, X_umap, n_neighbors):
    """
    计算UMAP散度

    参数:
    X_original: 原始高维数据
    X_umap: UMAP降维后的数据
    n_neighbors: UMAP参数

    返回:
    divergence: UMAP散度值
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    # 原始空间中的k近邻
    nn_orig = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn_orig.fit(X_original)
    indices_orig = nn_orig.kneighbors(X_original, return_distance=False)

    # 降维空间中的k近邻
    nn_umap = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn_umap.fit(X_umap)
    indices_umap = nn_umap.kneighbors(X_umap, return_distance=False)

    # 计算邻居保留率
    neighbor_preservation = 0.0
    n_samples = X_original.shape[0]

    for i in range(n_samples):
        orig_neighbors = set(indices_orig[i][1:])  # 排除自身
        umap_neighbors = set(indices_umap[i][1:])  # 排除自身
        preservation = len(orig_neighbors.intersection(umap_neighbors)) / n_neighbors
        neighbor_preservation += preservation

    neighbor_preservation /= n_samples

    # UMAP散度 = 1 - 邻居保留率
    divergence = 1.0 - neighbor_preservation

    return divergence


def calculate_manifold_divergence(X_original, X_embedded):
    """
    计算流形散度

    参数:
    X_original: 原始高维数据
    X_embedded: 降维后的数据

    返回:
    divergence: 流形散度值
    """
    import numpy as np
    from sklearn.metrics import pairwise_distances

    # 计算原始空间中的成对距离
    D_original = pairwise_distances(X_original)

    # 计算嵌入空间中的成对距离
    D_embedded = pairwise_distances(X_embedded)

    # 归一化距离矩阵
    D_original = D_original / np.max(D_original)
    D_embedded = D_embedded / np.max(D_embedded)

    # 计算距离保留误差
    distance_preservation_error = np.mean(np.abs(D_original - D_embedded))

    return distance_preservation_error


def save_embedding_csv(X_embedded, output_path, filename):
    """
    保存降维结果为CSV文件
    """
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, filename)

    pd.DataFrame(X_embedded).to_csv(output_file, index=False)
    print(f"降维结果已保存到: {output_file}")


def plot_2d_embedding(X_embedded, output_path, filename, title, cluster_labels=None):
    """
    绘制2D降维可视化
    """
    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(10, 8))

    if cluster_labels is not None:
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7)
        plt.colorbar(label='Cluster')
    else:
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.7)

    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    output_file = os.path.join(output_path, filename)
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"可视化已保存到: {output_file}")


def plot_clustering_comparison(results_df, output_path):
    """
    绘制不同聚类方法的轮廓系数比较图
    """
    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(12, 8))
    methods = results_df['Method'].unique()

    # 为每种聚类方法选择不同的颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for i, method in enumerate(methods):
        subset = results_df[results_df['Method'] == method]
        plt.bar(i, subset['Silhouette Score'].values[0], color=colors[i], alpha=0.7)

    plt.xlabel('Clustering Method')
    plt.ylabel('Silhouette Score')
    plt.title('Clustering Method Comparison')
    plt.xticks(range(len(methods)), methods, rotation=45)
    plt.tight_layout()

    output_file = os.path.join(output_path, 'clustering_comparison.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"聚类比较图已保存到: {output_file}")


def run_graph_dimensionality_reduction(graph_dir, output_path, gnn_model_type='GCN'):
    """
    主函数，运行图表示数据的降维和聚类分析

    参数:
    graph_dir: 图表示数据所在目录
    output_path: 结果输出目录
    gnn_model_type: GNN模型类型 ('GCN', 'GAT', 'GraphSAGE')
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 使用GNN提取特征
    start_time = time.time()
    features, molecule_ids = load_graph_data_with_gnn(graph_dir, output_path, gnn_model_type)
    feature_extraction_time = time.time() - start_time

    # 标准化数据
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 保存原始特征
    feature_df = pd.DataFrame(features)
    if molecule_ids:
        feature_df['molecule_id'] = molecule_ids
    feature_df.to_csv(os.path.join(output_path, f"{gnn_model_type}_graph_features.csv"), index=False)

    # 初始化结果存储
    dimensionality_results = []
    clustering_results = []

    # 执行PCA降维
    print("\n正在执行PCA降维...")
    pca_results = []
    pca_clustering = []

    for n_components in [2, 5, 10, 20]:
        start_time = time.time()
        X_pca, pca_model, pca_metrics = perform_pca(features_scaled, n_components)
        runtime = time.time() - start_time

        # 计算PCA散度
        pca_divergence = calculate_pca_divergence(features_scaled, X_pca, pca_model)
        pca_metrics['pca_divergence'] = pca_divergence

        # 计算流形散度
        manifold_divergence = calculate_manifold_divergence(features_scaled, X_pca)
        pca_metrics['manifold_divergence'] = manifold_divergence

        # 计算聚类评估指标
        eval_metrics = evaluate_embedding(features_scaled, X_pca)
        all_metrics = {**eval_metrics, **pca_metrics}

        # 保存降维结果
        save_embedding_csv(X_pca, output_path, f"pca_{n_components}d.csv")

        # 对降维后的数据执行各种聚类算法
        print(f"\n在PCA {n_components}D投影上运行聚类...")

        # KMeans
        kmeans_labels, kmeans_silhouette, history = run_kmeans_and_ils(X_pca)
        all_metrics['kmeans_score'] = kmeans_silhouette
        print(f"PCA {n_components}D: KMeans 轮廓系数 = {kmeans_silhouette:.4f}")

        # DBSCAN (仅对较低维度数据运行以提高效率)
        if n_components <= 10:
            dbscan_labels, dbscan_silhouette, _ = run_dbscan_clustering(X_pca)
            all_metrics['dbscan_silhouette'] = dbscan_silhouette if dbscan_labels is not None else None
            if dbscan_labels is not None:
                print(f"PCA {n_components}D: DBSCAN 轮廓系数 = {dbscan_silhouette:.4f}")

        # 层次聚类 (仅对较低维度数据运行以提高效率)
        if n_components <= 10:
            hierarchical_labels, hierarchical_silhouette, _ = run_hierarchical_clustering(X_pca)
            all_metrics['hierarchical_silhouette'] = hierarchical_silhouette if hierarchical_labels is not None else None
            if hierarchical_labels is not None:
                print(f"PCA {n_components}D: 层次聚类 轮廓系数 = {hierarchical_silhouette:.4f}")

        # 记录PCA结果
        result = {
            'Method': 'PCA',
            'Parameters': f'n_components={n_components}',
            'Runtime (s)': runtime,
            'Neighbor Preservation': eval_metrics.get('neighbor_preservation'),
            'Trustworthiness': eval_metrics.get('trustworthiness'),
            'Continuity': eval_metrics.get('continuity'),
            'Variance Explained': pca_metrics.get('variance_explained'),
            'Reconstruction Error': pca_metrics.get('reconstruction_error'),
            'PCA Divergence': pca_divergence,
            'Manifold Divergence': manifold_divergence,
            'KMeans Silhouette': kmeans_silhouette,
            'DBSCAN Silhouette': all_metrics.get('dbscan_silhouette'),
            'Hierarchical Silhouette': all_metrics.get('hierarchical_silhouette'),
            'GNN Model': gnn_model_type,
            'Feature Extraction Time (s)': feature_extraction_time
        }
        pca_results.append(result)

        # 记录聚类结果
        pca_clustering.append({
            'Method': 'KMeans',
            'Dimensionality Reduction': f'PCA {n_components}D',
            'Silhouette Score': kmeans_silhouette
        })

        if n_components <= 10 and 'dbscan_silhouette' in all_metrics and all_metrics['dbscan_silhouette'] is not None:
            pca_clustering.append({
                'Method': 'DBSCAN',
                'Dimensionality Reduction': f'PCA {n_components}D',
                'Silhouette Score': all_metrics['dbscan_silhouette']
            })

        if n_components <= 10 and 'hierarchical_silhouette' in all_metrics and all_metrics['hierarchical_silhouette'] is not None:
            pca_clustering.append({
                'Method': 'Hierarchical',
                'Dimensionality Reduction': f'PCA {n_components}D',
                'Silhouette Score': all_metrics['hierarchical_silhouette']
            })

        # 如果是2D，创建可视化
        if n_components == 2:
            plot_2d_embedding(X_pca, output_path, 'pca_2d_plot.png', 'PCA 2D Projection of Graph Features')
            plot_2d_embedding(X_pca, output_path, 'pca_kmeans_clusters.png',
                              'PCA 2D with KMeans Clustering', kmeans_labels)

            if 'dbscan_silhouette' in all_metrics and all_metrics['dbscan_silhouette'] is not None:
                plot_2d_embedding(X_pca, output_path, 'pca_dbscan_clusters.png',
                                  'PCA 2D with DBSCAN Clustering', dbscan_labels)

            if 'hierarchical_silhouette' in all_metrics and all_metrics['hierarchical_silhouette'] is not None:
                plot_2d_embedding(X_pca, output_path, 'pca_hierarchical_clusters.png',
                                  'PCA 2D with Hierarchical Clustering', hierarchical_labels)

    # 执行UMAP降维
    print("\n正在执行UMAP降维...")
    umap_results = []
    umap_clustering = []

    for n_neighbors in [5, 15, 30]:
        for min_dist in [0.1, 0.5]:
            for n_components in [2, 5]:
                start_time = time.time()
                X_umap, umap_model, umap_metrics = perform_umap(features_scaled, n_components,
                                                                n_neighbors, min_dist)
                runtime = time.time() - start_time

                # 计算UMAP散度
                umap_divergence = calculate_umap_divergence(features_scaled, X_umap, n_neighbors)
                umap_metrics['umap_divergence'] = umap_divergence

                # 计算流形散度
                manifold_divergence = calculate_manifold_divergence(features_scaled, X_umap)
                umap_metrics['manifold_divergence'] = manifold_divergence

                # 计算聚类评估指标
                eval_metrics = evaluate_embedding(features_scaled, X_umap)
                all_metrics = {**eval_metrics, **umap_metrics}

                # 保存结果
                save_embedding_csv(X_umap, output_path,
                                   f"umap_nn{n_neighbors}_md{min_dist}_{n_components}d.csv")

                # 执行聚类
                print(f"\n在UMAP {n_components}D投影上运行聚类 (n_neighbors={n_neighbors}, min_dist={min_dist})...")

                # KMeans
                kmeans_labels, kmeans_silhouette, history= run_kmeans_and_ils(X_umap)

                all_metrics['kmeans_score'] = kmeans_silhouette
                print(f"UMAP {n_components}D: KMeans 轮廓系数 = {kmeans_silhouette:.4f}")

                # DBSCAN
                dbscan_labels, dbscan_silhouette, _ = run_dbscan_clustering(X_umap)
                all_metrics['dbscan_silhouette'] = dbscan_silhouette if dbscan_labels is not None else None
                if dbscan_labels is not None:
                    print(f"UMAP {n_components}D: DBSCAN 轮廓系数 = {dbscan_silhouette:.4f}")

                # 层次聚类
                hierarchical_labels, hierarchical_silhouette, _ = run_hierarchical_clustering(X_umap)
                all_metrics['hierarchical_silhouette'] = hierarchical_silhouette if hierarchical_labels is not None else None
                if hierarchical_labels is not None:
                    print(f"UMAP {n_components}D: 层次聚类 轮廓系数 = {hierarchical_silhouette:.4f}")

                # 记录UMAP结果
                result = {
                    'Method': 'UMAP',
                    'Parameters': f'n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}',
                    'Runtime (s)': runtime,
                    'Neighbor Preservation': eval_metrics.get('neighbor_preservation'),
                    'Trustworthiness': eval_metrics.get('trustworthiness'),
                    'Continuity': eval_metrics.get('continuity'),
                    'UMAP Divergence': umap_divergence,
                    'Manifold Divergence': manifold_divergence,
                    'KMeans Silhouette': kmeans_silhouette,
                    'DBSCAN Silhouette': all_metrics.get('dbscan_silhouette'),
                    'Hierarchical Silhouette': all_metrics.get('hierarchical_silhouette'),
                    'GNN Model': gnn_model_type,
                    'Feature Extraction Time (s)': feature_extraction_time
                }
                umap_results.append(result)

                # 记录聚类结果
                umap_clustering.append({
                    'Method': 'KMeans',
                    'Dimensionality Reduction': f'UMAP {n_components}D (n_neighbors={n_neighbors}, min_dist={min_dist})',
                    'Silhouette Score': kmeans_silhouette
                })

                if all_metrics.get('dbscan_silhouette') is not None:
                    umap_clustering.append({
                        'Method': 'DBSCAN',
                        'Dimensionality Reduction': f'UMAP {n_components}D (n_neighbors={n_neighbors}, min_dist={min_dist})',
                        'Silhouette Score': all_metrics['dbscan_silhouette']
                    })

                if all_metrics.get('hierarchical_silhouette') is not None:
                    umap_clustering.append({
                        'Method': 'Hierarchical',
                        'Dimensionality Reduction': f'UMAP {n_components}D (n_neighbors={n_neighbors}, min_dist={min_dist})',
                        'Silhouette Score': all_metrics['hierarchical_silhouette']
                    })

                # 如果是2D，创建可视化
                if n_components == 2 and n_neighbors == 15 and min_dist == 0.1:
                    plot_2d_embedding(X_umap, output_path,
                                      f'umap_nn{n_neighbors}_md{min_dist}_2d_plot.png',
                                      f'UMAP 2D (n_neighbors={n_neighbors}, min_dist={min_dist})')

                    plot_2d_embedding(X_umap, output_path,
                                      f'umap_nn{n_neighbors}_md{min_dist}_kmeans_clusters.png',
                                      'UMAP 2D with KMeans Clustering', kmeans_labels)

                    if dbscan_labels is not None:
                        plot_2d_embedding(X_umap, output_path,
                                          f'umap_nn{n_neighbors}_md{min_dist}_dbscan_clusters.png',
                                          'UMAP 2D with DBSCAN Clustering', dbscan_labels)

                    if hierarchical_labels is not None:
                        plot_2d_embedding(X_umap, output_path,
                                          f'umap_nn{n_neighbors}_md{min_dist}_hierarchical_clusters.png',
                                          'UMAP 2D with Hierarchical Clustering', hierarchical_labels)

    # 执行自编码器降维
    print("\n正在执行自编码器降维...")
    ae_results = []
    ae_clustering = []

    for encoding_dim in [2, 5, 10, 20]:
        for intermediate_dim in [128, 256, 512]:
            start_time = time.time()
            X_ae, ae_model, ae_metrics = perform_autoencoder(features_scaled, encoding_dim, intermediate_dim)
            runtime = time.time() - start_time

            # 计算流形散度
            manifold_divergence = calculate_manifold_divergence(features_scaled, X_ae)
            ae_metrics['manifold_divergence'] = manifold_divergence

            # 计算聚类评估指标
            eval_metrics = evaluate_embedding(features_scaled, X_ae)
            all_metrics = {**eval_metrics, **ae_metrics}

            # 保存结果
            save_embedding_csv(X_ae, output_path, f"ae_ed{encoding_dim}_id{intermediate_dim}.csv")

            # 执行聚类
            print(f"\n在自编码器 {encoding_dim}D投影上运行聚类 (intermediate_dim={intermediate_dim})...")

            # KMeans
            kmeans_labels, kmeans_silhouette, history = run_kmeans_and_ils(X_ae)

            all_metrics['kmeans_score'] = kmeans_silhouette
            print(f"AE {encoding_dim}D: KMeans 轮廓系数 = {kmeans_silhouette:.4f}")

            # DBSCAN (仅对较低维度数据运行以提高效率)
            if encoding_dim <= 10:
                dbscan_labels, dbscan_silhouette, _ = run_dbscan_clustering(X_ae)
                all_metrics['dbscan_silhouette'] = dbscan_silhouette if dbscan_labels is not None else None
                if dbscan_labels is not None:
                    print(f"AE {encoding_dim}D: DBSCAN 轮廓系数 = {dbscan_silhouette:.4f}")

            # 层次聚类 (仅对较低维度数据运行以提高效率)
            if encoding_dim <= 10:
                hierarchical_labels, hierarchical_silhouette, _ = run_hierarchical_clustering(X_ae)
                all_metrics['hierarchical_silhouette'] = hierarchical_silhouette if hierarchical_labels is not None else None
                if hierarchical_labels is not None:
                    print(f"AE {encoding_dim}D: 层次聚类 轮廓系数 = {hierarchical_silhouette:.4f}")

            # 记录自编码器结果
            result = {
                'Method': 'Autoencoder',
                'Parameters': f'encoding_dim={encoding_dim}, intermediate_dim={intermediate_dim}',
                'Runtime (s)': runtime,
                'Neighbor Preservation': eval_metrics.get('neighbor_preservation'),
                'Trustworthiness': eval_metrics.get('trustworthiness'),
                'Continuity': eval_metrics.get('continuity'),
                'Reconstruction Error': ae_metrics.get('reconstruction_error'),
                'AE Divergence': ae_metrics.get('ae_divergence'),
                'Manifold Divergence': manifold_divergence,
                'KMeans Silhouette': kmeans_silhouette,
                'DBSCAN Silhouette': all_metrics.get('dbscan_silhouette'),
                'Hierarchical Silhouette': all_metrics.get('hierarchical_silhouette'),
                'GNN Model': gnn_model_type,
                'Feature Extraction Time (s)': feature_extraction_time
            }
            ae_results.append(result)

            # 记录聚类结果
            ae_clustering.append({
                'Method': 'KMeans',
                'Dimensionality Reduction': f'Autoencoder {encoding_dim}D (intermediate_dim={intermediate_dim})',
                'Silhouette Score': kmeans_silhouette
            })

            if encoding_dim <= 10 and all_metrics.get('dbscan_silhouette') is not None:
                ae_clustering.append({
                    'Method': 'DBSCAN',
                    'Dimensionality Reduction': f'Autoencoder {encoding_dim}D (intermediate_dim={intermediate_dim})',
                    'Silhouette Score': all_metrics['dbscan_silhouette']
                })

            if encoding_dim <= 10 and all_metrics.get('hierarchical_silhouette') is not None:
                ae_clustering.append({
                    'Method': 'Hierarchical',
                    'Dimensionality Reduction': f'Autoencoder {encoding_dim}D (intermediate_dim={intermediate_dim})',
                    'Silhouette Score': all_metrics['hierarchical_silhouette']
                })

            # 如果是2D，创建可视化
            if encoding_dim == 2 and intermediate_dim == 256:
                plot_2d_embedding(X_ae, output_path,
                                  f'ae_ed{encoding_dim}_id{intermediate_dim}_2d_plot.png',
                                  f'Autoencoder 2D (encoding_dim={encoding_dim}, intermediate_dim={intermediate_dim})')

                plot_2d_embedding(X_ae, output_path,
                                  f'ae_ed{encoding_dim}_id{intermediate_dim}_kmeans_clusters.png',
                                  'Autoencoder 2D with KMeans Clustering', kmeans_labels)

                if 'dbscan_silhouette' in all_metrics and all_metrics['dbscan_silhouette'] is not None:
                    plot_2d_embedding(X_ae, output_path,
                                      f'ae_ed{encoding_dim}_id{intermediate_dim}_dbscan_clusters.png',
                                      'Autoencoder 2D with DBSCAN Clustering', dbscan_labels)

                if 'hierarchical_silhouette' in all_metrics and all_metrics['hierarchical_silhouette'] is not None:
                    plot_2d_embedding(X_ae, output_path,
                                      f'ae_ed{encoding_dim}_id{intermediate_dim}_hierarchical_clusters.png',
                                      'Autoencoder 2D with Hierarchical Clustering', hierarchical_labels)

    # 合并按方法排序的结果
    dimensionality_results = pca_results + umap_results + ae_results
    clustering_results = pca_clustering + umap_clustering + ae_clustering

    # 保存降维结果摘要
    dim_results_df = pd.DataFrame(dimensionality_results)
    dim_results_df.to_csv(os.path.join(output_path, "dimensionality_reduction_results.csv"), index=False)
    print(f"\n降维结果摘要已保存到 {os.path.join(output_path, 'dimensionality_reduction_results.csv')}")

    # 保存聚类结果摘要
    clustering_df = pd.DataFrame(clustering_results)
    clustering_df.to_csv(os.path.join(output_path, "clustering_results.csv"), index=False)
    print(f"聚类结果摘要已保存到 {os.path.join(output_path, 'clustering_results.csv')}")

    # 创建聚类方法比较图
    plot_clustering_comparison(clustering_df, output_path)

    # 执行聚类维度分析
    plot_clustering_by_dimensions(dim_results_df, output_path)

    # 执行相关性分析
    correlation_analysis(dim_results_df, output_path)

    print("\n所有分析完成！结果已保存到", output_path)


def plot_clustering_by_dimensions(results_df, output_path):
    """
    绘制不同降维维度下聚类轮廓系数的比较图

    参数:
    results_df: 降维结果DataFrame
    output_path: 输出目录
    """
    # 提取维度信息
    results_df['Dimension'] = results_df['Parameters'].apply(
        lambda x: int(re.search(r'n_components=(\d+)', x).group(1)) if 'n_components=' in x
        else (int(re.search(r'encoding_dim=(\d+)', x).group(1)) if 'encoding_dim=' in x else None)
    )

    # 对每种降维方法绘制不同维度下的聚类性能
    for method in results_df['Method'].unique():
        method_df = results_df[results_df['Method'] == method].copy()
        if method_df.empty or 'Dimension' not in method_df.columns:
            continue

        plt.figure(figsize=(12, 8))

        # 绘制KMeans轮廓系数
        if 'KMeans Silhouette' in method_df.columns:
            grouped = method_df.groupby('Dimension')['KMeans Silhouette'].mean()
            plt.plot(grouped.index, grouped.values, 'o-', label='KMeans')

        # 绘制DBSCAN轮廓系数
        if 'DBSCAN Silhouette' in method_df.columns:
            grouped = method_df.groupby('Dimension')['DBSCAN Silhouette'].mean()
            plt.plot(grouped.index, grouped.values, 's-', label='DBSCAN')

        # 绘制层次聚类轮廓系数
        if 'Hierarchical Silhouette' in method_df.columns:
            grouped = method_df.groupby('Dimension')['Hierarchical Silhouette'].mean()
            plt.plot(grouped.index, grouped.values, '^-', label='Hierarchical')

        plt.title(f'{method}comparisons under different dimensionality reduction methods')
        plt.xlabel('dimension')
        plt.ylabel('average silhouette score')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 保存图表
        plt.savefig(os.path.join(output_path, f'{method.lower()}_clustering_by_dimension.png'), dpi=300)
        plt.close()

    # 创建一个维度-聚类方法混合比较图
    clustering_columns = ['KMeans Silhouette', 'DBSCAN Silhouette', 'Hierarchical Silhouette']
    available_columns = [col for col in clustering_columns if col in results_df.columns]

    if available_columns and 'Dimension' in results_df.columns:
        # 创建聚类方法-维度比较热图
        pivot_data = pd.pivot_table(
            results_df,
            values=available_columns,
            index='Method',
            columns='Dimension',
            aggfunc='mean'
        )

        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt='.3f')
        plt.title('clustering performance heatmap under different methods and dimensions')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'clustering_dimension_heatmap.png'), dpi=300)
        plt.close()


def correlation_analysis(results_df, output_path):
    """
    分析不同降维和评估指标之间的相关性

    参数:
    results_df: 降维结果DataFrame
    output_path: 输出目录
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\n执行指标相关性分析...")

    # 选择数值列
    numeric_columns = [
        'Runtime (s)', 'Neighbor Preservation', 'Trustworthiness',
        'Continuity', 'Variance Explained', 'Reconstruction Error',
        'PCA Divergence', 'UMAP Divergence', 'Manifold Divergence',
        'AE Divergence'
    ]

    # 过滤存在的列
    valid_columns = [col for col in numeric_columns if col in results_df.columns]

    # 计算相关系数矩阵
    correlation = results_df[valid_columns].corr()

    # 绘制相关性热图
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5)
    plt.title('指标相关性矩阵', fontsize=16)
    plt.tight_layout()

    # 保存相关性矩阵图
    corr_path = os.path.join(output_path, 'metrics_correlation.png')
    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 保存相关性矩阵为CSV
    correlation.to_csv(os.path.join(output_path, 'metrics_correlation.csv'))

    # 分析每种降维方法的性能
    method_comparison = results_df.groupby('Method')[valid_columns].mean()

    # 创建降维方法比较图
    plt.figure(figsize=(14, 8))
    method_comparison.plot(kind='bar', rot=0)
    plt.title('comparisons of average performance under different methods', fontsize=16)
    plt.ylabel('average indicator value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存方法比较图
    plt.savefig(os.path.join(output_path, 'method_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 分析不同维度下的性能
    dimension_patterns = {
        'PCA': r'n_components=(\d+)',
        'UMAP': r'n_components=(\d+)',
        'Autoencoder': r'encoding_dim=(\d+)'
    }

    for method, pattern in dimension_patterns.items():
        import re

        # 筛选当前方法的结果
        method_results = results_df[results_df['Method'] == method].copy()

        # 提取维度信息
        dimensions = []
        for params in method_results['Parameters']:
            match = re.search(pattern, params)
            if match:
                dimensions.append(int(match.group(1)))
            else:
                dimensions.append(np.nan)

        method_results['Dimension'] = dimensions

        # 按维度分组计算平均性能
        dim_performance = method_results.groupby('Dimension')[valid_columns].mean()

        if not dim_performance.empty:
            # 绘制不同维度的性能图
            plt.figure(figsize=(12, 6))
            for metric in ['Neighbor Preservation', 'Trustworthiness', 'Continuity']:
                if metric in dim_performance.columns:
                    plt.plot(dim_performance.index, dim_performance[metric], marker='o', label=metric)

            plt.title(f'{method}Preservation performance in different dimensions', fontsize=14)
            plt.xlabel('dimension')
            plt.ylabel('indicator value')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            # 保存图表
            plt.savefig(os.path.join(output_path, f'{method.lower()}_dimension_performance.png'), dpi=300)
            plt.close()

            # 绘制散度指标
            plt.figure(figsize=(12, 6))
            divergence_metrics = ['Reconstruction Error', 'Manifold Divergence']
            if method == 'PCA':
                divergence_metrics.append('PCA Divergence')
            elif method == 'UMAP':
                divergence_metrics.append('UMAP Divergence')
            elif method == 'Autoencoder':
                divergence_metrics.append('AE Divergence')

            for metric in divergence_metrics:
                if metric in dim_performance.columns:
                    plt.plot(dim_performance.index, dim_performance[metric], marker='o', label=metric)

            plt.title(f'{method}Divergence indicators in different dimensions', fontsize=14)
            plt.xlabel('dimension')
            plt.ylabel('indicator value')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            # 保存图表
            plt.savefig(os.path.join(output_path, f'{method.lower()}_divergence_metrics.png'), dpi=300)
            plt.close()

    # 聚类方法比较分析
    clustering_file = os.path.join(output_path, "clustering_results.csv")
    if os.path.exists(clustering_file):
        clustering_df = pd.read_csv(clustering_file)

        # 比较不同聚类方法在同一降维技术上的性能
        methods = clustering_df['Method'].unique()
        dim_reductions = clustering_df['Dimensionality Reduction'].unique()

        for dim_reduction in dim_reductions:
            subset = clustering_df[clustering_df['Dimensionality Reduction'] == dim_reduction]
            plt.figure(figsize=(10, 6))
            x = np.arange(len(methods))
            width = 0.35

            silhouette_scores = []
            for method in methods:
                method_scores = subset[subset['Method'] == method]['Silhouette Score'].values
                if len(method_scores) > 0:
                    silhouette_scores.append(method_scores[0])
                else:
                    silhouette_scores.append(0)

            plt.bar(x, silhouette_scores, width, label='Silhouette Score')
            plt.xlabel('clustering method')
            plt.ylabel('silhouette Score')
            plt.title(f'{dim_reduction}on comparison of silhouette coefficients of different clustering methods')
            plt.xticks(x, methods)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()

            # 保存图表
            dim_name = dim_reduction.replace(' ', '_').lower()
            plt.savefig(os.path.join(output_path, f'{dim_name}_clustering_comparison.png'), dpi=300)
            plt.close()

    # 生成综合报告
    report_file = os.path.join(output_path, "dimensionality_reduction_report.md")
    with open(report_file, 'w') as f:
        f.write("# 图表示降维与聚类分析报告\n\n")

        f.write("## 降维方法比较\n\n")
        f.write("本报告对比了以下降维方法在图表示数据上的性能：\n")
        f.write("- PCA (主成分分析)\n")
        f.write("- UMAP (统一流形近似与投影)\n")
        f.write("- Autoencoder (自编码器)\n\n")

        f.write("### 平均性能指标\n\n")
        f.write("| 方法 | 邻居保留率 | 可信度 | 连续性 | 重构误差 | 运行时间(秒) |\n")
        f.write("|------|------------|--------|--------|----------|-------------|\n")

        for method in method_comparison.index:
            row = method_comparison.loc[method]
            neighbor = row.get('Neighbor Preservation', 'N/A')
            trust = row.get('Trustworthiness', 'N/A')
            cont = row.get('Continuity', 'N/A')
            recon = row.get('Reconstruction Error', 'N/A')
            runtime = row.get('Runtime (s)', 'N/A')

            neighbor = f"{neighbor:.4f}" if not isinstance(neighbor, str) else neighbor
            trust = f"{trust:.4f}" if not isinstance(trust, str) else trust
            cont = f"{cont:.4f}" if not isinstance(cont, str) else cont
            recon = f"{recon:.4f}" if not isinstance(recon, str) else recon
            runtime = f"{runtime:.2f}" if not isinstance(runtime, str) else runtime

            f.write(f"| {method} | {neighbor} | {trust} | {cont} | {recon} | {runtime} |\n")

        f.write("\n### 散度指标比较\n\n")
        f.write("散度指标反映了降维过程中信息的损失程度，较低的散度值表示更好的维度减少质量。\n\n")

        f.write("| 方法 | 流形散度 | 方法特有散度 |\n")
        f.write("|------|----------|------------|\n")

        for method in method_comparison.index:
            row = method_comparison.loc[method]
            manifold_div = row.get('Manifold Divergence', 'N/A')

            method_div = 'N/A'
            if method == 'PCA' and 'PCA Divergence' in row:
                method_div = row['PCA Divergence']
            elif method == 'UMAP' and 'UMAP Divergence' in row:
                method_div = row['UMAP Divergence']
            elif method == 'Autoencoder' and 'AE Divergence' in row:
                method_div = row['AE Divergence']

            manifold_div = f"{manifold_div:.4f}" if not isinstance(manifold_div, str) else manifold_div
            method_div = f"{method_div:.4f}" if not isinstance(method_div, str) else method_div

            f.write(f"| {method} | {manifold_div} | {method_div} |\n")

        f.write("\n## 聚类方法比较\n\n")
        f.write("在图数据的降维表示上，我们比较了以下聚类方法：\n")
        f.write("- KMeans (K均值聚类)\n")
        f.write("- DBSCAN (基于密度的聚类)\n")
        f.write("- Hierarchical (层次聚类)\n\n")

        f.write("### 轮廓系数比较\n\n")
        f.write("轮廓系数是衡量聚类质量的指标，值越接近1表示聚类质量越高。\n\n")

        if os.path.exists(clustering_file):
            f.write("| 降维方法 | 聚类方法 | 轮廓系数 |\n")
            f.write("|----------|----------|----------|\n")

            for _, row in clustering_df.iterrows():
                dim_red = row['Dimensionality Reduction']
                method = row['Method']
                score = row['Silhouette Score']
                score_str = f"{score:.4f}" if not pd.isna(score) else 'N/A'

                f.write(f"| {dim_red} | {method} | {score_str} |\n")

        f.write("\n## 关键发现\n\n")

        # 找出最佳降维方法
        best_preservation = results_df.loc[
            results_df['Neighbor Preservation'].idxmax() if 'Neighbor Preservation' in results_df else 0]
        best_trust = results_df.loc[results_df['Trustworthiness'].idxmax() if 'Trustworthiness' in results_df else 0]
        best_runtime = results_df.loc[results_df['Runtime (s)'].idxmin() if 'Runtime (s)' in results_df else 0]

        f.write("1. **最佳邻居保留率**：")
        if 'Neighbor Preservation' in results_df:
            f.write(
                f"{best_preservation['Method']} ({best_preservation['Parameters']}) - {best_preservation['Neighbor Preservation']:.4f}\n")
        else:
            f.write("数据不足以确定\n")

        f.write("2. **最佳可信度**：")
        if 'Trustworthiness' in results_df:
            f.write(f"{best_trust['Method']} ({best_trust['Parameters']}) - {best_trust['Trustworthiness']:.4f}\n")
        else:
            f.write("数据不足以确定\n")

        f.write("3. **最快运行时间**：")
        if 'Runtime (s)' in results_df:
            f.write(f"{best_runtime['Method']} ({best_runtime['Parameters']}) - {best_runtime['Runtime (s)']:.2f}秒\n")
        else:
            f.write("数据不足以确定\n")

        # 找出最佳聚类方法
        if os.path.exists(clustering_file):
            best_cluster = clustering_df.loc[
                clustering_df['Silhouette Score'].idxmax() if 'Silhouette Score' in clustering_df else 0]

            f.write("4. **最佳聚类性能**：")
            if 'Silhouette Score' in clustering_df:
                f.write(
                    f"{best_cluster['Method']} on {best_cluster['Dimensionality Reduction']} - {best_cluster['Silhouette Score']:.4f}\n")
            else:
                f.write("数据不足以确定\n")

        f.write("\n## 结论与建议\n\n")

        # 基于结果给出建议
        best_method = method_comparison.index[0]  # 默认第一个
        for metric in ['Neighbor Preservation', 'Trustworthiness', 'Continuity']:
            if metric in method_comparison.columns:
                best_for_metric = method_comparison[metric].idxmax()
                if pd.notna(best_for_metric):
                    best_method = best_for_metric
                    break

        f.write(f"基于多种评估指标的综合分析，{best_method}在保留图数据拓扑结构方面表现最佳。")
        f.write("对于不同的应用场景，我们有以下建议：\n\n")

        f.write("- **可视化目的**：推荐使用UMAP或t-SNE，它们在保留局部结构方面表现优异。\n")
        f.write("- **聚类分析**：KMeans聚类在多种降维结果上表现稳定，是一个可靠的选择。\n")
        f.write("- **特征工程**：PCA提供了最好的计算效率和方差解释能力，适合作为初步降维步骤。\n")
        f.write("- **非线性关系建模**：自编码器能够捕捉复杂的非线性关系，适合高度非线性的图数据。\n")

    print(f"综合分析报告已生成: {report_file}")


def evaluate_embedding(X_original, X_embedded):
    """
    评估降维质量

    参数:
    X_original: 原始高维数据
    X_embedded: 降维后的数据

    返回:
    metrics: 评估指标字典
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    # 初始化指标字典
    metrics = {}

    # 计算邻居保留率
    k = min(20, len(X_original) - 1)  # 邻居数量，不超过样本数减1

    # 在原始空间中寻找k近邻
    nn_orig = NearestNeighbors(n_neighbors=k + 1)  # +1是因为包含自身
    nn_orig.fit(X_original)
    indices_orig = nn_orig.kneighbors(X_original, return_distance=False)

    # 在嵌入空间中寻找k近邻
    nn_embed = NearestNeighbors(n_neighbors=k + 1)
    nn_embed.fit(X_embedded)
    indices_embed = nn_embed.kneighbors(X_embedded, return_distance=False)

    # 计算邻居保留率
    preserved_neighbors = 0
    total_neighbors = 0

    for i in range(len(X_original)):
        # 排除自身(第一个邻居)
        orig_neighbors = set(indices_orig[i][1:])
        embed_neighbors = set(indices_embed[i][1:])
        preserved_neighbors += len(orig_neighbors.intersection(embed_neighbors))
        total_neighbors += k

    metrics['neighbor_preservation'] = preserved_neighbors / total_neighbors

    # 计算可信度(Trustworthiness)
    trustworthiness = calculate_trustworthiness(X_original, X_embedded)
    metrics['trustworthiness'] = trustworthiness

    # 计算连续性(Continuity)
    continuity = calculate_continuity(X_original, X_embedded)
    metrics['continuity'] = continuity

    print(f"邻居保留率: {metrics['neighbor_preservation']:.4f}")
    print(f"可信度: {metrics['trustworthiness']:.4f}")
    print(f"连续性: {metrics['continuity']:.4f}")

    return metrics


def calculate_trustworthiness(X_original, X_embedded, n_neighbors=5):
    """
    计算降维的可信度

    参数:
    X_original: 原始高维数据
    X_embedded: 降维后的数据
    n_neighbors: 近邻数量

    返回:
    trust: 可信度得分(0-1，越高越好)
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    n = X_original.shape[0]
    n_neighbors = min(n_neighbors, n - 1)

    # 在原始空间中寻找k近邻
    nn_orig = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1是因为包含自身
    nn_orig.fit(X_original)
    ind_orig = nn_orig.kneighbors(X_original, return_distance=False)

    # 在嵌入空间中寻找k近邻
    nn_embed = NearestNeighbors(n_neighbors=n)  # 查找所有点
    nn_embed.fit(X_embedded)
    ind_embed = nn_embed.kneighbors(X_embedded, return_distance=False)

    # 计算可信度
    trustworthiness = 0.0

    for i in range(n):
        # 嵌入空间中的近邻但不在原始空间中的近邻
        embed_neighbors = set(ind_embed[i][1:n_neighbors + 1])  # 排除自身
        orig_neighbors = set(ind_orig[i][1:])  # 排除自身
        false_neighbors = embed_neighbors - orig_neighbors

        # 计算违规分数
        ranks = []
        for j in false_neighbors:
            # 找到j在原始空间中的秩
            rank_j_orig = np.where(ind_orig[i] == j)[0]
            if len(rank_j_orig) == 0:  # j不在原始空间的k近邻中
                rank_j_orig = n  # 最大可能秩
            else:
                rank_j_orig = rank_j_orig[0]
            ranks.append(rank_j_orig - n_neighbors)

        trustworthiness += np.sum(ranks)

    # 归一化
    trustworthiness = 1.0 - (2.0 / (n * n_neighbors * (2.0 * n - 3.0 * n_neighbors - 1.0)) * trustworthiness)

    return trustworthiness


def calculate_continuity(X_original, X_embedded, n_neighbors=5):
    """
    计算降维的连续性

    参数:
    X_original: 原始高维数据
    X_embedded: 降维后的数据
    n_neighbors: 近邻数量

    返回:
    cont: 连续性得分(0-1，越高越好)
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    n = X_original.shape[0]
    n_neighbors = min(n_neighbors, n - 1)

    # 在原始空间中寻找k近邻
    nn_orig = NearestNeighbors(n_neighbors=n)  # 查找所有点
    nn_orig.fit(X_original)
    ind_orig = nn_orig.kneighbors(X_original, return_distance=False)

    # 在嵌入空间中寻找k近邻
    nn_embed = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1是因为包含自身
    nn_embed.fit(X_embedded)
    ind_embed = nn_embed.kneighbors(X_embedded, return_distance=False)

    # 计算连续性
    continuity = 0.0

    for i in range(n):
        # 原始空间中的近邻但不在嵌入空间中的近邻
        orig_neighbors = set(ind_orig[i][1:n_neighbors + 1])  # 排除自身
        embed_neighbors = set(ind_embed[i][1:])  # 排除自身
        missing_neighbors = orig_neighbors - embed_neighbors

        # 计算违规分数
        ranks = []
        for j in missing_neighbors:
            # 找到j在嵌入空间中的秩
            rank_j_embed = np.where(ind_embed[i] == j)[0]
            if len(rank_j_embed) == 0:  # j不在嵌入空间的k近邻中
                rank_j_embed = n  # 最大可能秩
            else:
                rank_j_embed = rank_j_embed[0]
            ranks.append(rank_j_embed - n_neighbors)

        continuity += np.sum(ranks)

    # 归一化
    continuity = 1.0 - (2.0 / (n * n_neighbors * (2.0 * n - 3.0 * n_neighbors - 1.0)) * continuity)

    return continuity


if __name__ == "__main__":
    # 设置输入和输出路径
    graph_dir = r"D:\materproject\all-reps\GO_qdots\GO_qdots-graph"
    output_path = r"D:\materproject\single-rep-rd\graph\GO_qdots"

    # 选择GNN模型类型: 'GCN', 'GAT', 或 'GraphSAGE'
    gnn_model_type = 'GCN'

    run_graph_dimensionality_reduction(graph_dir, output_path, gnn_model_type)