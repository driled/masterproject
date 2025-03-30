import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# 不使用 seaborn，只用 matplotlib

# 设置随机种子以确保结果可重现
np.random.seed(42)
tf.random.set_seed(42)

# 设置输入和输出路径
input_path = r"D:\materproject\all-reps\ESOL\ESOL-table"
output_path = r"D:\materproject\single-rep-rd\table\ESOL"

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)


# 加载数据
def load_data(path):
    # 查找目录中的CSV文件
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    # 加载第一个找到的CSV文件
    file_path = os.path.join(path, files[0])
    print(f"Loading data from {file_path}")

    df = pd.read_csv(file_path)

    # 检查是否有名称或ID列，如果有则移除
    non_feature_cols = []
    for col in df.columns:
        if 'name' in col.lower() or 'id' in col.lower() or 'smiles' in col.lower() or 'target' in col.lower() or 'label' in col.lower():
            non_feature_cols.append(col)

    # 保存标签列和非特征列
    labels = df[
        'measured log solubility in mols per litre'] if 'measured log solubility in mols per litre' in df.columns else None
    features = df.drop(columns=non_feature_cols, errors='ignore')

    print(f"Data shape: {features.shape}")
    return features, labels


# 定义评估指标
def evaluate_embedding(X, X_embedded, labels=None):
    """评估降维结果的质量"""
    metrics = {}

    # 1. 重构误差 (仅适用于PCA和VAE，但现在我们直接传递降维后的数据，不再传递模型对象)
    # 因此这个检查不再适用，我们将在各方法中单独计算重构误差

    # 2. 轮廓系数 (如果有标签)
    if labels is not None and len(np.unique(labels)) > 1:
        try:
            metrics['silhouette_score'] = silhouette_score(X_embedded, labels)
        except:
            metrics['silhouette_score'] = None

    # 3. 邻居保持率
    k = min(20, len(X) - 1)  # 邻居数量，不超过样本数减1

    # 计算原始空间中的k近邻
    nbrs_orig = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances_orig, indices_orig = nbrs_orig.kneighbors(X)

    # 计算嵌入空间中的k近邻
    nbrs_embedded = NearestNeighbors(n_neighbors=k + 1).fit(X_embedded)
    distances_embedded, indices_embedded = nbrs_embedded.kneighbors(X_embedded)

    # 计算邻居保持率
    preserved_neighbors = 0
    total_neighbors = 0

    for i in range(len(X)):
        # 跳过第一个邻居（即样本本身）
        orig_neighbors = set(indices_orig[i][1:])
        embedded_neighbors = set(indices_embedded[i][1:])
        preserved_neighbors += len(orig_neighbors.intersection(embedded_neighbors))
        total_neighbors += k

    metrics['neighbor_preservation'] = preserved_neighbors / total_neighbors

    return metrics


# VAE模型定义 - 完全使用Keras层级API
def create_vae(input_dim, latent_dim, intermediate_dim=256):
    from tensorflow.keras import layers, Model
    import tensorflow.keras.backend as K

    # 自定义VAE损失层
    class VAELoss(layers.Layer):
        def __init__(self, **kwargs):
            super(VAELoss, self).__init__(**kwargs)

        def call(self, inputs):
            x_input, x_decoded, z_mean, z_log_var = inputs
            # 重构损失
            reconstruction_loss = K.mean(K.square(x_input - x_decoded), axis=-1)
            # KL散度
            kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            # 总损失
            total_loss = K.mean(reconstruction_loss + kl_loss)
            # 将损失作为层的输出
            self.add_loss(total_loss)
            # 返回解码的输出
            return x_decoded

    # 采样层
    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = K.shape(z_mean)[0]
            dim = K.shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

    # 编码器
    encoder_inputs = layers.Input(shape=(input_dim,), name='encoder_input')
    x = layers.Dense(intermediate_dim, activation='relu')(encoder_inputs)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])

    # 解码器
    decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')
    x = layers.Dense(intermediate_dim, activation='relu')(decoder_input)
    decoder_output = layers.Dense(input_dim, activation='linear')(x)

    # 定义各个模型
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(decoder_input, decoder_output, name='decoder')

    # VAE模型
    vae_input = layers.Input(shape=(input_dim,))
    z_mean, z_log_var, z = encoder(vae_input)
    vae_output = decoder(z)

    # 添加损失
    vae_output = VAELoss()([vae_input, vae_output, z_mean, z_log_var])
    vae = Model(vae_input, vae_output)
    vae.compile(optimizer='adam')

    return vae, encoder, decoder


# 执行降维并评估
def run_dimensionality_reduction():
    # 加载数据
    X, y = load_data(input_path)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 存储结果
    results = []

    # 1. PCA降维
    print("\n执行PCA降维...")
    for n_components in [2, 5, 10, 20]:
        print(f"  n_components = {n_components}")
        start_time = time.time()
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        runtime = time.time() - start_time

        # 保存结果
        output_file = os.path.join(output_path, f"pca_{n_components}d.csv")
        pd.DataFrame(X_pca).to_csv(output_file, index=False)

        # 评估
        metrics = evaluate_embedding(X_scaled, X_pca, y)

        # 添加PCA特有的指标 - 方差解释率
        # 计算重构误差（仅适用于PCA）
        X_reconstructed = pca.inverse_transform(X_pca)
        metrics['reconstruction_error'] = np.mean(np.square(X_scaled - X_reconstructed))

        # 计算方差解释率
        metrics['variance_explained'] = np.sum(pca.explained_variance_ratio_)
        metrics['variance_explained'] = np.sum(pca.explained_variance_ratio_)

        # 记录结果
        results.append({
            'method': 'PCA',
            'params': f'n_components={n_components}',
            'runtime': runtime,
            'metrics': metrics
        })

        # 如果是2D，创建可视化
        if n_components == 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
            plt.colorbar(label='ESOL值')
            plt.title('PCA 2D投影')
            plt.savefig(os.path.join(output_path, 'pca_2d_plot.png'))
            plt.close()

    # 2. t-SNE降维
    print("\n执行t-SNE降维...")
    for perplexity in [5, 30, 50]:
        for n_components in [2, 3]:
            print(f"  perplexity = {perplexity}, n_components = {n_components}")
            start_time = time.time()
            tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=1000, random_state=42)
            X_tsne = tsne.fit_transform(X_scaled)
            runtime = time.time() - start_time

            # 保存结果
            output_file = os.path.join(output_path, f"tsne_perp{perplexity}_{n_components}d.csv")
            pd.DataFrame(X_tsne).to_csv(output_file, index=False)

            # 评估
            metrics = evaluate_embedding(X_scaled, X_tsne, y)

            # 记录结果
            results.append({
                'method': 't-SNE',
                'params': f'perplexity={perplexity}, n_components={n_components}',
                'runtime': runtime,
                'metrics': metrics
            })

            # 如果是2D，创建可视化
            if n_components == 2:
                plt.figure(figsize=(10, 8))
                plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
                plt.colorbar(label='ESOL值')
                plt.title(f't-SNE 2D投影 (perplexity={perplexity})')
                plt.savefig(os.path.join(output_path, f'tsne_perp{perplexity}_2d_plot.png'))
                plt.close()

    # 3. UMAP降维
    print("\n执行UMAP降维...")
    for n_neighbors in [5, 15, 30]:
        for min_dist in [0.1, 0.5]:
            for n_components in [2, 5]:
                print(f"  n_neighbors = {n_neighbors}, min_dist = {min_dist}, n_components = {n_components}")
                start_time = time.time()
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                    n_components=n_components, random_state=42)
                X_umap = reducer.fit_transform(X_scaled)
                runtime = time.time() - start_time

                # 保存结果
                output_file = os.path.join(output_path, f"umap_nn{n_neighbors}_md{min_dist}_{n_components}d.csv")
                pd.DataFrame(X_umap).to_csv(output_file, index=False)

                # 评估
                metrics = evaluate_embedding(X_scaled, X_umap, y)

                # 记录结果
                results.append({
                    'method': 'UMAP',
                    'params': f'n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}',
                    'runtime': runtime,
                    'metrics': metrics
                })

                # 如果是2D，创建可视化
                if n_components == 2:
                    plt.figure(figsize=(10, 8))
                    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.7)
                    plt.colorbar(label='ESOL值')
                    plt.title(f'UMAP 2D投影 (n_neighbors={n_neighbors}, min_dist={min_dist})')
                    plt.savefig(os.path.join(output_path, f'umap_nn{n_neighbors}_md{min_dist}_2d_plot.png'))
                    plt.close()

    # 4. VAE降维
    print("\n执行VAE降维...")
    for latent_dim in [2, 5, 10]:
        for intermediate_dim in [128, 256]:
            print(f"  latent_dim = {latent_dim}, intermediate_dim = {intermediate_dim}")
            start_time = time.time()

            # 创建VAE模型
            vae, encoder, decoder = create_vae(X_scaled.shape[1], latent_dim, intermediate_dim)

            # 训练VAE
            vae.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=0)

            # 获取嵌入
            _, _, X_vae = encoder.predict(X_scaled)
            runtime = time.time() - start_time

            # 保存结果
            output_file = os.path.join(output_path, f"vae_ld{latent_dim}_id{intermediate_dim}.csv")
            pd.DataFrame(X_vae).to_csv(output_file, index=False)

            # 评估
            metrics = evaluate_embedding(X_scaled, X_vae, y)

            # 记录结果
            results.append({
                'method': 'VAE',
                'params': f'latent_dim={latent_dim}, intermediate_dim={intermediate_dim}',
                'runtime': runtime,
                'metrics': metrics
            })

            # 如果是2D，创建可视化
            if latent_dim == 2:
                plt.figure(figsize=(10, 8))
                plt.scatter(X_vae[:, 0], X_vae[:, 1], c=y, cmap='viridis', alpha=0.7)
                plt.colorbar(label='ESOL值')
                plt.title(f'VAE 2D投影 (latent_dim={latent_dim}, intermediate_dim={intermediate_dim})')
                plt.savefig(os.path.join(output_path, f'vae_ld{latent_dim}_id{intermediate_dim}_2d_plot.png'))
                plt.close()

    # 保存结果摘要
    results_df = pd.DataFrame({
        'Method': [r['method'] for r in results],
        'Parameters': [r['params'] for r in results],
        'Runtime (s)': [r['runtime'] for r in results],
        'Neighbor Preservation': [r['metrics'].get('neighbor_preservation', None) for r in results],
        'Silhouette Score': [r['metrics'].get('silhouette_score', None) for r in results],
        'Variance Explained': [r['metrics'].get('variance_explained', None) for r in results],
    })

    results_df.to_csv(os.path.join(output_path, 'dimensionality_reduction_results.csv'), index=False)

    # 创建结果可视化 (使用 matplotlib 替代 seaborn)
    # 运行时间比较
    methods = results_df['Method'].unique()
    fig, ax = plt.subplots(figsize=(14, 7))

    bar_width = 0.15
    index = np.arange(len(methods))
    param_groups = results_df.groupby(['Method', 'Parameters'])

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0

    # 为每个方法-参数组合创建柱状图
    for method in methods:
        method_params = results_df[results_df['Method'] == method]['Parameters'].unique()
        for i, params in enumerate(method_params):
            data = results_df[(results_df['Method'] == method) & (results_df['Parameters'] == params)]
            pos = index[np.where(methods == method)[0][0]] + (i - len(method_params) / 2 + 0.5) * bar_width
            ax.bar(pos, data['Runtime (s)'].values[0], bar_width,
                   label=f"{method}: {params}", color=colors[color_idx % len(colors)])
            color_idx += 1

    ax.set_xlabel('Method')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('降维方法运行时间比较')
    ax.set_xticks(index)
    ax.set_xticklabels(methods, rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'runtime_comparison.png'))
    plt.close()

    # 邻居保持率比较
    fig, ax = plt.subplots(figsize=(14, 7))
    color_idx = 0

    for method in methods:
        method_params = results_df[results_df['Method'] == method]['Parameters'].unique()
        for i, params in enumerate(method_params):
            data = results_df[(results_df['Method'] == method) & (results_df['Parameters'] == params)]
            pos = index[np.where(methods == method)[0][0]] + (i - len(method_params) / 2 + 0.5) * bar_width
            ax.bar(pos, data['Neighbor Preservation'].values[0], bar_width,
                   label=f"{method}: {params}", color=colors[color_idx % len(colors)])
            color_idx += 1

    ax.set_xlabel('Method')
    ax.set_ylabel('Neighbor Preservation')
    ax.set_title('降维方法邻居保持率比较')
    ax.set_xticks(index)
    ax.set_xticklabels(methods, rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'neighbor_preservation_comparison.png'))
    plt.close()

    # 轮廓系数比较 (如果有数据)
    if any(~results_df['Silhouette Score'].isna()):
        fig, ax = plt.subplots(figsize=(14, 7))
        color_idx = 0

        for method in methods:
            method_params = results_df[results_df['Method'] == method]['Parameters'].unique()
            for i, params in enumerate(method_params):
                data = results_df[(results_df['Method'] == method) & (results_df['Parameters'] == params)]
                if not np.isnan(data['Silhouette Score'].values[0]):
                    pos = index[np.where(methods == method)[0][0]] + (i - len(method_params) / 2 + 0.5) * bar_width
                    ax.bar(pos, data['Silhouette Score'].values[0], bar_width,
                           label=f"{method}: {params}", color=colors[color_idx % len(colors)])
                color_idx += 1

        ax.set_xlabel('Method')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('降维方法轮廓系数比较')
        ax.set_xticks(index)
        ax.set_xticklabels(methods, rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'silhouette_score_comparison.png'))
        plt.close()

    print(f"\n所有分析已完成！结果已保存到 {output_path}")
    return results_df


if __name__ == "__main__":
    results = run_dimensionality_reduction()

    # 打印结果摘要
    print("\n降维结果摘要：")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results)