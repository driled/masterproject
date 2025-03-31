import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


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


def perform_tsne(X, n_components, perplexity):
    """
    执行t-SNE降维

    参数:
    X: 输入数据矩阵
    n_components: 降维后的维度
    perplexity: t-SNE的困惑度参数

    返回:
    X_tsne: 降维后的数据
    tsne: t-SNE模型对象
    metrics: 模型特有的指标（t-SNE没有特有指标）
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X)

    print(f"t-SNE (perplexity={perplexity}, n_components={n_components}):")
    print(f"  降维完成")

    return X_tsne, tsne, {}


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
    metrics: 模型特有的指标（UMAP没有特有指标）
    """
    umap_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                             n_components=n_components, random_state=42)
    X_umap = umap_reducer.fit_transform(X)

    print(f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}):")
    print(f"  降维完成")

    return X_umap, umap_reducer, {}


def create_vae(input_dim, latent_dim, intermediate_dim=256):
    """
    创建变分自编码器(VAE)模型

    参数:
    input_dim: 输入特征维度
    latent_dim: 潜在空间维度
    intermediate_dim: 中间层维度

    返回:
    vae: 完整的VAE模型
    encoder: 编码器部分
    decoder: 解码器部分
    """
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


def perform_vae(X, latent_dim, intermediate_dim=256, epochs=50, batch_size=32):
    """
    Execute VAE dimensionality reduction with KL divergence tracking

    Parameters:
    X: Input data matrix
    latent_dim: Latent space dimension
    intermediate_dim: Intermediate layer dimension
    epochs: Number of training epochs
    batch_size: Batch size

    Returns:
    X_vae: Reduced dimensionality data
    encoder: Encoder model
    metrics: Model-specific metrics including KL divergence
    """
    # Create VAE model
    vae, encoder, decoder = create_vae(X.shape[1], latent_dim, intermediate_dim)

    # Train VAE
    history = vae.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0)

    # Get embeddings and latent space parameters
    z_mean, z_log_var, X_vae = encoder.predict(X)

    # Calculate KL divergence
    from tensorflow.keras import backend as K
    kl_divergence = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))

    # Convert tensor to Python float if needed
    if hasattr(kl_divergence, 'numpy'):
        kl_divergence = float(kl_divergence.numpy())

    # Calculate reconstruction error
    X_reconstructed = decoder.predict(X_vae)
    reconstruction_error = np.mean(np.square(X - X_reconstructed))

    metrics = {
        'kl_divergence': kl_divergence,
        'reconstruction_error': reconstruction_error,
        'final_loss': history.history['loss'][-1] if history.history['loss'] else None
    }

    print(f"VAE (latent_dim={latent_dim}, intermediate_dim={intermediate_dim}):")
    print(f"  Training completed ({epochs} epochs)")
    print(f"  KL divergence: {kl_divergence:.4f}")
    print(f"  Reconstruction error: {reconstruction_error:.4f}")

    return X_vae, encoder, metrics


def calculate_kl_divergence(z_mean, z_log_var):
    """
    Calculate the KL divergence for a VAE model

    Parameters:
    z_mean: Mean values of the latent space
    z_log_var: Log variance values of the latent space

    Returns:
    kl_loss: KL divergence loss (scalar value)
    """
    import tensorflow as tf
    import tensorflow.keras.backend as K

    # KL divergence between the learned distribution and a standard normal distribution
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))

    return float(kl_loss.numpy())  # Convert tensor to Python float