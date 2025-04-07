import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def perform_pca(X, n_components):
    """
    执行PCA降维，添加散度计算

    参数:
    X: 输入数据矩阵
    n_components: 降维后的维度

    返回:
    X_pca: 降维后的数据
    pca: PCA模型对象
    metrics: 模型特有的指标
    """
    from pub_func.table.divergence_metrics import calculate_pca_divergence

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # 计算额外的指标
    X_reconstructed = pca.inverse_transform(X_pca)
    reconstruction_error = np.mean(np.square(X - X_reconstructed))
    variance_explained = np.sum(pca.explained_variance_ratio_)

    # 计算PCA散度
    pca_divergence = calculate_pca_divergence(X, X_pca, pca)

    metrics = {
        'reconstruction_error': reconstruction_error,
        'variance_explained': variance_explained,
        'pca_divergence': pca_divergence
    }

    print(f"PCA (n_components={n_components}):")
    print(f"  重构误差: {reconstruction_error:.4f}")
    print(f"  解释方差比例: {variance_explained:.4f}")
    print(f"  PCA散度: {pca_divergence:.6f}")

    return X_pca, pca, metrics


def perform_tsne(X, n_components, perplexity):
    """
    执行t-SNE降维，添加散度计算

    参数:
    X: 输入数据矩阵
    n_components: 降维后的维度
    perplexity: t-SNE的困惑度参数

    返回:
    X_tsne: 降维后的数据
    tsne: t-SNE模型对象
    metrics: 模型特有的指标
    """
    from pub_func.table.divergence_metrics import calculate_tsne_divergence

    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # 计算t-SNE散度
    tsne_divergence = calculate_tsne_divergence(X, X_tsne, perplexity)

    metrics = {
        'tsne_divergence': tsne_divergence
    }

    print(f"t-SNE (perplexity={perplexity}, n_components={n_components}):")
    print(f"  降维完成")
    print(f"  t-SNE散度: {tsne_divergence:.6f}")

    return X_tsne, tsne, metrics


def perform_umap(X, n_components, n_neighbors, min_dist):
    """
    执行UMAP降维，添加散度计算

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
    from pub_func.table.divergence_metrics import calculate_umap_divergence

    umap_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                             n_components=n_components, random_state=42)
    X_umap = umap_reducer.fit_transform(X)

    # 计算UMAP散度
    umap_divergence = calculate_umap_divergence(X, X_umap, n_neighbors)

    metrics = {
        'umap_divergence': umap_divergence
    }

    print(f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}):")
    print(f"  降维完成")
    print(f"  UMAP散度: {umap_divergence:.6f}")

    return X_umap, umap_reducer, metrics


def perform_vae(X, latent_dim, intermediate_dim=256, epochs=20, batch_size=32):
    """
    Execute VAE dimensionality reduction with enhanced KL divergence tracking

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
    from pub_func.table.divergence_metrics import calculate_vae_kl_divergence, numpy_calculate_vae_kl_divergence

    # Create VAE model
    vae, encoder, decoder = create_vae(X.shape[1], latent_dim, intermediate_dim)

    # Create callback to track KL divergence during training
    kl_history = []

    class KLDivergenceCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            z_mean, z_log_var, _ = encoder.predict(X)
            try:
                kl_div = calculate_vae_kl_divergence(z_mean, z_log_var)
            except:
                kl_div = numpy_calculate_vae_kl_divergence(z_mean, z_log_var)
            kl_history.append(kl_div)
            print(f"Epoch {epoch + 1}: KL Divergence = {kl_div:.6f}")

    # Train VAE with KL tracking
    history = vae.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0,
                      callbacks=[KLDivergenceCallback()])

    # Get embeddings and latent space parameters
    z_mean, z_log_var, X_vae = encoder.predict(X)

    # Calculate final KL divergence
    try:
        kl_divergence = calculate_vae_kl_divergence(z_mean, z_log_var)
    except:
        kl_divergence = numpy_calculate_vae_kl_divergence(z_mean, z_log_var)

    # Calculate reconstruction error
    X_reconstructed = decoder.predict(X_vae)
    reconstruction_error = np.mean(np.square(X - X_reconstructed))

    metrics = {
        'kl_divergence': kl_divergence,
        'kl_divergence_history': kl_history,
        'reconstruction_error': reconstruction_error,
        'final_loss': history.history['loss'][-1] if history.history['loss'] else None
    }

    print(f"VAE (latent_dim={latent_dim}, intermediate_dim={intermediate_dim}):")
    print(f"  Training completed ({epochs} epochs)")
    print(f"  KL divergence: {kl_divergence:.6f}")
    print(f"  Reconstruction error: {reconstruction_error:.4f}")

    # Print KL divergence trend
    if len(kl_history) > 0:
        print(f"  Initial KL divergence: {kl_history[0]:.6f}")
        print(f"  Final KL divergence: {kl_history[-1]:.6f}")
        if len(kl_history) > 1:
            kl_change = (kl_history[-1] - kl_history[0]) / kl_history[0] * 100
            print(f"  KL divergence change: {kl_change:.2f}%")

    return X_vae, encoder, metrics

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


def create_autoencoder(input_dim, latent_dim, intermediate_dim=256):
    """
    Create a standard autoencoder model for dimensionality reduction

    Parameters:
    input_dim: Input feature dimension
    latent_dim: Latent space dimension
    intermediate_dim: Intermediate layer dimension

    Returns:
    autoencoder: Complete autoencoder model
    encoder: Encoder part of the model
    decoder: Decoder part of the model
    """
    from tensorflow.keras import layers, Model
    import tensorflow.keras.backend as K

    # Encoder
    encoder_inputs = layers.Input(shape=(input_dim,), name='encoder_input')
    x = layers.Dense(intermediate_dim, activation='relu')(encoder_inputs)
    encoded = layers.Dense(latent_dim, name='encoded')(x)

    # Decoder
    decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')
    x = layers.Dense(intermediate_dim, activation='relu')(decoder_input)
    decoded = layers.Dense(input_dim, activation='linear')(x)

    # Models
    encoder = Model(encoder_inputs, encoded, name='encoder')
    decoder = Model(decoder_input, decoded, name='decoder')

    # Autoencoder (encoder + decoder)
    autoencoder_input = layers.Input(shape=(input_dim,))
    encoded_output = encoder(autoencoder_input)
    decoded_output = decoder(encoded_output)

    autoencoder = Model(autoencoder_input, decoded_output, name='autoencoder')
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder, decoder


def perform_autoencoder(X, latent_dim, intermediate_dim=256, epochs=50, batch_size=32):
    """
    Execute standard autoencoder dimensionality reduction with divergence measure

    Parameters:
    X: Input data matrix
    latent_dim: Latent space dimension
    intermediate_dim: Intermediate layer dimension
    epochs: Number of training epochs
    batch_size: Batch size

    Returns:
    X_ae: Reduced dimensionality data
    encoder: Encoder model
    metrics: Model-specific metrics including divergence
    """
    from pub_func.table.divergence_metrics import calculate_autoencoder_divergence

    # Create autoencoder model
    autoencoder, encoder, decoder = create_autoencoder(X.shape[1], latent_dim, intermediate_dim)

    # Train autoencoder
    history = autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0)

    # Get embeddings
    X_ae = encoder.predict(X)

    # Calculate reconstruction error
    X_reconstructed = autoencoder.predict(X)
    reconstruction_error = np.mean(np.square(X - X_reconstructed))

    # Calculate autoencoder divergence
    ae_divergence = calculate_autoencoder_divergence(X, encoder, decoder)

    metrics = {
        'reconstruction_error': reconstruction_error,
        'ae_divergence': ae_divergence,
        'final_loss': history.history['loss'][-1] if history.history['loss'] else None
    }

    print(f"Autoencoder (latent_dim={latent_dim}, intermediate_dim={intermediate_dim}):")
    print(f"  Training completed ({epochs} epochs)")
    print(f"  Reconstruction error: {reconstruction_error:.4f}")
    print(f"  Autoencoder divergence: {ae_divergence:.6f}")

    return X_ae, encoder, metrics