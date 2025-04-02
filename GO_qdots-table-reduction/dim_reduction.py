import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap







import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from tensorflow.keras import layers, Model, callbacks
import tensorflow.keras.backend as K


def perform_pca(X, n_components):
    """
    Perform PCA dimensionality reduction

    Parameters:
    X: Input data matrix
    n_components: Reduced dimension

    Returns:
    X_pca: Reduced dimensionality data
    pca: PCA model object
    metrics: Model-specific metrics
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Calculate additional metrics
    X_reconstructed = pca.inverse_transform(X_pca)
    reconstruction_error = np.mean(np.square(X - X_reconstructed))
    variance_explained = np.sum(pca.explained_variance_ratio_)

    metrics = {
        'reconstruction_error': reconstruction_error,
        'variance_explained': variance_explained
    }

    print(f"PCA (n_components={n_components}):")
    print(f"  Reconstruction error: {reconstruction_error:.4f}")
    print(f"  Explained variance ratio: {variance_explained:.4f}")

    return X_pca, pca, metrics


def perform_tsne(X, n_components, perplexity):
    """
    Perform t-SNE dimensionality reduction

    Parameters:
    X: Input data matrix
    n_components: Reduced dimension
    perplexity: t-SNE perplexity parameter

    Returns:
    X_tsne: Reduced dimensionality data
    tsne: t-SNE model object
    metrics: Model-specific metrics (t-SNE has no specific metrics)
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X)

    print(f"t-SNE (perplexity={perplexity}, n_components={n_components}):")
    print(f"  Dimensionality reduction completed")

    return X_tsne, tsne, {}


def perform_umap(X, n_components, n_neighbors, min_dist):
    """
    Perform UMAP dimensionality reduction

    Parameters:
    X: Input data matrix
    n_components: Reduced dimension
    n_neighbors: Local neighbor count
    min_dist: Minimum distance parameter

    Returns:
    X_umap: Reduced dimensionality data
    umap_reducer: UMAP model object
    metrics: Model-specific metrics (UMAP has no specific metrics)
    """
    umap_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                             n_components=n_components, random_state=42)
    X_umap = umap_reducer.fit_transform(X)

    print(f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}):")
    print(f"  Dimensionality reduction completed")

    return X_umap, umap_reducer, {}


# Custom VAE loss layer with gradient clipping
class VAELoss(layers.Layer):
    def __init__(self, **kwargs):
        super(VAELoss, self).__init__(**kwargs)

    def call(self, inputs):
        x_input, x_decoded, z_mean, z_log_var = inputs
        # Reconstruction loss
        reconstruction_loss = K.mean(K.square(x_input - x_decoded), axis=-1)
        # KL divergence with clipping to prevent NaN
        z_log_var_clipped = K.clip(z_log_var, -20, 20)  # Prevent extreme values
        kl_loss = -0.5 * K.mean(1 + z_log_var_clipped - K.square(z_mean) - K.exp(z_log_var_clipped), axis=-1)
        # Total loss
        total_loss = K.mean(reconstruction_loss + kl_loss)
        # Add loss as layer output
        self.add_loss(total_loss)
        # Return decoded output
        return x_decoded


# Sampling layer with numerical stability improvements
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        # Clip to prevent extreme values
        z_log_var_clipped = K.clip(z_log_var, -20, 20)
        return z_mean + K.exp(0.5 * z_log_var_clipped) * epsilon


def create_vae(input_dim, latent_dim, intermediate_dim=256, dropout_rate=0.2):
    """
    Create variational autoencoder (VAE) model with improved stability

    Parameters:
    input_dim: Input feature dimension
    latent_dim: Latent space dimension
    intermediate_dim: Intermediate layer dimension
    dropout_rate: Dropout rate for regularization

    Returns:
    vae: Complete VAE model
    encoder: Encoder part
    decoder: Decoder part
    """
    # Encoder
    encoder_inputs = layers.Input(shape=(input_dim,), name='encoder_input')

    # Add batch normalization and dropout for stability
    x = layers.BatchNormalization()(encoder_inputs)
    x = layers.Dense(intermediate_dim, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Add an additional layer for complex relationships
    x = layers.Dense(intermediate_dim // 2, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var',
                             kernel_initializer='zeros',  # Initialize with zeros for stability
                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

    z = Sampling()([z_mean, z_log_var])

    # Decoder
    decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')

    # Mirror encoder architecture
    x = layers.Dense(intermediate_dim // 2, activation='relu')(decoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(intermediate_dim, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    decoder_output = layers.Dense(input_dim, activation='linear')(x)

    # Define models
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(decoder_input, decoder_output, name='decoder')

    # VAE model
    vae_input = layers.Input(shape=(input_dim,))
    z_mean, z_log_var, z = encoder(vae_input)
    vae_output = decoder(z)

    # Add loss
    vae_output = VAELoss()([vae_input, vae_output, z_mean, z_log_var])
    vae = Model(vae_input, vae_output)

    # Use Adam optimizer with reduced learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    vae.compile(optimizer=optimizer)

    return vae, encoder, decoder


def perform_vae(X, latent_dim, intermediate_dim=256, epochs=50, batch_size=32):
    """
    Execute VAE dimensionality reduction with improved stability and NaN handling

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
    # Create VAE model with improved stability
    vae, encoder, decoder = create_vae(X.shape[1], latent_dim, intermediate_dim, dropout_rate=0.2)

    # Add callbacks for early stopping and model saving
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        ),
        # Add learning rate reduction on plateau
        callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
    ]

    # Train VAE with validation split
    try:
        history = vae.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=callbacks_list,
            verbose=0
        )

        print(f"VAE training completed in {len(history.history['loss'])} epochs")

        # Get embeddings and latent space parameters
        z_mean, z_log_var, X_vae = encoder.predict(X)

        # Check for NaN values
        if np.isnan(X_vae).any():
            print("Warning: NaN values detected in VAE output. Attempting recovery...")

            # Try training with smaller batch size and learning rate
            tf.keras.backend.clear_session()
            vae, encoder, decoder = create_vae(X.shape[1], latent_dim, intermediate_dim, dropout_rate=0.3)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            vae.compile(optimizer=optimizer)

            # Try with smaller batch size
            history = vae.fit(X, X, epochs=epochs // 2, batch_size=max(8, batch_size // 4), verbose=0)
            z_mean, z_log_var, X_vae = encoder.predict(X)

            # If still have NaNs, fall back to PCA
            if np.isnan(X_vae).any():
                print("Still detected NaN values. Falling back to PCA for same dimensionality.")
                X_vae, _, _ = perform_pca(X, latent_dim)

                # Use PCA results but still calculate some VAE metrics
                kl_divergence = 0.0
                reconstruction_error = 0.0

                metrics = {
                    'kl_divergence': kl_divergence,
                    'reconstruction_error': reconstruction_error,
                    'final_loss': 0.0,
                    'fallback_to_pca': True
                }

                print("Used PCA fallback due to VAE numerical instability")
                return X_vae, None, metrics

        # Calculate KL divergence with safeguards
        try:
            kl_divergence = -0.5 * np.mean(
                1 + np.clip(z_log_var, -20, 20) - np.square(z_mean) - np.exp(np.clip(z_log_var, -20, 20)))
        except Exception as e:
            print(f"Error calculating KL divergence: {str(e)}")
            kl_divergence = 0.0

        # Calculate reconstruction error
        try:
            X_reconstructed = decoder.predict(X_vae)
            reconstruction_error = np.mean(np.square(X - X_reconstructed))
        except Exception as e:
            print(f"Error calculating reconstruction error: {str(e)}")
            reconstruction_error = 0.0

        metrics = {
            'kl_divergence': float(kl_divergence),
            'reconstruction_error': float(reconstruction_error),
            'final_loss': float(history.history['loss'][-1]) if history.history['loss'] else 0.0
        }

        print(f"VAE (latent_dim={latent_dim}, intermediate_dim={intermediate_dim}):")
        print(f"  Training completed ({len(history.history['loss'])} epochs)")
        print(f"  KL divergence: {kl_divergence:.4f}")
        print(f"  Reconstruction error: {reconstruction_error:.4f}")

        return X_vae, encoder, metrics

    except Exception as e:
        print(f"Error during VAE training: {str(e)}")
        print("Falling back to PCA for same dimensionality")

        # Fallback to PCA with same dimensionality
        X_vae, _, pca_metrics = perform_pca(X, latent_dim)

        metrics = {
            'kl_divergence': 0.0,
            'reconstruction_error': pca_metrics['reconstruction_error'],
            'final_loss': 0.0,
            'fallback_to_pca': True
        }

        return X_vae, None, metrics


def calculate_kl_divergence(z_mean, z_log_var):
    """
    Calculate the KL divergence for a VAE model with safety checks

    Parameters:
    z_mean: Mean values of the latent space
    z_log_var: Log variance values of the latent space

    Returns:
    kl_loss: KL divergence loss (scalar value)
    """
    # Clip values to prevent numerical instabilities
    z_log_var_clipped = np.clip(z_log_var, -20, 20)

    # KL divergence between the learned distribution and a standard normal distribution
    kl_loss = -0.5 * np.mean(1 + z_log_var_clipped - np.square(z_mean) - np.exp(z_log_var_clipped))

    return float(kl_loss)  # Convert to Python float


def create_autoencoder(input_dim, encoding_dim, intermediate_dim=256, dropout_rate=0.2):
    """
    创建标准自编码器模型

    参数:
    input_dim: 输入特征维度
    encoding_dim: 编码空间维度
    intermediate_dim: 中间层维度
    dropout_rate: dropout比率用于正则化

    返回:
    autoencoder: 完整的自编码器模型
    encoder: 编码器部分
    decoder: 解码器部分
    """
    from tensorflow.keras import layers, Model

    # 编码器
    input_layer = layers.Input(shape=(input_dim,))

    # 添加批归一化和dropout提高稳定性
    x = layers.BatchNormalization()(input_layer)
    x = layers.Dense(intermediate_dim, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # 添加另一层用于复杂关系
    x = layers.Dense(intermediate_dim // 2, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # 编码层
    encoded = layers.Dense(encoding_dim, activation='relu')(x)

    # 定义编码器模型
    encoder = Model(input_layer, encoded, name='encoder')

    # 解码器
    decoder_input = layers.Input(shape=(encoding_dim,))

    # 镜像编码器架构
    x = layers.Dense(intermediate_dim // 2, activation='relu')(decoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(intermediate_dim, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # 输出层
    decoded = layers.Dense(input_dim, activation='linear')(x)

    # 定义解码器模型
    decoder = Model(decoder_input, decoded, name='decoder')

    # 定义自编码器模型(编码器+解码器)
    autoencoder_output = decoder(encoder(input_layer))
    autoencoder = Model(input_layer, autoencoder_output, name='autoencoder')

    # 编译模型
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder, decoder


def perform_autoencoder(X, encoding_dim, intermediate_dim=256, epochs=50, batch_size=32):
    """
    执行自编码器降维

    参数:
    X: 输入数据矩阵
    encoding_dim: 编码空间维度
    intermediate_dim: 中间层维度
    epochs: 训练周期数
    batch_size: 批量大小

    返回:
    X_encoded: 降维后的数据
    encoder: 编码器模型
    metrics: 模型特定指标
    """
    from tensorflow.keras import callbacks
    import time

    # 创建自编码器模型
    autoencoder, encoder, decoder = create_autoencoder(X.shape[1], encoding_dim, intermediate_dim)

    # 添加早停和学习率减少的回调函数
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]

    # 记录开始时间
    start_time = time.time()

    # 训练自编码器
    history = autoencoder.fit(
        X, X,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks_list,
        verbose=1
    )

    # 计算训练时间
    training_time = time.time() - start_time

    # 获取编码的数据
    X_encoded = encoder.predict(X)

    # 检查是否有NaN值
    if np.isnan(X_encoded).any():
        print("警告: 自编码器输出中检测到NaN值。尝试使用更保守的参数...")

        # 尝试使用更保守的参数
        autoencoder, encoder, decoder = create_autoencoder(
            X.shape[1], encoding_dim, intermediate_dim, dropout_rate=0.1
        )
        # 使用更慢的学习率
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

        # 减少周期和批量大小
        history = autoencoder.fit(X, X, epochs=30, batch_size=max(8, batch_size // 2), verbose=1)
        X_encoded = encoder.predict(X)

        # 如果仍然有NaN值，回退到PCA
        if np.isnan(X_encoded).any():
            print("仍然检测到NaN值。回退到PCA进行相同维度的降维。")
            X_encoded, _, pca_metrics = perform_pca(X, encoding_dim)

            metrics = {
                'reconstruction_error': pca_metrics['reconstruction_error'],
                'training_time': training_time,
                'final_loss': 0.0,
                'fallback_to_pca': True
            }

            return X_encoded, None, metrics

    # 计算重构误差
    X_reconstructed = decoder.predict(X_encoded)
    reconstruction_error = np.mean(np.square(X - X_reconstructed))

    metrics = {
        'reconstruction_error': reconstruction_error,
        'training_time': training_time,
        'final_loss': history.history['loss'][-1] if history.history['loss'] else None,
        'val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history and history.history[
            'val_loss'] else None
    }

    print(f"自编码器 (encoding_dim={encoding_dim}, intermediate_dim={intermediate_dim}):")
    print(f"  训练完成 ({len(history.history['loss'])} 个周期)")
    print(f"  训练时间: {training_time:.2f} 秒")
    print(f"  重构误差: {reconstruction_error:.4f}")

    return X_encoded, encoder, metrics