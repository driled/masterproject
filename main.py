# TensorFlow
import tensorflow as tf
print("TF GPU available:", tf.config.list_physical_devices('GPU'))

# PyTorch
import torch
print("Torch CUDA:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
