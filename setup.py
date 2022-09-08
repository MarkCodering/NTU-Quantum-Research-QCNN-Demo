import tensorflow as tf
import torch

print("Num GPUs Available for TensorFlow: ", len(tf.config.list_physical_devices('GPU')))
print("Num GPUs Available for PyTorch: ", torch.cuda.device_count())
#print("GPU Device Name for PyTorch: ", torch.cuda.get_device_name(0))