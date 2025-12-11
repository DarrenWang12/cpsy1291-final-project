"""
Hyperparameters for the CNN model training.
"""

# Learning rate for the optimizer
learning_rate = 0.001

# Batch size for training
# Reduced to 16 for Colab memory constraints (can lower to 8 if still having issues)
batch_size = 16

# Number of epochs to train
num_epochs = 20

# Image dimensions (height, width)
img_height = 224
img_width = 224

# Data augmentation parameters
rotation_range = 20
width_shift_range = 0.1
height_shift_range = 0.1
horizontal_flip = True
zoom_range = 0.1

# Paths
data_dir = "../data"
train_dir = "../data/train"
test_dir = "../data/test"
model_save_path = "./models"
checkpoint_dir = "./checkpoints"

