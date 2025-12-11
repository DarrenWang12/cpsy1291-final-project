"""
Data loading and preprocessing utilities for the CNN classifier.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import hyperparameters as hp


def load_dataset(data_dir, subset='training', validation_split=0.2):
    """
    Load and preprocess the dataset from directory structure.
    Args:
        data_dir: Path to the data directory, should contain 'train' and 'test' subdirectories
        subset: 'training' or 'validation' or 'testing'
        validation_split: Fraction of training data to use for validation
    Returns:
        A tf.data.Dataset object
    """
    if subset == 'testing':
        data_path = Path(data_dir) / 'test'
    else:
        data_path = Path(data_dir) / 'train'
    # Create dataset from directory structure
    # FAKE = 0, REAL = 1
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        labels='inferred',
        label_mode='binary', 
        class_names=['FAKE', 'REAL'],
        color_mode='rgb',
        batch_size=hp.batch_size,
        image_size=(hp.img_height, hp.img_width),
        shuffle=(subset != 'testing'),
        seed=42,
        validation_split=validation_split if subset != 'testing' else None,
        subset=subset if subset != 'testing' else None
    )
    # Normalize pixel values to [0, 1]
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    # Optimize dataset performance
    # For large training datasets, don't cache in RAM to avoid memory issues
    # Cache only for smaller validation/test sets to speed them up
    if subset == 'training':
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    else:
        # Validation and test sets are smaller, so caching is safe and beneficial
        dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)
    return dataset


def get_data_augmentation():
    """
    Create data augmentation layer for training.
    Returns:
        A Sequential model with augmentation layers
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(hp.rotation_range / 360.0),
        tf.keras.layers.RandomTranslation(
            hp.height_shift_range, 
            hp.width_shift_range
        ),
        tf.keras.layers.RandomFlip('horizontal' if hp.horizontal_flip else None),
        tf.keras.layers.RandomZoom(hp.zoom_range),
    ])

def preprocess_image(image_path, target_size=(hp.img_height, hp.img_width)):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
        target_size: Target size (height, width)
    
    Returns:
        Preprocessed image tensor
    """
    img = tf.keras.utils.load_img(image_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = img_array / 255.0  # Normalize
    return img_array
