import tensorflow as tf
from keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D

import hyperparameters as hp

from keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from keras.optimizers import Adam


class YourModel(tf.keras.Model):
    """ Your own neural network model for 15-way classification. """

    def __init__(self):
        super(YourModel, self).__init__()
        self.optimizer = Adam(learning_rate=hp.learning_rate)
        self.architecture = [
            Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'),
            MaxPool2D(pool_size=(2, 2), padding='same'),
            Dropout(rate=0.25),

            Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
            MaxPool2D(pool_size=(2, 2), padding='same'),
            Dropout(rate=0.25),

            Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
            MaxPool2D(pool_size=(2, 2), padding='same'),
            Dropout(rate=0.4),

            Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
            MaxPool2D(pool_size=(2, 2), padding='same'),
            Dropout(rate=0.4),

            Flatten(),
            Dense(units=512, activation='relu'),
            Dropout(rate=0.5),
            Dense(units=256, activation='relu'),
            Dropout(rate=0.5),
            Dense(units=15, activation='softmax')
        ]

    def call(self, x):
        """ Passes input image through the network. """
        for layer in self.architecture:
            x = layer(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model (multi-class). """
        lossfxn = SparseCategoricalCrossentropy()
        return lossfxn(labels, predictions)


class VGGModel(tf.keras.Model):
    """
    VGG16-based binary classifier (e.g., AI vs real images) using
    ImageNet-pretrained weights and a small custom head.
    """

    def __init__(self):
        super(VGGModel, self).__init__()

        self.optimizer = Adam(learning_rate=hp.learning_rate)

        # Pretrained VGG16 base (no top classifier, ImageNet weights).
        # No input_shape specified => works for 3-channel images with
        # spatial dims >= 32x32 when include_top=False.
        self.vgg16 = tf.keras.applications.VGG16(
            include_top=False,
            weights="imagenet"
        )
        # Freeze the convolutional base for feature extraction
        self.vgg16.trainable = False

        # Small classification head for binary output
        self.head = tf.keras.Sequential(
            [
                GlobalAveragePooling2D(name="gap"),
                Dropout(rate=0.5),
                Dense(units=1, activation='sigmoid', name="binary_output")
            ],
            name="vgg_head"
        )

    def call(self, x, training=False):
        """
        Passes the image through the pretrained VGG16 base
        and the custom binary classification head.
        """
        # VGG16 as a frozen feature extractor
        x = self.vgg16(x, training=False)
        # Head may still use dropout, so respect `training` flag there
        x = self.head(x, training=training)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Binary cross-entropy loss for AI/real classification. """
        lossfxn = BinaryCrossentropy()
        return lossfxn(labels, predictions)