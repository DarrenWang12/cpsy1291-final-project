# CNN Classifier for Real vs AI Image Classification

A deep learning project for classifying images as either real or AI-generated using a VGG16-based convolutional neural network.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

Train the model on the dataset:

```bash
cd cnn_classifer
python main.py train
```

Or use the training script directly:

```bash
python train.py
```

The trained model will be saved to:
- `./models/final_model/` - Full model with augmentation
- `./models/base_model/` - Base model for inference
- `./checkpoints/best_model.weights.h5` - Best weights during training

### Prediction

#### Single Image
```bash
python main.py predict --image path/to/image.jpg
```

#### Batch Prediction (Directory)
```bash
python main.py predict --dir path/to/images --output results.csv
```

#### Using the Predict Script Directly
```bash
# Single image
python predict.py --image path/to/image.jpg

# Batch
python predict.py --dir path/to/images --output results.csv
```

### Evaluation

Evaluate the trained model on the test set:

```bash
python main.py evaluate
```

Or:

```bash
python predict.py --evaluate
```

## Model Architecture

The project uses a **VGG16-based model** (`VGGModel`) that:
- Uses ImageNet-pretrained VGG16 as a frozen feature extractor
- Adds a custom binary classification head with:
  - Global Average Pooling
  - Dropout (0.5)
  - Dense layer with sigmoid activation (binary output)

## Hyperparameters

Key hyperparameters can be modified in `hyperparameters.py`:
- Learning rate: 0.001
- Batch size: 32
- Number of epochs: 20
- Image dimensions: 224x224
- Data augmentation parameters

## Features

- **Data Augmentation**: Random rotation, translation, flipping, and zooming
- **Callbacks**: Model checkpointing, early stopping, learning rate reduction, TensorBoard logging
- **GPU Support**: Automatically uses GPU if available
- **Flexible Inference**: Supports single image, batch, and test set evaluation

## Training Outputs

- Model checkpoints saved during training
- TensorBoard logs in `./logs/`
- Final trained models in `./models/`
