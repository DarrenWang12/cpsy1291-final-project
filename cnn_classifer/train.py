"""
Training script for the CNN classifier.
"""

import tensorflow as tf
import os
from pathlib import Path
import hyperparameters as hp
from model import VGGModel
from data_loader import load_dataset, get_data_augmentation


def train_model():
    """
    Train the VGGModel on the dataset.
    """
    print("Loading datasets...")
    
    # Load training and validation datasets
    train_dataset = load_dataset(hp.data_dir, subset='training', validation_split=0.2)
    val_dataset = load_dataset(hp.data_dir, subset='validation', validation_split=0.2)
    
    # Get data augmentation
    data_augmentation = get_data_augmentation()
    
    # Create model
    print("Creating model...")
    model = VGGModel()
    
    # Add data augmentation as first layer (wrapped in a new model)
    inputs = tf.keras.Input(shape=(hp.img_height, hp.img_width, 3))
    x = data_augmentation(inputs)
    x = model(x, training=True)
    trainable_model = tf.keras.Model(inputs, x)
    
    # Compile the model
    trainable_model.compile(
        optimizer=model.optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', 'binary_accuracy']
    )
    
    # Create directories for saving
    os.makedirs(hp.model_save_path, exist_ok=True)
    os.makedirs(hp.checkpoint_dir, exist_ok=True)
    
    # Calculate steps per epoch (needed for checkpoint frequency)
    train_size = len(list(Path(hp.train_dir).glob('**/*.jpg')))
    val_size = len(list(Path(hp.test_dir).glob('**/*.jpg')))
    steps_per_epoch = train_size // hp.batch_size
    validation_steps = val_size // hp.batch_size
    
    # Callbacks
    callbacks = [
        # Save best model based on validation accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(hp.checkpoint_dir, 'best_model.weights.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1,
            save_freq='epoch'
        ),
        # Save checkpoint every 5 epochs (as full model for backup)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(hp.checkpoint_dir, 'checkpoint_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.h5'),
            monitor='val_accuracy',
            save_best_only=False,
            save_weights_only=False,
            save_freq=5 * steps_per_epoch,  # Save every 5 epochs
            verbose=1
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Learning rate reduction
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        # CSV logger for easy analysis
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(hp.checkpoint_dir, 'training_log.csv'),
            append=False,
            separator=','
        ),
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    # Print model summary
    trainable_model.summary()
    
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Training images:      {train_size:,}")
    print(f"Validation images:    {val_size:,}")
    print(f"Batch size:           {hp.batch_size}")
    print(f"Steps per epoch:      {steps_per_epoch:,}")
    print(f"Validation steps:     {validation_steps:,}")
    print(f"Epochs:               {hp.num_epochs}")
    print(f"Learning rate:        {hp.learning_rate}")
    print(f"Image size:           {hp.img_height}x{hp.img_width}")
    print(f"Checkpoint directory: {hp.checkpoint_dir}")
    print(f"Model save path:      {hp.model_save_path}")
    print("="*70)
    print("\nStarting training...\n")
    history = trainable_model.fit(
        train_dataset,
        epochs=hp.num_epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Print training summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    if len(history.history['accuracy']) > 0:
        print(f"Final training accuracy:   {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"Best validation accuracy:  {max(history.history['val_accuracy']):.4f}")
        print(f"Final training loss:       {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss:     {history.history['val_loss'][-1]:.4f}")
    print("="*70)
    
    # Save final model
    print("\nSaving models...")
    final_model_path = os.path.join(hp.model_save_path, 'final_model')
    trainable_model.save(final_model_path)
    print(f"  ✓ Final model saved to: {final_model_path}")
    
    # Also save the base model without augmentation for inference
    base_model_path = os.path.join(hp.model_save_path, 'base_model')
    model.save(base_model_path)
    print(f"  ✓ Base model saved to:  {base_model_path}")
    
    # Print checkpoint info
    print(f"\nCheckpoints available at: {hp.checkpoint_dir}")
    print(f"Training log CSV: {os.path.join(hp.checkpoint_dir, 'training_log.csv')}")
    print(f"TensorBoard logs: ./logs")
    print("\n" + "="*70 + "\n")
    
    return trainable_model, history


if __name__ == "__main__":
    # Set memory growth for GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    train_model()

