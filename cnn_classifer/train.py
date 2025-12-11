"""
Training script for the CNN classifier.
"""

import tensorflow as tf
import os
import re
import glob
import csv
from pathlib import Path
import hyperparameters as hp
from model import VGGModel
from data_loader import load_dataset, get_data_augmentation


def find_best_checkpoint(checkpoint_dir):
    """
    Find the checkpoint with the highest validation accuracy.
    
    Returns:
        Tuple of (checkpoint_path, epoch, val_accuracy) or (None, 0, 0.0) if no checkpoint found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None, 0, 0.0
    
    best_checkpoint = None
    best_acc = 0.0
    best_epoch = 0
    
    # Check periodic checkpoints (format: checkpoint_epoch_XX_val_acc_X.XXXX.h5)
    pattern = checkpoint_dir / 'checkpoint_epoch_*_val_acc_*.h5'
    for checkpoint_path in glob.glob(str(pattern)):
        # Extract epoch and accuracy from filename
        match = re.search(r'epoch_(\d+)_val_acc_([\d.]+)', checkpoint_path)
        if match:
            epoch = int(match.group(1))
            acc = float(match.group(2))
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_checkpoint = checkpoint_path
    
    # Also check the best_model.weights.h5 - we'll need to get accuracy from CSV log
    best_weights_path = checkpoint_dir / 'best_model.weights.h5'
    if best_weights_path.exists():
        # Try to get accuracy from CSV log
        csv_path = checkpoint_dir / 'training_log.csv'
        if csv_path.exists():
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        # Find row with highest val_accuracy
                        best_row = max(rows, key=lambda x: float(x.get('val_accuracy', 0)))
                        csv_acc = float(best_row.get('val_accuracy', 0))
                        csv_epoch = int(best_row.get('epoch', 0)) + 1  # epoch is 0-indexed in CSV
                        
                        if csv_acc > best_acc:
                            best_acc = csv_acc
                            best_epoch = csv_epoch
                            best_checkpoint = str(best_weights_path)
            except (IOError, ValueError, KeyError) as e:
                print(f"Warning: Could not read training log: {e}")
    
    return best_checkpoint, best_epoch, best_acc


def get_starting_epoch(checkpoint_dir):
    """
    Determine the starting epoch from the training log CSV.
    
    Returns:
        Starting epoch number (0 if no previous training found)
    """
    csv_path = Path(checkpoint_dir) / 'training_log.csv'
    if csv_path.exists():
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    # Get the last epoch number and add 1
                    last_epoch = int(rows[-1].get('epoch', 0))
                    return last_epoch + 1
        except (IOError, ValueError, KeyError) as e:
            print(f"Warning: Could not determine starting epoch from CSV: {e}")
    
    return 0


def train_model(resume=True, initial_epoch=None):
    """
    Train the VGGModel on the dataset.
    
    Args:
        resume: If True, attempt to resume from the best checkpoint
        initial_epoch: Starting epoch (if None, will be determined from checkpoint/CSV)
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
    
    # Check for existing checkpoints and resume if requested
    start_epoch = 0
    if resume:
        best_checkpoint, best_epoch, best_acc = find_best_checkpoint(hp.checkpoint_dir)
        
        if best_checkpoint and os.path.exists(best_checkpoint):
            print(f"\n{'='*70}")
            print("RESUMING FROM CHECKPOINT")
            print(f"{'='*70}")
            print(f"Best checkpoint found: {best_checkpoint}")
            print(f"Epoch: {best_epoch}, Validation Accuracy: {best_acc:.4f}")
            print(f"{'='*70}\n")
            
            # Load the checkpoint
            if best_checkpoint.endswith('.weights.h5'):
                # Load weights only
                trainable_model.load_weights(best_checkpoint)
                print(f"Loaded weights from {best_checkpoint}")
            else:
                # Load full model
                trainable_model = tf.keras.models.load_model(best_checkpoint)
                print(f"Loaded full model from {best_checkpoint}")
            
            # Determine starting epoch
            if initial_epoch is None:
                start_epoch = get_starting_epoch(hp.checkpoint_dir)
                if start_epoch == 0:
                    start_epoch = best_epoch
            else:
                start_epoch = initial_epoch
            
            print(f"Resuming training from epoch {start_epoch}\n")
        else:
            print("No checkpoint found. Starting training from scratch.\n")
    
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
        # Save best model based on validation accuracy (saves every epoch when accuracy improves)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(hp.checkpoint_dir, 'best_model.weights.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1,
            save_freq='epoch'
        ),
        # Save epoch-specific checkpoint every epoch when validation accuracy improves
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(hp.checkpoint_dir, 'checkpoint_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.h5'),
            monitor='val_accuracy',
            save_best_only=True,  # Only save when accuracy improves
            save_weights_only=False,
            save_freq='epoch',
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
        # CSV logger for easy analysis (append=True to preserve history when resuming)
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(hp.checkpoint_dir, 'training_log.csv'),
            append=resume and start_epoch > 0,  # Append if resuming, otherwise overwrite
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
    print(f"Starting epoch:       {start_epoch}")
    print(f"Learning rate:        {hp.learning_rate}")
    print(f"Image size:           {hp.img_height}x{hp.img_width}")
    print(f"Checkpoint directory: {hp.checkpoint_dir}")
    print(f"Model save path:      {hp.model_save_path}")
    print("="*70)
    print("\nStarting training...\n")
    
    # Adjust total epochs if resuming
    total_epochs = hp.num_epochs
    if start_epoch >= total_epochs:
        print(f"Warning: Starting epoch {start_epoch} is >= total epochs {total_epochs}.")
        print(f"Training will complete immediately. Adjust num_epochs in hyperparameters.py if needed.\n")
    
    history = trainable_model.fit(
        train_dataset,
        epochs=total_epochs,
        initial_epoch=start_epoch,
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

