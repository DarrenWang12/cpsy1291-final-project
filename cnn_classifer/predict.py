"""
Prediction script for the CNN classifier.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import os
import sys
import hyperparameters as hp
from model import VGGModel
from data_loader import preprocess_image

# Try to import tqdm for progress bars, fallback to simple counter if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install 'tqdm' for better progress bars: pip install tqdm")


def load_trained_model(model_path=None):
    """
    Load a trained model for inference.
    
    Args:
        model_path: Path to the saved model. If None, uses default path.
    
    Returns:
        Loaded model
    """
    if model_path is None:
        # Try to load from default locations
        base_model_path = os.path.join(hp.model_save_path, 'base_model')
        final_model_path = os.path.join(hp.model_save_path, 'final_model')
        checkpoint_path = os.path.join(hp.checkpoint_dir, 'best_model.weights.h5')
        
        if os.path.exists(final_model_path):
            print(f"Loading model from {final_model_path}")
            return tf.keras.models.load_model(final_model_path)
        elif os.path.exists(base_model_path):
            print(f"Loading model from {base_model_path}")
            return tf.keras.models.load_model(base_model_path)
        elif os.path.exists(checkpoint_path):
            print(f"Loading model from checkpoint {checkpoint_path}")
            model = VGGModel()
            model.load_weights(checkpoint_path)
            return model
        else:
            raise FileNotFoundError(
                "No trained model found. Please train the model first using train.py"
            )
    else:
        if os.path.isdir(model_path):
            print(f"Loading model from {model_path}")
            return tf.keras.models.load_model(model_path)
        else:
            # Assume it's a weights file
            model = VGGModel()
            model.load_weights(model_path)
            return model


def predict_single_image(model, image_path, verbose=False):
    """
    Predict whether a single image is REAL or FAKE.
    
    Args:
        model: Trained model
        image_path: Path to the image file
        verbose: Whether to print progress
    
    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    img_array = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(img_array, verbose=1 if verbose else 0)[0][0]
    
    # Interpret results
    # Model outputs probability of being REAL (1)
    is_real = prediction > 0.5
    confidence = prediction if is_real else (1 - prediction)
    
    return {
        'image_path': str(image_path),
        'prediction': 'REAL' if is_real else 'FAKE',
        'confidence': float(confidence),
        'probability_real': float(prediction),
        'probability_fake': float(1 - prediction)
    }


def predict_batch(model, image_dir, output_file=None):
    """
    Predict on a batch of images in a directory.
    
    Args:
        model: Trained model
        image_dir: Directory containing images
        output_file: Optional file to save results (CSV format)
    
    Returns:
        List of prediction dictionaries
    """
    image_path = Path(image_dir)
    if not image_path.exists():
        raise ValueError(f"Directory {image_dir} does not exist")
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_path.rglob(f'*{ext}'))
        image_files.extend(image_path.rglob(f'*{ext.upper()}'))
    
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")
    
    print(f"\nFound {len(image_files)} images. Processing...")
    print("="*70)
    
    results = []
    errors = []
    
    # Use tqdm if available, otherwise use simple counter
    iterator = tqdm(image_files, desc="Predicting", unit="image") if HAS_TQDM else image_files
    
    for idx, img_file in enumerate(iterator, 1):
        try:
            result = predict_single_image(model, img_file)
            results.append(result)
            
            # Print progress every 10 images or if tqdm is not available
            if not HAS_TQDM and (idx % 10 == 0 or idx == len(image_files)):
                print(f"[{idx}/{len(image_files)}] {img_file.name}: {result['prediction']} "
                      f"(confidence: {result['confidence']:.3f})")
            elif not HAS_TQDM and idx <= 10:
                # Print first 10 for quick feedback
                print(f"[{idx}/{len(image_files)}] {img_file.name}: {result['prediction']} "
                      f"(confidence: {result['confidence']:.3f})")
        except Exception as e:
            error_msg = f"Error processing {img_file}: {e}"
            errors.append({'file': str(img_file), 'error': str(e)})
            if HAS_TQDM:
                tqdm.write(error_msg)
            else:
                print(error_msg)
    
    print("="*70)
    print(f"\nProcessing complete!")
    print(f"  Successfully processed: {len(results)} images")
    if errors:
        print(f"  Errors: {len(errors)} images")
    
    # Save results if output file specified
    if output_file:
        import csv
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {output_file}")
    
    return results


def evaluate_on_test_set(model_path=None):
    """
    Evaluate the model on the test set.
    
    Args:
        model_path: Path to the saved model. If None, uses default path.
    
    Returns:
        Evaluation metrics
    """
    from data_loader import load_dataset
    
    # Load model
    model = load_trained_model(model_path)
    
    # Load test dataset
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    print("Loading test dataset...")
    test_dataset = load_dataset(hp.data_dir, subset='testing')
    
    # Compile model with metrics
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', 'binary_accuracy', 'precision', 'recall']
    )
    
    # Evaluate
    print("Evaluating on test set...")
    print("-"*70)
    results = model.evaluate(test_dataset, verbose=1)
    print("-"*70)
    
    metrics = {
        'test_loss': results[0],
        'test_accuracy': results[1],
        'test_binary_accuracy': results[2] if len(results) > 2 else results[1]
    }
    
    # Add precision and recall if available
    if len(results) > 3:
        metrics['test_precision'] = results[3]
    if len(results) > 4:
        metrics['test_recall'] = results[4]
    
    print("\nTest Set Results:")
    print("-"*70)
    for key, value in metrics.items():
        print(f"  {key:25s}: {value:.4f}")
    print("="*70 + "\n")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict images using trained CNN model')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--dir', type=str, help='Path to directory of images')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate on test set')
    parser.add_argument('--output', type=str, help='Output CSV file for batch predictions')
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_on_test_set(args.model)
    elif args.image:
        model = load_trained_model(args.model)
        result = predict_single_image(model, args.image)
        print("\nPrediction Results:")
        print(f"  Image: {result['image_path']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Probability (REAL): {result['probability_real']:.3f}")
        print(f"  Probability (FAKE): {result['probability_fake']:.3f}")
    elif args.dir:
        model = load_trained_model(args.model)
        predict_batch(model, args.dir, args.output)
    else:
        parser.print_help()

