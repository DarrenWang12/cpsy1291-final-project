"""
Main entry point for the CNN classifier project. Supports training, prediction, and evaluation.
"""

import argparse
import sys
from train import train_model
from predict import predict_single_image, predict_batch, evaluate_on_test_set, load_trained_model

def main():
    """
    Main function to handle command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='CNN Classifier for Real vs AI Image Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python main.py train

  # Predict on a single image
  python main.py predict --image path/to/image.jpg

  # Predict on a directory of images
  python main.py predict --dir path/to/images --output results.csv

  # Evaluate on test set
  python main.py evaluate
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.set_defaults(func=lambda args: train_model())
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict on image(s)')
    predict_parser.add_argument('--image', type=str, help='Path to single image file')
    predict_parser.add_argument('--dir', type=str, help='Path to directory of images')
    predict_parser.add_argument('--model', type=str, help='Path to trained model (optional)')
    predict_parser.add_argument('--output', type=str, help='Output CSV file for batch predictions')
    predict_parser.set_defaults(func=predict_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on test set')
    eval_parser.add_argument('--model', type=str, help='Path to trained model (optional)')
    eval_parser.set_defaults(func=lambda args: evaluate_on_test_set(args.model))
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


def predict_command(args):
    """
    Handle predict command.
    """
    if not args.image and not args.dir:
        print("Error: Either --image or --dir must be specified")
        return
    
    model = load_trained_model(args.model)
    
    if args.image:
        result = predict_single_image(model, args.image)
        print("\nPrediction Results:")
        print(f"  Image: {result['image_path']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Probability (REAL): {result['probability_real']:.3f}")
        print(f"  Probability (FAKE): {result['probability_fake']:.3f}")
    elif args.dir:
        predict_batch(model, args.dir, args.output)

if __name__ == "__main__":
    main()
