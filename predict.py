#!/usr/bin/env python3
"""
predict.py

This script loads a pre-trained model to predict classes for images in a specified directory.
Usage:
    python predict.py <path_to_images>
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import shutil
import warnings
warnings.filterwarnings("ignore")

seed = 99
tf.random.set_seed(42)


def load_model(model_path):
    """
    Load the trained Keras model from the specified path.

    Args:
        model_path (str): Path to the saved Keras model.

    Returns:
        keras.Model: Loaded Keras model.
    """
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please ensure the model exists.")
        sys.exit(1)
    model = tf.keras.models.load_model(model_path)
    return model


def preprocess_images(image_dir, img_size, batch_size):
    """
    Create a data generator for preprocessing images.

    Args:
        image_dir (str): Directory containing images to predict.
        img_size (int): Target image size for resizing.
        batch_size (int): Number of images per batch.

    Returns:
        generator: Keras generator yielding preprocessed images.
        file_names (list): List of image file names.
    """
    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        image_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='input',  # No labels provided
        shuffle=False  # Keep order for matching predictions
    )

    return generator, generator.filenames


def save_predictions(image_dir, file_names, predictions, output_filename='predictions.csv'):
    """
    Save the predictions to a CSV file in the specified directory.

    Args:
        image_dir (str): Directory where the CSV file will be saved.
        file_names (list): List of image file names.
        predictions (list): List of prediction labels.
        output_filename (str): Name of the output CSV file.
    """
    results = pd.DataFrame({
        'filename': [os.path.basename(f) for f in file_names],
        'prediction': predictions
    })

    output_path = os.path.join(image_dir, output_filename)
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main():
    # Check if the image directory path is provided
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_images>")
        sys.exit(1)

    image_dir = sys.argv[1]

    # Verify that the provided path exists and is a directory
    if not os.path.isdir(image_dir):
        print(f"The provided path '{image_dir}' is not a valid directory.")
        sys.exit(1)

    # Define model path
    model_path = os.path.join('models', 'model.keras')

    # Load the trained model
    model = load_model(model_path)

    # Define image parameters
    img_size = 100
    batch_size = 32

    # Create a temporary subdirectory structure as ImageDataGenerator expects subfolders
    # All images will be placed in a single 'unknown' subfolder
    temp_dir = os.path.join(image_dir, 'unknown')
    os.makedirs(temp_dir, exist_ok=True)

    # Move all images to the 'unknown' subfolder
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, os.path.join(temp_dir, filename))

    # Preprocess images
    generator, file_names = preprocess_images(image_dir, img_size, batch_size)

    # Perform predictions
    predictions = model.predict(generator, verbose=0)
    # Convert probabilities to binary classes with threshold 0.8
    predicted_classes = (predictions > 0.8).astype(int).flatten()

    # Map predicted classes to labels
    # class_labels = {v: k for k, v in generator.class_indices.items()}
    class_labels = {0: 'Parasitized', 1: 'Uninfected'}
    predicted_labels = [class_labels[int(pred)] for pred in predicted_classes]

    # Save predictions
    save_predictions(image_dir, file_names, predicted_labels)

    # Clean up copied images in temp_dir
    shutil.rmtree(temp_dir)

    print('Done!')
    print('Predictions generated in {}.'.format(os.path.join(image_dir, 'predictions.csv')))


if __name__ == "__main__":
    main()
