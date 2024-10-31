# ASG-TestDataScience-1

## Project Overview

This repository contains a comprehensive project focused on the detection of malaria-infected cells using deep learning techniques. The project utilizes a convolutional neural network (CNN) architecture, specifically leveraging transfer learning with a pre-trained MobileNetV2 model.

## Repository Structure

### **data**

- **external**: New data to be processed in production.
- **processed**: Final datasets ready for modeling.
- **raw**: Original data dumps that remain unchanged.

### **models**

- Stores trained models.

### **notebooks**

#### `main.ipynb`

- This notebook is used for exploratory data analysis and contains code snippets to understand and visualize the dataset. It may also include steps for model training or evaluation.

### **scripts**

#### `predict.py`

- **Functionality**: This script loads a pre-trained Keras model to predict classes for images in a specified directory.
- **Key Functions**:
  - `load_model(model_path)`: Loads the trained Keras model from a given path.
  - `preprocess_images(image_dir, img_size, batch_size)`: Prepares images for prediction by resizing and normalizing them.
  - `save_predictions(image_dir, file_names, predictions)`: Saves the predicted labels to a CSV file.
  - `main()`: Orchestrates the prediction process including loading the model, preprocessing images, making predictions, and saving results.


## Setup Instructions

1. **Environment Setup**: Ensure you have all dependencies installed using `requirements.txt` with the command:
   ```bash
   pip install -r requirements.txt
   ```

2. **Running Predictions**:
   - Place your images in a directory.
   - Run the prediction script:
     ```bash
     python predict.py <path_to_image_directory>
     ```

3. **Model Training**: Use Jupyter notebooks in the `notebooks` directory to train models or perform further analysis.