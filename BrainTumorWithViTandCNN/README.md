# Brain Tumor Classification

This repository contains code for classifying brain tumor images using **ViT (Vision Transformer)** and a simple **CNN**. The dataset consists of a small number of labeled images, with each class stored in separate folders.

## Models

- **CNN**  
  - Achieved **86% accuracy** on the validation set.  
  - Uses two convolutional blocks and a fully connected classifier.  

- **ViT (Vision Transformer)**  
  - Achieved **76% accuracy** on the same dataset.  
  - Small ViT with 4 transformer blocks, 4 attention heads, and an embedding dimension of 8.  

## Dataset

- Brain tumor images (very limited dataset, ~200 images).  
- Two classes: Tumor vs No Tumor.  
- Images resized to 128x128.  
- Training data augmented with flips, rotations, and resized crops.

## Usage

1. Mount your dataset (example is from Google Drive).  
2. Adjust hyperparameters in the notebook as needed.  
3. Run training loops for CNN or ViT.  
4. Visualize predictions on test images.  

## Notes

- Overfitting is expected due to the very small dataset.  
- Consider adding more data or stronger augmentation for better generalization.

## Observations
- CNN performs better than ViT on this dataset due to the limited number of images.
