# Black-and-White-to-Color

A deep learning project for automatically colorizing black and white photographs using computer vision techniques. This project uses a U-Net based convolutional neural network to predict color information in the LAB color space.

## Overview

This project implements an end-to-end solution for image colorization:
- **Model Architecture**: U-Net based CNN that takes grayscale (L channel) as input and predicts color (AB channels)
- **Color Space**: Uses LAB color space where L represents lightness and AB represent color information
- **Training**: Full training pipeline with validation, checkpointing, and visualization
- **Inference**: Easy-to-use script for colorizing single images or batches

## Project Structure

```
Black-and-White-to-Color/
├── models/
│   └── colorization_model.py    # U-Net model architecture
├── utils/
│   └── data_utils.py             # Data loading and preprocessing utilities
├── scripts/
│   ├── train.py                  # Training script
│   └── inference.py              # Inference script
├── data/
│   ├── raw/                      # Place your training images here
│   └── processed/                # Processed data (auto-generated)
├── outputs/                      # Model checkpoints and results
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model on your dataset:

```bash
python scripts/train.py --data_dir data/raw --epochs 50 --batch_size 16
```

**Training Arguments**:
- `--data_dir`: Directory containing training images (required)
- `--val_dir`: Directory containing validation images (optional)
- `--batch_size`: Batch size for training (default: 16)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--image_size`: Image size for processing (default: 256)
- `--save_dir`: Directory to save checkpoints (default: outputs)
- `--resume`: Path to checkpoint to resume training from (optional)

**Example**:
```bash
python scripts/train.py \
    --data_dir data/raw \
    --val_dir data/val \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --image_size 256
```

During training, the script will:
- Save model checkpoints every 10 epochs
- Save the best model based on validation loss
- Generate sample colorized images every 5 epochs
- Log metrics to TensorBoard (view with: `tensorboard --logdir outputs/logs`)

### Inference

To colorize black and white images:

**Single Image**:
```bash
python scripts/inference.py \
    --model_path outputs/checkpoints/best_model.pth \
    --input path/to/grayscale_image.jpg \
    --output path/to/colorized_output.jpg
```

**Directory of Images**:
```bash
python scripts/inference.py \
    --model_path outputs/checkpoints/best_model.pth \
    --input data/test_images/ \
    --output outputs/colorized_results/
```

**Inference Arguments**:
- `--model_path`: Path to trained model checkpoint (required)
- `--input`: Path to input image or directory (required)
- `--output`: Path to save output image(s) (optional)
- `--image_size`: Image size for processing (default: 256)

## How It Works

1. **Color Space Conversion**: Images are converted from RGB to LAB color space
   - **L channel**: Lightness (grayscale information)
   - **A channel**: Green-Red color information
   - **B channel**: Blue-Yellow color information

2. **Model Architecture**: The U-Net model:
   - Takes the L channel as input
   - Predicts the AB channels through an encoder-decoder architecture
   - Uses skip connections to preserve fine details

3. **Training**: The model learns to predict color by:
   - Comparing predicted AB channels with ground truth
   - Minimizing mean squared error loss
   - Learning color patterns from training data

4. **Inference**: During inference:
   - Input grayscale image is normalized
   - Model predicts AB channels
   - L and predicted AB are combined and converted back to RGB

## Dataset Preparation

1. Collect color images for training (the model will convert them to grayscale automatically)
2. Place training images in `data/raw/` directory
3. Optionally, create a validation set in a separate directory
4. Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

**Tips for better results**:
- Use diverse, high-quality color images
- Include various scenes (portraits, landscapes, objects, etc.)
- Ensure good lighting and contrast in training images
- More data generally leads to better results

## Model Architecture

The model uses a U-Net architecture with:
- **Encoder**: 5 convolutional blocks with max pooling (downsampling)
- **Decoder**: 5 upsampling blocks with skip connections
- **Skip Connections**: Preserve fine-grained details from encoder to decoder
- **Batch Normalization**: Stabilizes training and improves convergence

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)
- Sufficient RAM for your dataset size

## Troubleshooting

**Out of Memory Errors**:
- Reduce `batch_size` in training script
- Reduce `image_size` parameter
- Use fewer `num_workers` in data loading

**Poor Colorization Results**:
- Train for more epochs
- Use a larger, more diverse dataset
- Adjust learning rate
- Try different model architectures

**Slow Training**:
- Use GPU if available
- Increase `batch_size` if memory allows
- Reduce `image_size` for faster processing

## Future Improvements

- [ ] Add support for pre-trained models (transfer learning)
- [ ] Implement different loss functions (perceptual loss, adversarial loss)
- [ ] Add data augmentation for better generalization
- [ ] Support for higher resolution images
- [ ] Web interface for easy inference
- [ ] Real-time video colorization