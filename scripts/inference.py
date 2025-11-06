"""
Inference script for colorizing black and white images
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.colorization_model import ColorizationModel
from utils.data_utils import lab_to_rgb


def colorize_image(model, image_path, device, output_path=None, size=(256, 256)):
    """
    Colorize a single grayscale image
    
    Args:
        model: Trained colorization model
        image_path: Path to input grayscale image
        device: Device to run inference on
        output_path: Path to save colorized image (optional)
        size: Target size for processing
    
    Returns:
        Colorized RGB image as numpy array
    """
    # Load image
    img = cv2.imread(image_path)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # Resize
    original_size = img_gray.shape[:2]
    img_gray_resized = cv2.resize(img_gray, size)
    
    # Normalize L channel
    l_channel = img_gray_resized.astype(np.float32) / 255.0
    
    # Convert to tensor
    l_tensor = torch.from_numpy(l_channel).unsqueeze(0).unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        predicted_ab = model(l_tensor)
        predicted_ab = torch.clamp(predicted_ab, 0, 1)
    
    # Convert back to RGB
    l_channel_tensor = l_tensor.squeeze(0)
    rgb_result = lab_to_rgb(l_channel_tensor, predicted_ab.squeeze(0))
    
    # Resize back to original size
    rgb_result = cv2.resize(rgb_result, (original_size[1], original_size[0]))
    
    # Save if output path is provided
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(rgb_result, cv2.COLOR_RGB2BGR))
        print(f"Colorized image saved to {output_path}")
    
    return rgb_result


def main():
    parser = argparse.ArgumentParser(description='Colorize black and white images')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input grayscale image or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save colorized image(s)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for processing')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = ColorizationModel(input_channels=1, output_channels=2)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        output_path = args.output or args.input.replace('.', '_colorized.')
        colorize_image(model, args.input, device, output_path, (args.image_size, args.image_size))
        
        # Display comparison
        img_gray = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        img_colorized = cv2.imread(output_path)
        img_colorized = cv2.cvtColor(img_colorized, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_gray, cmap='gray')
        axes[0].set_title('Input (Grayscale)')
        axes[0].axis('off')
        
        axes[1].imshow(img_colorized)
        axes[1].set_title('Colorized')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    elif os.path.isdir(args.input):
        # Directory of images
        output_dir = args.output or os.path.join(args.input, 'colorized')
        os.makedirs(output_dir, exist_ok=True)
        
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(args.input) 
                      if f.lower().endswith(extensions)]
        
        print(f"Found {len(image_files)} images to process...")
        
        for img_file in image_files:
            input_path = os.path.join(args.input, img_file)
            output_path = os.path.join(output_dir, f"colorized_{img_file}")
            colorize_image(model, input_path, device, output_path, 
                          (args.image_size, args.image_size))
        
        print(f"All images processed and saved to {output_dir}")
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == '__main__':
    main()

