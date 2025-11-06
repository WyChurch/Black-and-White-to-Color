"""
Colorization Model using U-Net architecture
Converts grayscale images to color images in LAB color space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorizationModel(nn.Module):
    """
    U-Net based model for image colorization.
    Takes L channel (grayscale) as input and predicts AB channels (color).
    """
    
    def __init__(self, input_channels=1, output_channels=2):
        super(ColorizationModel, self).__init__()
        
        # Encoder (downsampling path)
        self.enc1 = self._conv_block(input_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        self.enc5 = self._conv_block(512, 512)
        
        # Decoder (upsampling path)
        self.dec1 = self._upconv_block(512, 512)
        self.dec2 = self._upconv_block(512 + 512, 256)
        self.dec3 = self._upconv_block(256 + 256, 128)
        self.dec4 = self._upconv_block(128 + 128, 64)
        self.dec5 = self._upconv_block(64 + 64, 64)
        
        # Final output layer
        self.final = nn.Conv2d(64, output_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def _conv_block(self, in_channels, out_channels):
        """Convolutional block with batch normalization and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_channels, out_channels):
        """Upsampling block with skip connections"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))
        
        # Decoder path with skip connections
        dec1 = self.dec1(self.pool(enc5))
        dec1 = torch.cat([dec1, enc5], dim=1)
        
        dec2 = self.dec2(dec1)
        dec2 = torch.cat([dec2, enc4], dim=1)
        
        dec3 = self.dec3(dec2)
        dec3 = torch.cat([dec3, enc3], dim=1)
        
        dec4 = self.dec4(dec3)
        dec4 = torch.cat([dec4, enc2], dim=1)
        
        dec5 = self.dec5(dec4)
        dec5 = torch.cat([dec5, enc1], dim=1)
        
        # Final output
        output = self.final(dec5)
        
        return output


if __name__ == "__main__":
    # Test the model
    model = ColorizationModel()
    x = torch.randn(1, 1, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

