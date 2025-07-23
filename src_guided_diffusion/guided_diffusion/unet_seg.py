import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .fp16_util import convert_module_to_f16, convert_module_to_f32



class UNetModelSeg(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
       
        #need input layer 
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.down1 = self.downsample(64, 128)
        self.down2 = self.downsample(128, 256)
        self.down3 = self.downsample(256, 512)
        self.down4 = self.downsample(512, 1024)
        
        # Bottleneck block
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Upsample blocks
        self.up4 = self.upsample(1024, 512)
        self.up3 = self.upsample(512, 256)
        self.up2 = self.upsample(256, 128)
        self.up1 = self.upsample(128, 64)
        
        # Final output block
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoding
        x1 = self.encoder(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Bottleneck
        x_bottleneck = self.bottleneck(x5)
        
        # Decoding
        x_up4 = self.up4(x_bottleneck)
        x_up3 = self.up3(x_up4)
        x_up2 = self.up2(x_up3)
        x_up1 = self.up1(x_up2)
        
        # Final output
        out = self.out_conv(x_up1)
        
        return out

    def convert_to_fp16(self):
        """
        Convert the encoder, bottleneck, and decoder to float16.
        """
        self.encoder.apply(convert_module_to_f16)
        self.bottleneck.apply(convert_module_to_f16)
        self.up4.apply(convert_module_to_f16)
        self.up3.apply(convert_module_to_f16)
        self.up2.apply(convert_module_to_f16)
        self.up1.apply(convert_module_to_f16)
        self.out_conv.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the encoder, bottleneck, and decoder to float32.
        """
        self.encoder.apply(convert_module_to_f32)
        self.bottleneck.apply(convert_module_to_f32)
        self.up4.apply(convert_module_to_f32)
        self.up3.apply(convert_module_to_f32)
        self.up2.apply(convert_module_to_f32)
        self.up1.apply(convert_module_to_f32)
        self.out_conv.apply(convert_module_to_f32)

    def convert_module_to_f16(module):
        if isinstance(module, nn.Module):
            module.half()

    def convert_module_to_f32(module):
        if isinstance(module, nn.Module):
            module.float()