"""Image backbones for GHIST+.

The primary path uses the UNI2-h foundation model through `Uni2HAdapter`.
`Backbone` is retained as the legacy convolutional fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import unetConv2
from .initialisation import init_weights

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import hf_hub_download


class Uni2HEncoder(nn.Module):
    """ViT-H/14 encoder (MahmoodLab/UNI2-h) with HF weights."""
    def __init__(self, pretrained: bool = True):
        """
        Instantiate the ViT-H/14 encoder used in MahmoodLab/UNI2-h.

        We disable timm's built-in weight loader (pretrained=False) and
        instead pull the checkpoint with `hf_hub_download` and load it
        manually.  This bypasses the positional-embedding resampling bug
        that crashes when reg_tokens > 0.
        """
        super().__init__()

        # Exact architecture parameters from the model card.
        self.vit = timm.create_model(
            "vit_giant_patch14_224",        # matching backbone name
            pretrained=False,               # we'll load weights ourselves
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            init_values=1e-5,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )

        # load MahmoodLab weights if requested
        if pretrained:
            ckpt_path = hf_hub_download(
                "MahmoodLab/UNI2-h",
                "pytorch_model.bin",
                cache_dir=None,    # default HF cache
                local_dir=None,    # let hf decide
                force_download=False,
            )
            state_dict = torch.load(ckpt_path, map_location="cpu")
            missing, unexpected = self.vit.load_state_dict(state_dict, strict=True)
            if missing or unexpected:
                raise RuntimeError(
                    f"Weight-loading mismatch: missing={missing}, unexpected={unexpected}"
                )

        self.out_channels = self.vit.embed_dim      # 1536
        self.data_cfg   = resolve_data_config({}, model=self.vit)
        self._transform = create_transform(**self.data_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, 3, 224, 224) RGB

        Returns
        -------
        feat : (B, C, H_t, W_t) tensor of patch features.
                For 224x224 inputs and patch_size=14, this gives 16x16 patches.
        """
        seq = self.vit.forward_features(x)          # (B, 1+R+P, C)
        if isinstance(seq, dict):                   # timm >=0.9 returns dict
            seq = seq["x_norm"]

        # timm orders tokens: [CLS] + [reg_tokens] + [patch_tokens]
        num_patches = self.vit.patch_embed.num_patches        # e.g. 256
        patch_tokens = seq[:, -num_patches:, :]               # keep last N
        B, N, C = patch_tokens.shape
        side = int(num_patches ** 0.5)                        # 16 for 256
        feat = patch_tokens.transpose(1, 2).reshape(B, C, side, side)
        return feat
    
class Uni2HAdapter(nn.Module):
    """Adapter exposing UNI2-h features through the legacy GHIST+ backbone API."""

    def __init__(self, pretrained: bool = True, n_seg_classes: int = 2):
        super().__init__()
        self.enc = Uni2HEncoder(pretrained=pretrained)   # ViT-H/14
        # Channel projections to match the legacy tensor shapes.
        self.proj_hd1 = nn.Conv2d(self.enc.out_channels, 320, 1)  # "hd1"
        self.proj_h1  = nn.Conv2d(self.enc.out_channels,  64, 1)  # "h1"
        self.seg_head = nn.Conv2d(320, n_seg_classes, 1)                    # dummy seg map

    def forward(self, x):
        # Make H and W divisible by 14 so the ViT patch embedding is valid.
        B, C, H, W = x.shape
        pad_h = (14 - H % 14) % 14
        pad_w = (14 - W % 14) % 14
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        feat = self.enc(x)                                         # (B,1536,h',w')
        # Use the padded spatial size as the target for up-sampling.
        target_H = H + pad_h
        target_W = W + pad_w

        hd1 = F.interpolate(self.proj_hd1(feat), size=(target_H, target_W),
                            mode="bilinear", align_corners=False)
        h1  = F.interpolate(self.proj_h1(feat),  size=(target_H, target_W),
                            mode="bilinear", align_corners=False)
        seg = F.interpolate(self.seg_head(hd1),  size=(target_H, target_W),
                            mode="bilinear", align_corners=False)

        # crop any padding so outputs align with the original nuclei mask
        if pad_h or pad_w:
            hd1 = hd1[:, :, :H, :W]
            h1  = h1[:, :, :H, :W]
            seg = seg[:, :, :H, :W]

        return seg, hd1, h1
    

class Backbone(nn.Module):
    def __init__(
        self,
        n_channels=3,
        bilinear=True,
        is_deconv=True,
        is_batchnorm=True,
        n_classes=8,
    ):
        super(Backbone, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.n_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        """stage 4d"""
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode="bilinear")  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        """stage 3d"""
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode="bilinear")  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode="bilinear")  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        """stage 2d """
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode="bilinear")  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode="bilinear")  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode="bilinear")  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        """stage 1d"""
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode="bilinear")  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode="bilinear")  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode="bilinear")  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode="bilinear")  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv_seg = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(
            self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1)))
        )
        h2_PT_hd4 = self.h2_PT_hd4_relu(
            self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2)))
        )
        h3_PT_hd4 = self.h3_PT_hd4_relu(
            self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3)))
        )
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(
            self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5)))
        )
        hd4 = self.relu4d_1(
            self.bn4d_1(
                self.conv4d_1(
                    torch.cat(
                        (h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1
                    )
                )
            )
        )  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(
            self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1)))
        )
        h2_PT_hd3 = self.h2_PT_hd3_relu(
            self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2)))
        )
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(
            self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4)))
        )
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(
            self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5)))
        )
        hd3 = self.relu3d_1(
            self.bn3d_1(
                self.conv3d_1(
                    torch.cat(
                        (h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1
                    )
                )
            )
        )  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(
            self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1)))
        )
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(
            self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))
        )
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(
            self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4)))
        )
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(
            self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5)))
        )
        hd2 = self.relu2d_1(
            self.bn2d_1(
                self.conv2d_1(
                    torch.cat(
                        (h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1
                    )
                )
            )
        )  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(
            self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))
        )
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(
            self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))
        )
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(
            self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4)))
        )
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(
            self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5)))
        )
        hd1 = self.relu1d_1(
            self.bn1d_1(
                self.conv1d_1(
                    torch.cat(
                        (h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1
                    )
                )
            )
        )  # hd1->320*320*UpChannels

        seg = self.outconv_seg(hd1)  # d1->320*320*2

        return seg, hd1, h1

__all__ = ["Backbone", "Uni2HEncoder", "Uni2HAdapter"]
