"""HRNetV2-W32 backbone: 4 parallel branches, strides 4/8/16/32, ALL returned.

Layout (input 480x480):

    stem   conv1 3x3 s2 -> conv2 3x3 s2        [B, 64, 120, 120]   stride 4
    layer1 4 x Bottleneck (64 -> 256)          [B, 256, 120, 120]  stride 4
    transition1: split into 2 branches         [32 @ s4, 64 @ s8]
    stage2 1 x HighResolutionModule(2 branches)
    transition2: grow a 3rd branch             [+128 @ s16]
    stage3 4 x HighResolutionModule(3 branches)
    transition3: grow a 4th branch             [+256 @ s32]
    stage4 3 x HighResolutionModule(4 branches)
    output: [c(32,s4), c(64,s8), c(128,s16), c(256,s32)]

Contrast with the earlier backbones in this repo: ResNet/DeepLab compute
resolutions IN SERIES (each stage replaces the previous resolution); HRNet
computes them IN PARALLEL and lets them exchange information after every
module, so the stride-4 output is deep-supervised semantics, not just early
edges. The head decides how to combine the 4 outputs (HRNetV2 concat / OCR).

Pretrained weights: torchvision has no HRNet, so we replicate the OFFICIAL
module naming exactly (conv1/bn1/conv2/bn2, layer1, transition1..3,
stage2..4.<m>.branches/fuse_layers) and load the official ImageNet checkpoint
hrnetv2_w32_imagenet_pretrained.pth directly -- see load_pretrained(). The
checkpoint's classification-only extras (incre_modules, classifier, ...) are
skipped via key filtering.
"""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from .branch import BasicBlock, Bottleneck, HighResolutionModule
except ImportError:
    from branch import BasicBlock, Bottleneck, HighResolutionModule


def _conv_bn_relu_3x3(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    """3x3 conv + BN + ReLU (the transition-layer building block)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class HRNetBackbone(nn.Module):
    """HRNetV2-W32 (hand-built, official-checkpoint-compatible naming).

    Args:
        channels: branch widths at strides 4/8/16/32 -- (32, 64, 128, 256)
            for W32. ("W48" would be (48, 96, 192, 384) etc.)
        num_blocks: BasicBlocks per branch in every module (official: 4).
        num_modules: modules per stage for (stage2, stage3, stage4) --
            official HRNetV2: (1, 4, 3).
        pretrained: path to hrnetv2_w32_imagenet_pretrained.pth, or None /
            missing file -> random init (a loud warning is printed).

    Attributes:
        out_channels: tuple of the 4 output widths (the head reads this).
        strides: (4, 8, 16, 32).
    """

    strides = (4, 8, 16, 32)

    def __init__(
        self,
        channels: Sequence[int] = (32, 64, 128, 256),
        num_blocks: int = 4,
        num_modules: Sequence[int] = (1, 4, 3),
        pretrained: str | None = None,
    ) -> None:
        super().__init__()
        self.out_channels: Tuple[int, ...] = tuple(channels)
        c = list(channels)

        # ---- Stem: two 3x3 stride-2 convs -> stride 4, 64 channels ----------
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ---- Stage 1: 4 Bottlenecks, 64 -> 256 channels (still stride 4) ----
        # First block needs a 1x1 projection shortcut for the channel change.
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, 1, bias=False), nn.BatchNorm2d(256))
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # ---- Transitions: create the parallel branches ----------------------
        # transition1 SPLITS the single 256-ch map into the first two branches
        # (one 3x3 per branch; the stride-8 one downsamples). Later transitions
        # only ADD one new, lower branch from the previous LAST branch; the
        # existing branches pass through unchanged (None).
        # (The inner extra Sequential on downsampling entries mirrors the
        # official checkpoint's key nesting: transition1.1.0.0.weight.)
        self.transition1 = nn.ModuleList([
            _conv_bn_relu_3x3(256, c[0], stride=1),
            nn.Sequential(_conv_bn_relu_3x3(256, c[1], stride=2)),
        ])
        self.transition2 = nn.ModuleList([
            None, None,
            nn.Sequential(_conv_bn_relu_3x3(c[1], c[2], stride=2)),
        ])
        self.transition3 = nn.ModuleList([
            None, None, None,
            nn.Sequential(_conv_bn_relu_3x3(c[2], c[3], stride=2)),
        ])

        # ---- Stages 2-4: repeated HighResolutionModules ---------------------
        # nn.Sequential happily chains modules whose input/output is a LIST of
        # tensors, and gives the official key prefixes stage2.0., stage3.1., ...
        m2, m3, m4 = num_modules
        self.stage2 = nn.Sequential(*[
            HighResolutionModule(2, c[:2], num_blocks) for _ in range(m2)])
        self.stage3 = nn.Sequential(*[
            HighResolutionModule(3, c[:3], num_blocks) for _ in range(m3)])
        self.stage4 = nn.Sequential(*[
            HighResolutionModule(4, c[:4], num_blocks) for _ in range(m4)])

        self.pretrained_loaded = False
        if pretrained:
            self.load_pretrained(pretrained)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run the backbone.

        Input:
            x: image batch [B, 3, H, W] (H, W multiples of 32).

        Output:
            list of 4 maps: [B,32,H/4,W/4], [B,64,H/8,W/8],
            [B,128,H/16,W/16], [B,256,H/32,W/32].
        """
        x = self.relu(self.bn1(self.conv1(x)))   # stride 2
        x = self.relu(self.bn2(self.conv2(x)))   # stride 4
        x = self.layer1(x)                       # [B, 256, H/4, W/4]

        # Split into two branches, then alternate stage / grow-a-branch.
        feats = [self.transition1[0](x), self.transition1[1](x)]
        feats = self.stage2(feats)

        feats = feats + [self.transition2[2](feats[-1])]
        feats = self.stage3(feats)

        feats = feats + [self.transition3[3](feats[-1])]
        feats = self.stage4(feats)
        return feats

    # ---- Pretrained loading -------------------------------------------------
    def load_pretrained(self, path: str) -> None:
        """Load the official HRNetV2-W32 ImageNet checkpoint (if present).

        The classification checkpoint shares this exact module naming for the
        trunk but adds classification-only heads (incre_modules,
        downsamp_modules, final_layer, classifier). Filtering to keys that
        exist here WITH matching shapes keeps the trunk and drops the rest.

        Missing file is NOT an error: the backbone stays randomly initialized
        and training runs from scratch (expect a large mIoU drop vs.
        pretrained -- ImageNet init was worth several points for every
        backbone in this repo).
        """
        if not os.path.isfile(path):
            print(f"[HRNet] WARNING: pretrained weights not found at\n"
                  f"        {path}\n"
                  f"        -> training FROM SCRATCH. Download "
                  f"hrnetv2_w32_imagenet_pretrained.pth (HRNet-Image-"
                  f"Classification releases) to enable ImageNet init.")
            return

        state = torch.load(path, map_location="cpu")
        # Some releases wrap the weights in a dict under "state_dict".
        if "state_dict" in state:
            state = state["state_dict"]

        own = self.state_dict()
        usable = {
            k: v for k, v in state.items()
            if k in own and own[k].shape == v.shape
        }
        missing = [k for k in own if k not in usable]
        self.load_state_dict(usable, strict=False)
        self.pretrained_loaded = True
        print(f"[HRNet] loaded pretrained backbone: {len(usable)} tensors "
              f"matched, {len(state) - len(usable)} checkpoint extras skipped, "
              f"{len(missing)} own tensors left at init "
              f"(expected 0 for the trunk)")

    # ---- Two-stage finetuning helpers ---------------------------------------
    # LOW  = stem + layer1 (generic early features, purely pretrained).
    # HIGH = transitions + stage2/3/4 (the parallel-branch machinery; its
    #        fusion convs must adapt to segmentation from the start).
    def _low_level_modules(self) -> List[nn.Module]:
        return [self.conv1, self.bn1, self.conv2, self.bn2, self.layer1]

    def _high_level_modules(self) -> List[nn.Module]:
        return [self.transition1, self.transition2, self.transition3,
                self.stage2, self.stage3, self.stage4]

    def freeze_low_layers(self) -> None:
        """Freeze stem + layer1 (weights AND BatchNorm statistics)."""
        for module in self._low_level_modules():
            for p in module.parameters():
                p.requires_grad = False
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def unfreeze(self) -> None:
        """Unfreeze the entire backbone (weights AND BatchNorm statistics)."""
        for p in self.parameters():
            p.requires_grad = True
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()

    def low_level_parameters(self) -> Iterator[nn.Parameter]:
        """Yield stem + layer1 parameters (slowest LR tier)."""
        for module in self._low_level_modules():
            yield from module.parameters()

    def high_level_parameters(self) -> Iterator[nn.Parameter]:
        """Yield transition + stage parameters (middle LR tier)."""
        for module in self._high_level_modules():
            yield from module.parameters()


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python model/backbone.py
if __name__ == "__main__":
    net = HRNetBackbone(pretrained=None)  # no checkpoint for a shape check
    dummy = torch.randn(2, 3, 480, 480)
    feats = net(dummy)
    expected = [(2, 32, 120, 120), (2, 64, 60, 60),
                (2, 128, 30, 30), (2, 256, 15, 15)]
    for i, (f, e) in enumerate(zip(feats, expected)):
        print(f"branch {i}: {tuple(f.shape)} (expected {e})")
    n = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"backbone params: {n:.2f}M (HRNetV2-W32 trunk ~ 28-29M)")
