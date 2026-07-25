"""ResNet-18/34 backbone for a DeepLab-v1-style segmenter.

The complete pretrained ResNet is retained. The downsampling operations in
layer3 and layer4 are removed, so the final feature stays at OUTPUT STRIDE 8
(H/8 x W/8) instead of the classifier's stride 32. This is THE core DeepLab
idea: don't throw away spatial resolution and try to upsample it back later
(the FCN approach) -- just never throw it away in the first place.

Removing a stride alone would SHRINK every following conv's receptive field
(each conv now looks at a 2x smaller window of the input than it was trained
for). Dilation (atrous convolution) compensates: a 3x3 conv with dilation d
samples the same 3x3 pattern but with gaps of d-1 pixels, covering a
(2d+1)x(2d+1) window with the SAME 9 weights -- so the pretrained weights
still "see" the input at the geometry they were trained on, only on a denser
output grid.

Dilation schedule (matches torchvision's replace_stride_with_dilation, so the
pretrained weights transfer exactly as intended):

    layer3 (stride 2 removed -> stays at stride 8):
        first BasicBlock -> dilation 1   (inherits the rate BEFORE this stage)
        later blocks     -> dilation 2   (2x to make up for the removed /2)

    layer4 (stride 2 removed -> stays at stride 8):
        first BasicBlock -> dilation 2   (inherits layer3's rate)
        later blocks     -> dilation 4   (another 2x for the second removed /2)

The "first block inherits, later blocks double" pattern keeps the receptive
field growing at exactly the same rate as the original stride-32 network.
"""

from __future__ import annotations

from typing import Dict, Iterator, Tuple

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    resnet18,
    resnet34,
)


# Supported backbones -> (constructor, weights enum, final feature channels).
# ResNet-18 and -34 both use BasicBlock, so layer4 outputs 512 channels for
# both; only the number of blocks per stage differs. (ResNet-50+ would need
# Bottleneck support in the helpers below -- its stride sits on conv2, not
# conv1 -- so they are intentionally not listed.)
_RESNET_ARCHS = {
    "resnet18": (resnet18, ResNet18_Weights, 512),
    "resnet34": (resnet34, ResNet34_Weights, 512),
}


def _set_conv_dilation(
    conv: nn.Conv2d,
    dilation: int,
) -> None:
    """Set a 3x3 conv's dilation while keeping its output size unchanged.

    Input:
        conv: an existing 3x3 nn.Conv2d (modified IN PLACE -- the weights are
            untouched, only the sampling geometry changes).
        dilation: the new atrous rate (1 = ordinary convolution).

    Only dilation and padding are changed; the 9 pretrained weights are reused
    as-is -- that is the whole point of atrous convolution.
    """

    if not isinstance(conv, nn.Conv2d):
        raise TypeError(
            f"expected Conv2d, got {type(conv).__name__}"
        )

    if conv.kernel_size != (3, 3):
        raise ValueError(
            f"expected 3x3 convolution, "
            f"got kernel_size={conv.kernel_size}"
        )

    conv.dilation = (dilation, dilation)

    # For a 3x3 conv, padding must equal the dilation to preserve H, W:
    # dilation=1 -> padding=1
    # dilation=2 -> padding=2
    # dilation=4 -> padding=4
    # (general rule: padding = dilation * (kernel_size - 1) // 2)
    conv.padding = (dilation, dilation)


def _remove_stage_downsampling(
    stage: nn.Sequential,
) -> None:
    """Remove the 2x downsampling of a ResNet stage's FIRST block (in place).

    Input:
        stage: one residual stage (net.layer3 or net.layer4) whose first block
            downsamples by stride 2.

    A ResNet stage downsamples exactly once, in its first block, and in TWO
    parallel places that must stay consistent: the main branch's strided conv
    AND the 1x1 shortcut ("downsample") conv. Both are set to stride 1 here;
    if only one were changed, the residual add would fail on mismatched H, W.
    """

    if len(stage) == 0:
        raise ValueError("cannot modify an empty residual stage")

    first_block = stage[0]

    # ResNet-18/34 BasicBlocks downsample in the first block's conv1.
    first_block.conv1.stride = (1, 1)

    # BasicBlock also records its stride as an attribute; keep it in sync
    # (torchvision only uses it for repr/debugging, but stale info misleads).
    first_block.stride = 1

    # The shortcut branch must drop its downsampling too, otherwise the main
    # branch (now full resolution) and the shortcut (still halved) could not
    # be added together.
    if first_block.downsample is None:
        raise ValueError(
            "expected first block to contain a downsample path"
        )

    shortcut_conv = first_block.downsample[0]

    if not isinstance(shortcut_conv, nn.Conv2d):
        raise TypeError(
            "expected downsample[0] to be Conv2d"
        )

    shortcut_conv.stride = (1, 1)


def _replace_stage_stride_with_progressive_dilation(
    stage: nn.Sequential,
    first_block_dilation: int,
    later_block_dilation: int,
) -> None:
    """Remove a stage's downsampling and apply progressive dilation (in place).

    Input:
        stage: one residual stage (net.layer3 or net.layer4).
        first_block_dilation: rate for the FIRST block -- the rate inherited
            from the previous stage (its convs still consume the previous
            stage's feature geometry).
        later_block_dilation: rate for all LATER blocks -- doubled, to restore
            the receptive-field growth the removed stride would have provided.

    This mirrors torchvision's replace_stride_with_dilation bookkeeping: the
    first block keeps the OLD rate, subsequent blocks use the NEW rate.
    """

    if first_block_dilation < 1:
        raise ValueError(
            "first_block_dilation must be positive"
        )

    if later_block_dilation < 1:
        raise ValueError(
            "later_block_dilation must be positive"
        )

    if len(stage) == 0:
        raise ValueError("cannot modify an empty residual stage")

    # First cancel the stage's stride-2 downsampling.
    _remove_stage_downsampling(stage)

    for block_index, block in enumerate(stage):
        if block_index == 0:
            dilation = first_block_dilation
        else:
            dilation = later_block_dilation

        # A ResNet-18/34 BasicBlock contains two 3x3 convs; both get the rate.
        _set_conv_dilation(
            block.conv1,
            dilation=dilation,
        )

        _set_conv_dilation(
            block.conv2,
            dilation=dilation,
        )

        # The 1x1 conv inside the shortcut ("downsample") branch is left
        # alone: dilation is meaningless for a 1x1 kernel (there are no
        # neighboring taps to spread apart).


class ResNetBackbone(nn.Module):
    """Full ResNet-18/34 with layer3/layer4 dilated -> output stride 8.

    Args:
        arch: which ResNet, "resnet18" or "resnet34". Both end at 512
            channels, so the neck needs no change when switching.
        pretrained: if True, load ImageNet-pretrained weights BEFORE the
            dilation surgery (the surgery reuses the weights unchanged).
        freeze: if True, freeze the whole backbone immediately.

    Attributes:
        out_channels (int): channels of the returned feature (512).
        output_stride (int): spatial stride of the returned feature (8).
        stage_channels (tuple): channels after (layer1..layer4) =
            (64, 128, 256, 512), exposed for return_intermediate users.

    Contrast with the FCN project's backbone: that one TAPPED 4 feature maps
    (strides 4/8/16/32) and needed an FPN to fuse them back to high
    resolution. Here the network never drops below stride 8, so ONE output
    suffices and no fusion neck is needed.
    """

    # Names of the low-level stages (generic edges/textures, changed least by
    # the dilation surgery) vs. the high-level ones (semantics, and the two
    # stages whose geometry we actually modified). The two-stage finetune
    # helpers below freeze/unfreeze along this split.
    LOW_LEVEL_STAGES = ("stem", "layer1", "layer2")
    HIGH_LEVEL_STAGES = ("layer3", "layer4")

    def __init__(
        self,
        arch: str = "resnet34",
        pretrained: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()

        if arch not in _RESNET_ARCHS:
            raise ValueError(
                f"unsupported backbone {arch!r}; "
                f"choose from {tuple(_RESNET_ARCHS)}"
            )

        constructor, weights_enum, final_channels = (
            _RESNET_ARCHS[arch]
        )

        weights = (
            weights_enum.DEFAULT
            if pretrained
            else None
        )

        # Load the complete ImageNet-pretrained network first; the dilation
        # surgery below only edits stride/dilation attributes, never weights.
        net = constructor(weights=weights)

        # Drop avgpool and fc (they collapse H, W for whole-image
        # classification); keep the entire convolutional trunk.
        # "stem" = conv1 -> bn1 -> relu -> maxpool, i.e. everything before
        # the residual stages; after it the feature map is at stride 4.
        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
        )

        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # Original ResNet strides:        After the surgery:
        #
        # layer1 -> OS4                   layer1 -> OS4   (untouched)
        # layer2 -> OS8                   layer2 -> OS8   (untouched)
        # layer3 -> OS16                  layer3 -> OS8   (stride removed,
        #                                   first block rate 1, later rate 2)
        # layer4 -> OS32                  layer4 -> OS8   (stride removed,
        #                                   first block rate 2, later rate 4)

        _replace_stage_stride_with_progressive_dilation(
            self.layer3,
            first_block_dilation=1,
            later_block_dilation=2,
        )

        _replace_stage_stride_with_progressive_dilation(
            self.layer4,
            first_block_dilation=2,
            later_block_dilation=4,
        )

        self.arch = arch
        self.out_channels = final_channels
        self.output_stride = 8

        self.stage_channels: Tuple[int, int, int, int] = (
            64,
            128,
            256,
            512,
        )

        if freeze:
            self.freeze()

    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """Extract the stride-8 high-level feature.

        Input:
            x: image batch [B, 3, H, W] (H, W ideally multiples of 8 so the
                stride-8 grid is exact).
            return_intermediate: if True, return all four stage outputs in a
                dict (useful for debugging / future skip connections).

        Output:
            c5 [B, 512, H/8, W/8] by default -- note c4 and c5 share c3's
            RESOLUTION now (that is the dilation surgery at work); only the
            channel count and semantic depth grow.
        """

        x = self.stem(x)
        # [B, 64, H/4, W/4]

        c2 = self.layer1(x)
        # [B, 64, H/4, W/4]

        c3 = self.layer2(c2)
        # [B, 128, H/8, W/8]

        c4 = self.layer3(c3)
        # [B, 256, H/8, W/8]  <- stride 8, NOT 16: downsampling removed

        c5 = self.layer4(c4)
        # [B, 512, H/8, W/8]  <- stride 8, NOT 32

        if return_intermediate:
            return {
                "c2": c2,
                "c3": c3,
                "c4": c4,
                "c5": c5,
            }

        return c5

    # ---- Helpers for two-stage finetuning -----------------------------------
    # Stage 1 of this project freezes only the LOW stages: stem/layer1/layer2
    # are untouched pretrained weights extracting generic edges/textures,
    # while layer3/layer4 -- the stages whose strides we surgically replaced
    # with dilation -- should adapt together with the new neck/head from the
    # start. Stage 2 then unfreezes everything with layered learning rates.

    def _stage_modules(self) -> Dict[str, nn.Module]:
        """Map stage name -> module, for the freeze/unfreeze helpers."""
        return {
            "stem": self.stem,
            "layer1": self.layer1,
            "layer2": self.layer2,
            "layer3": self.layer3,
            "layer4": self.layer4,
        }

    def freeze(self) -> None:
        """Freeze the entire backbone (weights AND BatchNorm statistics)."""

        for parameter in self.parameters():
            parameter.requires_grad = False

        self._set_batch_norm_training(False)

    def unfreeze(self) -> None:
        """Unfreeze the entire backbone (weights AND BatchNorm statistics)."""

        for parameter in self.parameters():
            parameter.requires_grad = True

        self._set_batch_norm_training(True)

    def freeze_low_layers(
        self,
        layers: Tuple[str, ...] = LOW_LEVEL_STAGES,
    ) -> None:
        """Freeze only the named low-level stages (stage-1 protocol).

        Input:
            layers: stages to freeze, default ("stem", "layer1", "layer2").

        Freezing stops both the gradients (requires_grad=False) and the
        BatchNorm running-stat updates (module.eval()) of those stages, so
        their pretrained state is fully preserved while the modified high
        stages + new neck/head warm up.

        NOTE: model.train() at each epoch start flips ALL BatchNorms back to
        training mode; the model's set_bn_eval_on_frozen() must be called
        after it to restore eval mode on the frozen ones.
        """

        available = self._stage_modules()
        unknown = set(layers) - set(available)

        if unknown:
            raise ValueError(
                f"unknown layers: {sorted(unknown)}"
            )

        for name in layers:
            module = available[name]

            for parameter in module.parameters():
                parameter.requires_grad = False

            for child in module.modules():
                if isinstance(child, nn.BatchNorm2d):
                    child.eval()

    def unfreeze_high_layers(
        self,
        layers: Tuple[str, ...] = HIGH_LEVEL_STAGES,
        train_batch_norm: bool = True,
    ) -> None:
        """Unfreeze only the named high-level stages.

        Input:
            layers: stages to unfreeze, default ("layer3", "layer4").
            train_batch_norm: whether the BatchNorms in those stages also
                update their running statistics. With small batch sizes the
                per-batch statistics are noisy -- set False to keep the
                pretrained statistics while still training the conv weights.
        """

        available = self._stage_modules()
        unknown = set(layers) - set(available)

        if unknown:
            raise ValueError(
                f"unknown layers: {sorted(unknown)}"
            )

        for name in layers:
            module = available[name]

            for parameter in module.parameters():
                parameter.requires_grad = True

            for child in module.modules():
                if isinstance(child, nn.BatchNorm2d):
                    child.train(train_batch_norm)

    def low_level_parameters(self) -> Iterator[nn.Parameter]:
        """Yield the parameters of stem/layer1/layer2 (for layered LRs)."""

        for name in self.LOW_LEVEL_STAGES:
            yield from self._stage_modules()[name].parameters()

    def high_level_parameters(self) -> Iterator[nn.Parameter]:
        """Yield the parameters of layer3/layer4 (for layered LRs)."""

        for name in self.HIGH_LEVEL_STAGES:
            yield from self._stage_modules()[name].parameters()

    def freeze_batch_norm(self) -> None:
        """Freeze the running statistics of ALL backbone BatchNorms."""

        self._set_batch_norm_training(False)

    def _set_batch_norm_training(
        self,
        training: bool,
    ) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train(training)


def _print_stage_dilations(
    stage: nn.Sequential,
    stage_name: str,
) -> None:
    """Print each BasicBlock's dilation and stride (surgery sanity check)."""

    for index, block in enumerate(stage):
        print(
            f"{stage_name}[{index}]: "
            f"conv1 dilation={block.conv1.dilation}, "
            f"conv2 dilation={block.conv2.dilation}, "
            f"conv1 stride={block.conv1.stride}"
        )


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python model/backbone.py
if __name__ == "__main__":
    model = ResNetBackbone(
        arch="resnet34",
        pretrained=False,  # no download needed for a shape check
    )

    dummy = torch.randn(
        2,
        3,
        416,
        416,
    )

    features = model(
        dummy,
        return_intermediate=True,
    )

    # Expected: c2 (2, 64, 104, 104), c3 (2, 128, 52, 52),
    #           c4 (2, 256, 52, 52),  c5 (2, 512, 52, 52)
    # -- c4/c5 keep c3's 52x52 resolution: that IS the point.
    for name, feature in features.items():
        print(name, tuple(feature.shape))

    _print_stage_dilations(
        model.layer3,
        "layer3",
    )

    _print_stage_dilations(
        model.layer4,
        "layer4",
    )

    print(
        "out_channels:",
        model.out_channels,
    )

    print(
        "output_stride:",
        model.output_stride,
    )
