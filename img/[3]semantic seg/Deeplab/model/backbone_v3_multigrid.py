"""ResNet-34 backbone for a DeepLab-v3-style segmenter with Multi-Grid.

This file is a drop-in replacement for the project's previous ``backbone.py``.
It keeps the complete ImageNet-pretrained ResNet-34 convolutional trunk and
converts it to an atrous backbone.

Important paper-mapping note
----------------------------
DeepLab-v3 applies Multi-Grid to ResNet's final residual stage, called
``block4`` in the paper.  In torchvision naming this is ``layer4``.  The paper
does NOT apply Multi-Grid to both layer3 and layer4.

For output stride 8, this ResNet-34 adaptation uses:

    layer3: all BasicBlocks use dilation 2
    layer4: block-level Multi-Grid with base dilation 4
            MG=(1, 2, 4) -> actual block dilations=(4, 8, 16)

For output stride 16:

    layer3: unchanged, still downsamples to OS16
    layer4: block-level Multi-Grid with base dilation 2
            MG=(1, 2, 4) -> actual block dilations=(2, 4, 8)

The original DeepLab-v3 paper used ResNet-101 Bottleneck units.  ResNet-34 uses
BasicBlock units containing two spatial 3x3 convolutions, so this adaptation
assigns one Multi-Grid rate to each BasicBlock and applies that rate to BOTH
3x3 convolutions inside the block.

Only stride, dilation, and padding attributes are changed.  Convolution weight
shapes and values are untouched, so ImageNet pretrained weights remain usable.
"""

from __future__ import annotations

from typing import Dict, Iterator, Sequence, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34


def _set_conv_dilation(conv: nn.Conv2d, dilation: int) -> None:
    """Set a 3x3 convolution's dilation and matching same-size padding."""

    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"expected Conv2d, got {type(conv).__name__}")
    if conv.kernel_size != (3, 3):
        raise ValueError(
            f"expected a 3x3 convolution, got kernel_size={conv.kernel_size}"
        )
    if dilation < 1:
        raise ValueError("dilation must be positive")

    conv.dilation = (dilation, dilation)
    conv.padding = (dilation, dilation)


def _remove_stage_downsampling(stage: nn.Sequential) -> None:
    """Remove the stride-2 transition in a ResNet-34 stage's first block."""

    if len(stage) == 0:
        raise ValueError("cannot modify an empty residual stage")

    first_block = stage[0]

    # torchvision ResNet-34 BasicBlock downsamples in conv1.
    first_block.conv1.stride = (1, 1)
    first_block.stride = 1

    # The shortcut must use the same spatial resolution as the main path.
    if first_block.downsample is None:
        raise ValueError("expected the first block to have a downsample path")

    shortcut_conv = first_block.downsample[0]
    if not isinstance(shortcut_conv, nn.Conv2d):
        raise TypeError("expected downsample[0] to be Conv2d")

    shortcut_conv.stride = (1, 1)


def _set_basicblock_dilation(block: nn.Module, dilation: int) -> None:
    """Apply one block-level atrous rate to both 3x3 convs of a BasicBlock."""

    if not hasattr(block, "conv1") or not hasattr(block, "conv2"):
        raise TypeError("expected a torchvision ResNet BasicBlock")

    _set_conv_dilation(block.conv1, dilation)
    _set_conv_dilation(block.conv2, dilation)


def _replace_stage_stride_with_uniform_dilation(
    stage: nn.Sequential,
    dilation: int,
) -> None:
    """Remove a stage's downsampling and use one dilation for every block."""

    _remove_stage_downsampling(stage)

    for block in stage:
        _set_basicblock_dilation(block, dilation)


def _replace_stage_stride_with_multigrid(
    stage: nn.Sequential,
    base_dilation: int,
    multi_grid: Sequence[int],
) -> Tuple[int, ...]:
    """Remove downsampling and apply paper-style block-level Multi-Grid.

    ``actual_dilation[i] = base_dilation * multi_grid[i]``

    ResNet-34 layer4 contains three BasicBlocks, so the paper's
    ``Multi-Grid=(1, 2, 4)`` maps naturally to the three blocks.
    """

    if base_dilation < 1:
        raise ValueError("base_dilation must be positive")
    if len(stage) != len(multi_grid):
        raise ValueError(
            "multi_grid length must equal the number of residual blocks: "
            f"len(stage)={len(stage)}, len(multi_grid)={len(multi_grid)}"
        )
    if any(rate < 1 for rate in multi_grid):
        raise ValueError("all Multi-Grid unit rates must be positive")

    _remove_stage_downsampling(stage)

    actual_dilations = tuple(
        base_dilation * int(unit_rate) for unit_rate in multi_grid
    )

    for block, dilation in zip(stage, actual_dilations):
        _set_basicblock_dilation(block, dilation)

    return actual_dilations


class ResNetBackbone(nn.Module):
    """DeepLab-v3-style ResNet-34 backbone with block4/layer4 Multi-Grid.

    Args:
        arch: kept for compatibility with the existing DeepLab model.  This
            file intentionally supports only ``"resnet34"`` because the default
            paper-style Multi-Grid tuple has three entries, matching ResNet-34's
            three layer4 BasicBlocks.
        pretrained: load ImageNet pretrained weights before atrous surgery.
        freeze: freeze all backbone weights and BatchNorm running statistics.
        output_stride: 8 or 16.  The current project normally uses 8.
        block4_multi_grid: unit rates for torchvision ``layer4``.  With OS8,
            ``(1,2,4)`` produces actual dilations ``(4,8,16)``.

    Output:
        By default, c5 ``[B, 512, H/output_stride, W/output_stride]``.
    """

    LOW_LEVEL_STAGES = ("stem", "layer1", "layer2")
    HIGH_LEVEL_STAGES = ("layer3", "layer4")

    def __init__(
        self,
        arch: str = "resnet34",
        pretrained: bool = True,
        freeze: bool = False,
        output_stride: int = 8,
        block4_multi_grid: Sequence[int] = (1, 2, 4),
    ) -> None:
        super().__init__()

        if arch != "resnet34":
            raise ValueError(
                "backbone_v3_multigrid.py supports only 'resnet34'; "
                f"got {arch!r}"
            )
        if output_stride not in (8, 16):
            raise ValueError("output_stride must be 8 or 16")

        weights = ResNet34_Weights.DEFAULT if pretrained else None
        net = resnet34(weights=weights)

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

        # torchvision layer names vs. DeepLab paper names:
        # layer3 == block3, layer4 == block4.
        # Multi-Grid is applied ONLY to block4/layer4.
        if output_stride == 8:
            # Remove layer3's /2 and use the OS8 base atrous rate 2.
            _replace_stage_stride_with_uniform_dilation(
                self.layer3,
                dilation=2,
            )
            block4_base_dilation = 4
        else:
            # Keep layer3's original stride-2 transition: its output is OS16.
            block4_base_dilation = 2

        self.block4_multi_grid = tuple(int(x) for x in block4_multi_grid)
        self.block4_actual_dilations = _replace_stage_stride_with_multigrid(
            self.layer4,
            base_dilation=block4_base_dilation,
            multi_grid=self.block4_multi_grid,
        )

        self.arch = arch
        self.out_channels = 512
        self.output_stride = output_stride
        self.stage_channels: Tuple[int, int, int, int] = (64, 128, 256, 512)

        if freeze:
            self.freeze()

    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)   # OS4
        c3 = self.layer2(c2)  # OS8
        c4 = self.layer3(c3)  # OS8 or OS16
        c5 = self.layer4(c4)  # same OS as c4

        if return_intermediate:
            return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}
        return c5

    # ---- Helpers retained for the project's two-stage training protocol -----

    def _stage_modules(self) -> Dict[str, nn.Module]:
        return {
            "stem": self.stem,
            "layer1": self.layer1,
            "layer2": self.layer2,
            "layer3": self.layer3,
            "layer4": self.layer4,
        }

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        self._set_batch_norm_training(False)

    def unfreeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True
        self._set_batch_norm_training(True)

    def freeze_low_layers(
        self,
        layers: Tuple[str, ...] = LOW_LEVEL_STAGES,
    ) -> None:
        available = self._stage_modules()
        unknown = set(layers) - set(available)
        if unknown:
            raise ValueError(f"unknown layers: {sorted(unknown)}")

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
        available = self._stage_modules()
        unknown = set(layers) - set(available)
        if unknown:
            raise ValueError(f"unknown layers: {sorted(unknown)}")

        for name in layers:
            module = available[name]
            for parameter in module.parameters():
                parameter.requires_grad = True
            for child in module.modules():
                if isinstance(child, nn.BatchNorm2d):
                    child.train(train_batch_norm)

    def low_level_parameters(self) -> Iterator[nn.Parameter]:
        for name in self.LOW_LEVEL_STAGES:
            yield from self._stage_modules()[name].parameters()

    def high_level_parameters(self) -> Iterator[nn.Parameter]:
        for name in self.HIGH_LEVEL_STAGES:
            yield from self._stage_modules()[name].parameters()

    def freeze_batch_norm(self) -> None:
        self._set_batch_norm_training(False)

    def _set_batch_norm_training(self, training: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train(training)


def _print_stage_dilations(stage: nn.Sequential, stage_name: str) -> None:
    for index, block in enumerate(stage):
        print(
            f"{stage_name}[{index}]: "
            f"conv1 dilation={block.conv1.dilation}, "
            f"conv2 dilation={block.conv2.dilation}, "
            f"conv1 stride={block.conv1.stride}"
        )


if __name__ == "__main__":
    model = ResNetBackbone(
        arch="resnet34",
        pretrained=False,
        output_stride=8,
        block4_multi_grid=(1, 2, 4),
    )

    dummy = torch.randn(2, 3, 416, 416)
    features = model(dummy, return_intermediate=True)

    for name, feature in features.items():
        print(name, tuple(feature.shape))

    print("layer3: uniform dilation 2; Multi-Grid is NOT applied here")
    _print_stage_dilations(model.layer3, "layer3")

    print("layer4/block4 Multi-Grid:", model.block4_multi_grid)
    print("layer4 actual dilations:", model.block4_actual_dilations)
    _print_stage_dilations(model.layer4, "layer4")

    print("out_channels:", model.out_channels)
    print("output_stride:", model.output_stride)
