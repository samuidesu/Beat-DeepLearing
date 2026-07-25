# DeepLab-v1-style on PASCAL VOC 2012

本项目实现了一个基于 **ResNet-34** 的 DeepLab-v1-style 语义分割模型，并在 **PASCAL VOC 2012 + SBD train_noval** 上训练和评估。

与此前的 FCN-add、FCN-concat 和 U-Net 实验相比，本次实验不再依赖复杂的多尺度 decoder 恢复空间分辨率，而是直接修改 ResNet 后半部分：

- 移除 `layer3` 和 `layer4` 的 stride-2 下采样；
- 使用逐级增大的空洞卷积保持原网络的感受野增长；
- 将最终输出步长保持在 `output stride = 8`；
- 使用单个 LargeFOV 空洞卷积模块进行上下文建模；
- 最后通过 `1×1` 卷积完成逐像素分类，并用双线性插值恢复到输入尺寸。

最终在 VOC 2012 validation 共 **1449 张图像**上取得：

```text
mIoU       0.7135
pixel_acc  0.9350
mean_acc   0.8048
```

这是当前 FCN / U-Net / DeepLab 系列实验中的最好结果。

---

## 1. 模型结构

整体数据流：

```text
image [B, 3, H, W]
  │
  ├─ ResNet-34 stem + layer1
  │      └─ stride 4
  │
  ├─ layer2
  │      └─ stride 8
  │
  ├─ dilated layer3
  │      ├─ 取消原 stride-2 下采样
  │      ├─ 第一个 BasicBlock: dilation = 1
  │      └─ 后续 BasicBlock: dilation = 2
  │
  ├─ dilated layer4
  │      ├─ 取消原 stride-2 下采样
  │      ├─ 第一个 BasicBlock: dilation = 2
  │      └─ 后续 BasicBlock: dilation = 4
  │
  └─ c5 [B, 512, H/8, W/8]
         │
         ├─ LargeFOV: 3×3 atrous conv, dilation = 12
         ├─ Dropout2d
         ├─ 1×1 projection
         └─ feat [B, 128, H/8, W/8]
                │
                ├─ 1×1 classifier
                └─ bilinear interpolation
                       └─ logits [B, 21, H, W]
```

### 1.1 Backbone 的空洞卷积安排

原始 ResNet-34 的空间步长为：

```text
layer1: output stride 4
layer2: output stride 8
layer3: output stride 16
layer4: output stride 32
```

本项目移除了 `layer3` 和 `layer4` 的下采样，使两层都在 stride-8 特征图上运行：

```text
layer1: output stride 4
layer2: output stride 8
layer3: output stride 8
layer4: output stride 8
```

对应 dilation：

| Stage | 第一个 BasicBlock | 后续 BasicBlock |
|---|---:|---:|
| layer3 | 1 | 2 |
| layer4 | 2 | 4 |

修改只改变卷积的 stride、padding 和 dilation，不改变卷积核参数形状，因此仍然可以复用 ImageNet 预训练权重。

### 1.2 LargeFOV neck

Backbone 输出后只使用一张高层 stride-8 特征图，不再使用 FPN 或 U-Net 式多层特征融合。

```text
512 channels
→ 3×3 atrous conv, dilation=12, 256 channels
→ Dropout2d(0.1)
→ 1×1 conv, 128 channels
```

最终 head 仅包含：

```text
1×1 Conv: 128 → 21
bilinear resize: H/8 × W/8 → H × W
```

训练输出为 raw logits，直接输入：

```python
nn.CrossEntropyLoss(ignore_index=255)
```

---

## 2. 数据与训练设置

### 2.1 数据集

训练数据：

```text
PASCAL VOC 2012 train
+
SBD train_noval
```

验证数据：

```text
PASCAL VOC 2012 val
共 1449 张图像
```

类别数：

```text
20 个前景类别 + 1 个 background = 21 类
```

标签中的 `255` 作为 ignore index，不参与 loss 和指标统计。

### 2.2 两阶段训练

本实验共训练 62 epochs：

| Stage | Epoch | 训练模块 |
|---|---:|---|
| Stage 1 | 32 | 冻结 `stem/layer1/layer2`；训练 `layer3/layer4/neck/head` |
| Stage 2 | 30 | 解冻完整 backbone，端到端微调 |

Stage 1 并不是简单的“只训练 head”。由于修改过采样几何的 `layer3/layer4` 从一开始就是可训练的，所以 Stage 1 已经训练了模型的大多数参数。

训练日志中的可训练参数为：

```text
21.15M
```

---

## 3. 训练曲线

### 3.1 Cross-entropy loss

![DeepLab loss](assets/loss_curve.png)

主要现象：

- Stage 1 的训练 loss 从约 `0.75` 持续下降到 `0.13`；
- Stage 1 后期验证 loss 稳定在约 `0.23`；
- Stage 2 开始时 train/val loss 均出现短暂上升；
- 后续训练 loss 继续下降到约 `0.10`；
- 验证 loss 最终维持在约 `0.23～0.24`。

### 3.2 Validation mIoU 与 pixel accuracy

![DeepLab validation metrics](assets/metrics_curve.png)

主要节点：

| Epoch | Stage | mIoU | pixel accuracy |
|---:|---:|---:|---:|
| 1 | 1 | 0.5264 | 0.8858 |
| 12 | 1 | 0.6147 | 0.9121 |
| 18 | 1 | 0.6614 | 0.9203 |
| 23 | 1 | 0.6817 | 0.9248 |
| 27 | 1 | **0.7030** | 0.9317 |
| 32 | 1 | 0.6985 | 0.9311 |
| 33 | 2 | 0.6366 | 0.9141 |
| 34 | 2 | 0.6967 | 0.9266 |
| 51 | 2 | 0.7013 | 0.9314 |
| 56 | 2 | 0.7058 | 0.9313 |
| 59 | 2 | **0.7110** | 0.9328 |
| 62 | 2 | 0.7087 | 0.9326 |

Stage 1 最佳 proxy mIoU 为：

```text
0.7030 @ epoch 27
```

Stage 2 最佳 proxy mIoU 为：

```text
0.7110 @ epoch 59
```

---

## 4. 最终评估结果

加载：

```text
outputs/best.pt
```

在完整 VOC 2012 validation 上评估：

```text
Val images: 1449
```

### 4.1 每类 IoU

| Class | IoU |
|---|---:|
| chair | 0.3808 |
| bicycle | 0.4114 |
| sofa | 0.4138 |
| diningtable | 0.4892 |
| pottedplant | 0.5726 |
| boat | 0.6232 |
| tvmonitor | 0.7240 |
| horse | 0.7297 |
| cow | 0.7297 |
| bottle | 0.7305 |
| train | 0.7876 |
| motorbike | 0.7879 |
| sheep | 0.7921 |
| dog | 0.8022 |
| car | 0.8053 |
| person | 0.8343 |
| bird | 0.8425 |
| cat | 0.8599 |
| aeroplane | 0.8610 |
| bus | 0.8760 |
| background | 0.9307 |

### 4.2 汇总指标

```text
mIoU       0.7135
pixel_acc  0.9350
mean_acc   0.8048
```

---

## 5. 与此前实验对比

此前三个模型在相同系列数据和训练协议下均停留在约 `0.69 mIoU`：

| Model | 最细特征层 | Decoder | mIoU | pixel acc | mean acc |
|---|---|---|---:|---:|---:|
| FCN-add | stride 4 | FPN add | 0.6910 | 0.9295 | 0.7811 |
| FCN-concat | stride 4 | FPN concat | 0.6932 | 0.9292 | 0.7875 |
| U-Net | stride 2 | transpose conv + concat | 0.6894 | 0.9299 | 0.7863 |
| **DeepLab-v1-style** | **stride 8** | **LargeFOV，无多层 decoder** | **0.7135** | **0.9350** | **0.8048** |

相对此前最好的 FCN-concat：

```text
mIoU: 0.6932 → 0.7135
提升: +0.0203
```

相对 FCN-add：

```text
mIoU:      0.6910 → 0.7135  (+0.0225)
pixel_acc: 0.9295 → 0.9350  (+0.0055)
mean_acc:  0.7811 → 0.8048  (+0.0237)
```

此前 FCN-add、FCN-concat 和 U-Net 的 mIoU 极差只有 `0.0038`，基本处于单次训练波动范围；本次约 2 个百分点的提升明显超过此前 decoder 结构间的差距。

这说明当前实验的主要收益并不是来自更高分辨率的 decoder，而是来自：

1. 不再把深层特征降到 stride 32；
2. 让完整 `layer3/layer4` 在 stride-8 特征图上继续进行高级语义计算；
3. 使用 LargeFOV 空洞卷积增强上下文建模。

---

## 6. 每类结果变化

与 FCN-add 对比，21 个类别中有 20 个类别提升，只有 `car` 略有下降。

主要提升：

| Class | FCN-add | DeepLab | Delta |
|---|---:|---:|---:|
| chair | 0.3080 | 0.3808 | **+0.0728** |
| sofa | 0.3534 | 0.4138 | **+0.0604** |
| tvmonitor | 0.6864 | 0.7240 | **+0.0376** |
| horse | 0.6939 | 0.7297 | **+0.0358** |
| cow | 0.6974 | 0.7297 | **+0.0323** |
| bird | 0.8172 | 0.8425 | **+0.0253** |
| diningtable | 0.4661 | 0.4892 | **+0.0231** |
| bottle | 0.7085 | 0.7305 | **+0.0220** |
| aeroplane | 0.8407 | 0.8610 | **+0.0203** |
| pottedplant | 0.5538 | 0.5726 | **+0.0188** |

下降类别：

| Class | FCN-add | DeepLab | Delta |
|---|---:|---:|---:|
| car | 0.8162 | 0.8053 | **-0.0109** |

背景 IoU 只提升：

```text
0.9265 → 0.9307
```

因此总 mIoU 的提升主要来自前景类别，而不是模型单纯变得更偏向背景。

---

## 7. Stage 2 开始时为什么指标下降

Stage 1 最后一轮：

```text
epoch 32
lr         2.41e-6
train loss 0.1325
val loss   0.2312
mIoU       0.6985
```

Stage 2 第一轮：

```text
epoch 33
lr         1.00e-4
train loss 0.1850
val loss   0.2826
mIoU       0.6366
```

阶段切换同时发生了以下变化：

1. 学习率从约 `2.4e-6` 重启到 `1e-4`；
2. `stem/layer1/layer2` 从冻结变为可训练；
3. 低层特征分布开始变化；
4. 原本与旧低层特征适配的 `layer3/layer4/neck/head` 需要重新协调；
5. 冻结低层中的 BatchNorm 从固定统计量转回训练状态。

因此 loss 上升和 mIoU 暂时下降属于合理的优化扰动。

模型在 Stage 2 第二个 epoch 已恢复到：

```text
mIoU = 0.6967
```

后续继续提升到：

```text
mIoU = 0.7110
```

说明这不是训练崩溃，只是阶段切换造成的短暂不稳定。

---

## 8. 实验结论

### 8.1 当前瓶颈并非 decoder 分辨率

此前：

```text
FCN-add:     stride 4
FCN-concat:  stride 4
U-Net:       stride 2
```

三个模型全部停留在约 `0.69 mIoU`。

当前 DeepLab 最终预测前只有 stride-8 特征，空间分辨率反而更低，但 mIoU 达到 `0.7135`。

因此当前条件下：

> 深层语义特征和上下文建模比单纯提高 decoder 输出分辨率更重要。

### 8.2 Stage 1 已完成大部分优化

Stage 1 已经训练：

```text
layer3
layer4
neck
head
```

并达到 `0.7030` proxy mIoU。

Stage 2 解冻低层后，最佳 proxy 提升到 `0.7110`，增益约为：

```text
+0.0080
```

Stage 2 仍然有效，但作用更接近最终微调，而不是重新学习整个分割模型。

### 8.3 当前模型的代价是计算量

虽然 head 和 neck 很轻，但 `layer3/layer4` 都在 stride-8 特征图上运行，带来了更高的 activation 和计算成本。

本次每个 epoch 约为：

```text
Stage 1: 262 s
Stage 2: 300 s
```

相比此前 FCN/U-Net 明显更慢。

因此 DeepLab 的提升并不是“免费”的，而是用更高的高分辨率深层计算换来的。

---

## 9. 下一步实验

### 9.1 优化两阶段训练

建议优先调整训练策略：

- Stage 1 从 32 epochs 缩短到约 26 epochs；
- Stage 2 从 Stage 1 最佳 checkpoint 开始；
- Stage 2 前 1～3 epochs 使用 warmup；
- 对不同参数组使用分层学习率；
- 对照实验：Stage 2 中固定 backbone BatchNorm running statistics。

建议的 Stage 2 初始学习率：

```text
backbone_low:  5e-6
backbone_high: 2e-5
neck_head:     5e-5
```

### 9.2 LargeFOV dilation 消融

保持其他设置完全不变，只比较：

```text
dilation = 6
dilation = 12
```

用于确认当前单一 LargeFOV 的最佳尺度。

### 9.3 DeepLab v2 / ASPP

将单个 dilation-12 分支改成并行多尺度空洞卷积：

```text
c5
├─ dilation 6
├─ dilation 12
├─ dilation 18
└─ dilation 24
```

这是当前结果之后最自然的结构升级方向。

### 9.4 Output stride 16 效率对照

当前 OS8 计算成本较高，可以增加 OS16 实验：

```text
OS8  vs.  OS16
```

比较：

- mIoU；
- 显存；
- 每 epoch 时间；
- 推理速度。

---

## 10. 最终结论

本次实验得到以下主要结论：

1. ResNet-34 DeepLab-v1-style 达到 **0.7135 mIoU**，刷新当前系列最好结果；
2. 相比 FCN/U-Net 的约 0.69 平台，提高约 2 个百分点；
3. 提升主要来自前景类别，而不是背景预测偏置；
4. chair、sofa、tvmonitor、horse、cow 等依赖整体语义和上下文的类别提升最明显；
5. 单纯增加 decoder 分辨率不是当前瓶颈；
6. 保持深层特征为 stride 8，并在高分辨率上继续进行深层语义处理，是本次提升的主要原因；
7. 下一步应优先优化 Stage 2 切换策略，并进入 DeepLab v2 ASPP 实验。
