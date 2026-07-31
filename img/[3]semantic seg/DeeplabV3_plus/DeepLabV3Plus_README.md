# DeepLabv3+ on PASCAL VOC 2012

基于 **ResNet-34 + DeepLabv3 ASPP + DeepLabv3+ decoder** 的语义分割实验。

本实验延续前面的 DeepLab v1 / v2 / v3 实现，在 v3 的 ASPP encoder 后加入轻量 decoder：将高层语义特征上采样到输出步长 4，与 ResNet `layer1` 的低层特征融合，再进行分类。实验重点不是复现论文最高成绩，而是在相同数据和训练框架下观察 decoder 对边界、细小结构和困难类别的影响。

## 实验结果

| 指标 | 结果 |
|---|---:|
| Full-val mIoU | **0.7214** |
| Pixel Accuracy | **0.9363** |
| Mean Accuracy | **0.8304** |
| 最佳 proxy mIoU | **0.7294** |
| 最佳 proxy epoch | **60** |
| 模型参数量 | **26.72M** |

> 最终结果在完整的 1,449 张 VOC 2012 validation images 上计算。训练过程中每轮使用固定的前 300 个 validation batches 作为快速 proxy，因此 `0.7294` 不能直接视为完整验证集成绩。

与此前相同配置下的 DeepLabv3 baseline 相比：

| 模型 | mIoU | Pixel Acc | Mean Acc |
|---|---:|---:|---:|
| DeepLabv3 | 0.7210 | 0.9360 | 0.8163 |
| DeepLabv3+ | **0.7214** | **0.9363** | **0.8304** |
| 变化 | +0.0004 | +0.0003 | **+0.0141** |

总体 mIoU 基本持平，但 Mean Accuracy 提高约 1.41 个百分点，说明 decoder 改变了类别间的表现分配，并帮助模型找回了部分困难类别的目标像素；这些收益尚未稳定转化为更高的整体 IoU。

---

## 数据集

- **训练集**：PASCAL VOC 2012 train + SBD `train_noval`
- **训练图像**：7,087
- **验证集**：PASCAL VOC 2012 val
- **验证图像**：1,449
- **类别数**：21，包括背景
- **Ignore index**：255

训练增强：

- Random scale：`0.5 ~ 2.0`
- Random crop：`480 × 480`
- 输入尺寸补齐到 8 的倍数

---

## 模型结构

```text
image [B, 3, H, W]
│
├── ResNet-34 stem + layer1
│     └── c2 [B, 64, H/4, W/4]             low-level feature
│
└── layer2 + dilated layer3/layer4
      └── c5 [B, 512, H/8, W/8]            high-level feature
             │
             └── ASPP
                  ├── 1×1 branch
                  ├── 3×3 atrous, rate=3
                  ├── 3×3 atrous, rate=6
                  ├── 3×3 atrous, rate=9
                  └── global pooling branch
                         │
                     concat: 5 × 256 = 1280 channels
                         │
                     1×1 projection: 1280 → 256
                         │
                     bilinear upsample: OS8 → OS4
                         │
                         ├───────────────┐
                         │               │
c2 [B,64,H/4,W/4]       │               │
│                        │               │
└── 1×1 projection       │               │
    64 → 48              │               │
       │                  │               │
       └──── concat 48 + 256 = 304 ──────┘
                         │
                    3×3, 304 → 256
                    3×3, 256 → 256
                         │
                    1×1 classifier
                    256 → 21
                         │
                    bilinear to H × W
```

### Backbone

使用 ImageNet 预训练的 ResNet-34，并将后两个 stage 改成 output stride 8：

```text
layer1: OS4
layer2: OS8
layer3: 移除 stride，dilation = (1, 2, 2, 2, 2, 2)
layer4: 移除 stride，dilation = (2, 4, 4)
```

当前版本没有使用 Multi-Grid。此前较激进的 `(4, 8, 16)` Multi-Grid 实验没有取得收益，因此本实验回到稳定的 progressive dilation backbone。

### ASPP encoder

每个 ASPP branch 输出 256 通道 feature：

```text
1×1
3×3 rate 3
3×3 rate 6
3×3 rate 9
global pooling
```

五个分支 concat 后得到 1,280 通道，再通过 `1×1 Conv + BN + ReLU + Dropout` 投影到 256 通道。

### Decoder

- `c2`：`64 → 48`
- ASPP feature：256 通道
- concat：304 通道
- 两个 `3×3, 256` refinement convolutions
- `1×1, 256 → 21` 分类
- 最终插值到输入尺寸

这是一种轻量 decoder，而不是 U-Net 式多层对称 decoder。

---

## 参数量

| 模块 | 参数量 |
|---|---:|
| ResNet-34 backbone | 21.28M |
| ASPP encoder | 4.13M |
| Decoder / head | 1.30M |
| **总计** | **26.72M** |

Stage 1 可训练参数约 **25.37M**；Stage 2 全部解冻后为 **26.72M**。

---

## 训练配置

| 项目 | 配置 |
|---|---|
| Batch size | 16 |
| Optimizer | Adam |
| Weight decay | 0.001 |
| Scheduler | CosineAnnealingLR |
| Stage 1 epochs | 40 |
| Stage 2 epochs | 30 |
| 总训练时间 | 约 7.66 小时 |

### Stage 1

冻结：

```text
stem + layer1 + layer2
```

训练：

```text
layer3 + layer4 + ASPP + decoder
```

学习率：

```text
neck/head:      1e-3
backbone-high:  1e-4
```

### Stage 2

全部 backbone 解冻，使用分层学习率：

```text
neck/head:      1e-4
backbone-high:  3e-5
backbone-low:   1e-5
```

运行命令：

```bash
python trainV3.py \
  --epochs-stage1 40 \
  --epochs-stage2 30 \
  --output-dir outputsv3plus
```

只运行 Stage 1：

```bash
python trainV3.py \
  --epochs-stage1 40 \
  --epochs-stage2 0 \
  --output-dir outputsv3plus_stage1
```

---

## 训练曲线

### Loss

![loss curve](assets/loss_curve.png)

### mIoU / Pixel Accuracy

![mIoU curve](assets/miou_curve.png)

### 曲线分析

Stage 1 整体收敛正常：

- epoch 1：proxy mIoU `0.3978`
- epoch 39：proxy mIoU `0.7190`
- epoch 40：proxy mIoU `0.7190`

Stage 2 开始时出现明显扰动：

```text
epoch 40:
train loss = 0.1446
val loss   = 0.2207
mIoU       = 0.7190

epoch 41:
train loss = 0.2003
val loss   = 0.2844
mIoU       = 0.6636
```

原因是 Stage 2 同时发生：

1. 低层 backbone 解冻；
2. optimizer 被重新创建，Adam 动量状态清空；
3. 学习率从 Stage 1 末尾的极小值重新跳到较大的初始值；
4. 低层 BatchNorm statistics 开始变化。

此后模型逐渐恢复，并在 epoch 60 达到最佳 proxy：

```text
train loss = 0.1345
val loss   = 0.2214
proxy mIoU = 0.7294
pixel acc  = 0.9371
```

epoch 60 以后 train loss 继续下降，但 validation 指标没有继续提高，出现轻微过拟合。因此当前 Stage 2 没必要固定跑满 30 epochs。

---

## 完整验证集 Per-class IoU

| Class | IoU |
|---|---:|
| background | 0.9331 |
| aeroplane | 0.8449 |
| bicycle | 0.3943 |
| bird | 0.8762 |
| boat | 0.6757 |
| bottle | 0.7338 |
| bus | 0.9179 |
| car | 0.8126 |
| cat | 0.8633 |
| chair | 0.3821 |
| cow | 0.7377 |
| diningtable | 0.5608 |
| dog | 0.7819 |
| horse | 0.7413 |
| motorbike | 0.7767 |
| person | 0.8288 |
| pottedplant | 0.5657 |
| sheep | 0.7385 |
| sofa | 0.4629 |
| train | 0.8300 |
| tvmonitor | 0.6919 |

表现较强的类别包括：

```text
background, bus, bird, cat, aeroplane, train, person, car
```

仍然困难的类别包括：

```text
chair, bicycle, sofa, diningtable, pottedplant
```

与此前 v3 baseline 相比，`diningtable`、`pottedplant`、`chair` 和 `boat` 有较明显改善，说明 OS4 low-level feature 对部分结构复杂类别确实有帮助。但 `sheep`、`motorbike` 等类别下降，最终导致总体 mIoU 几乎持平。

---

## 实验结论

1. **DeepLabv3+ decoder 实现成功。** 模型正确完成 ASPP 高层语义与 OS4 低层空间特征融合。
2. **总体 mIoU 没有显著超过 v3。** `0.7210 → 0.7214` 基本属于单次训练波动范围。
3. **Mean Accuracy 明显提高。** `0.8163 → 0.8304`，说明 decoder 对部分较难类别有积极作用。
4. **低层特征融合不是普遍增益。** 部分类别提升明显，但也有类别退步，收益被抵消。
5. **Stage 2 对 v3+ 有价值，但当前切换过于激进。** 解冻后最终超过 Stage 1，但前期出现明显性能冲击。
6. **当前 checkpoint 选择方式仍有偏差。** 每轮只用固定前 300 张验证图选择 best checkpoint，不能保证选中完整 1,449 张验证集上的真正最佳模型。

---

## 已知限制

- Backbone 使用 ResNet-34，而不是论文强配置中的 Xception / ResNet-101。
- ASPP rates 使用 `(3, 6, 9)`，用于保持与前一版 v3 的控制变量一致。
- 当前 ASPP 和 decoder 使用普通 convolution，没有实现 atrous separable convolution。
- 当前模型以 OS8 训练和推理。
- 每轮 proxy validation 只评估前 300 张图。
- 当前 neck 使用固定 `scale_factor=2` 对齐 OS8 与 OS4，因此输入应保证为 8 的倍数。
- 尚未计算 Boundary IoU、FLOPs、FPS 和显存占用。

---

## 下一步

### 1. 改进 Stage 2

建议下一次使用：

```text
Stage 2 epochs:       15 ~ 20
neck/head LR:         3e-5
backbone-high LR:     8e-6
backbone-low LR:      1e-6
warmup:               2 ~ 3 epochs
```

并从 Stage 1 最佳 checkpoint 开始 Stage 2，而不是直接从 Stage 1 最后一轮继续。

### 2. 改进 checkpoint 选择

推荐：

```text
每轮：300张 proxy validation
每5轮：完整 1,449 张 validation
每个 stage 结束：完整 validation
保留 proxy 最好的多个候选 checkpoint
```

最后对候选 checkpoint 全量验证后再决定最终 best model。

### 3. 增加边界指标

DeepLabv3+ 的主要目标是恢复边界，因此后续应增加：

- Boundary IoU
- trimap IoU
- 固定样本的 v3 / v3+ 边界可视化对比

重点观察：

```text
chair, bicycle, pottedplant, diningtable, person, motorbike
```

### 4. 后续学习路线

完成更稳健的 v3+ 训练与边界评估后，可以结束 DeepLab 系列，进入：

```text
SegFormer
```

用于比较 CNN atrous-convolution 系列与 hierarchical Transformer segmentation encoder。

---

## 文件结构

```text
model/
├── backbone.py       # OS8 dilated ResNet-18/34
├── neckV3.py         # ASPP encoder + 1×1 projection
├── headV3.py         # low-level decoder
└── deeplab.py        # full DeepLabv3+ wiring

trainV3.py            # v3+ training entry
train.py              # shared training/evaluation utilities
training_log.json     # current experiment log
```

## 输出文件

训练目录包含：

```text
best.pt
training_log.json
loss_curve.png
miou_curve.png
```
