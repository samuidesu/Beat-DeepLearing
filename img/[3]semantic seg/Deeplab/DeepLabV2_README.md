# DeepLab-v2-style ASPP on PASCAL VOC 2012

本项目在已有的 **DeepLab-v1-style ResNet-34 baseline** 上实现了更接近原版 DeepLab v2 的 ASPP：

- 保持 dilated ResNet-34 backbone 不变；
- 保持最终 `output stride = 8`；
- 使用四个并行 atrous branches；
- 每个分支独立输出 21 类 score map；
- 对各分支 logits 做逐元素求和；
- 最后通过双线性插值恢复到输入分辨率。

本次完整 VOC 2012 validation 评估结果：

```text
mIoU       0.7098
pixel_acc  0.9338
mean_acc   0.8041
```

与此前 DeepLab-v1-style LargeFOV 的 `0.7135 mIoU` 相比，本次结果低 `0.0037`；与第一版 feature-sum ASPP 的 `0.6976` 相比，则提升了 `0.0122`。

> 当前结论：原版式 logits-sum ASPP 明显优于中间 feature 直接求和，但在本实验配置下仍未显示出相对单分支 LargeFOV 的明确优势。

---

## 1. 模型结构

### 1.1 Dilated ResNet-34 backbone

原始 ResNet-34 的后两级会继续下采样：

```text
layer1 -> output stride 4
layer2 -> output stride 8
layer3 -> output stride 16
layer4 -> output stride 32
```

本项目取消 `layer3` 和 `layer4` 的 stride-2 下采样，并采用 progressive dilation：

| Stage | 第一个 BasicBlock | 后续 BasicBlock | 输出步长 |
|---|---:|---:|---:|
| layer3 | dilation 1 | dilation 2 | OS8 |
| layer4 | dilation 2 | dilation 4 | OS8 |

因此 backbone 输出：

```text
c5: [B, 512, H/8, W/8]
```

与 v1 实验相同，backbone 的 stride、dilation 和预训练权重策略均未改变。

### 1.2 DeepLab-v2-style ASPP

本次使用：

```text
ASPP rates = (3, 6, 9, 12)
```

每个分支结构：

```text
c5 [B, 512, H/8, W/8]
→ 3×3 atrous conv, 512 → 128
→ BatchNorm
→ ReLU
→ Dropout2d(0.1)
→ 1×1 classifier, 128 → 21
→ branch logits [B, 21, H/8, W/8]
```

四个分支输出：

```text
logits_3
logits_6
logits_9
logits_12
```

然后：

```text
logits = logits_3 + logits_6 + logits_9 + logits_12
→ bilinear interpolation
→ [B, 21, H, W]
```

这里求和的是具有固定类别含义的 logits，而不是任意的中间 feature，因此四个分支天然具有通道语义对齐。

---

## 2. 数据与训练设置

### 2.1 数据集

| 项目 | 设置 |
|---|---:|
| 训练集 | PASCAL VOC 2012 + SBD train_noval |
| 训练图像 | 7087 |
| 训练 batches / epoch | 442 |
| 验证集 | PASCAL VOC 2012 val |
| 验证图像 | 1449 |
| 类别数 | 21 |
| ignore index | 255 |

### 2.2 数据增强与运行配置

| 项目 | 设置 |
|---|---:|
| crop size | 480 |
| random scale | 0.5 – 2.0 |
| batch size | 16 |
| num workers | 4 |
| size divisor | 8 |
| random seed | 42 |
| device | cuda |

### 2.3 两阶段训练

本次共训练：

```text
Stage 1: 24 epochs
Stage 2: 20 epochs
Total:   44 epochs
```

#### Stage 1

冻结 `stem/layer1/layer2`，训练 `layer3/layer4/ASPP`。

| 参数组 | 初始 LR |
|---|---:|
| ASPP / classifier | 1e-03 |
| layer3 / layer4 | 1e-04 |

```text
Trainable parameters: 22.31M
```

#### Stage 2

解冻完整 backbone，使用分层学习率：

| 参数组 | 初始 LR |
|---|---:|
| ASPP / classifier | 1e-04 |
| layer3 / layer4 | 3e-05 |
| stem / layer1 / layer2 | 1e-05 |

```text
Trainable parameters: 23.66M
Weight decay: 0.001
```

每个 stage 均重新建立 optimizer，并使用独立的 cosine annealing schedule。

---

## 3. 训练曲线

### 3.1 Cross-entropy loss

![DeepLab-v2 loss](assets/loss_curve.png)

主要现象：

- Stage 1 train loss 从约 `0.645` 下降至 `0.140`；
- Stage 1 后期 val loss 稳定在约 `0.233～0.250`；
- Stage 2 开始时 train/val loss 同时上升；
- 后续 train loss 继续下降至约 `0.110`；
- val loss 最终仍停留在约 `0.233～0.245`，出现一定 generalization gap。

### 3.2 Validation mIoU 与 pixel accuracy

![DeepLab-v2 validation metrics](assets/metrics_curve.png)

关键节点：

| Epoch | Stage | Train loss | Val loss | Proxy mIoU | Pixel acc |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.6447 | 0.3378 | 0.5278 | 0.8890 |
| 8 | 1 | 0.3486 | 0.2990 | 0.6210 | 0.9063 |
| 16 | 1 | 0.2067 | 0.2419 | 0.6864 | 0.9264 |
| 20 | 1 | 0.1632 | 0.2353 | 0.6973 | 0.9287 |
| 24 | 1 | 0.1401 | 0.2330 | 0.6951 | 0.9298 |
| 25 | 2 | 0.1917 | 0.2643 | 0.6567 | 0.9183 |
| 27 | 2 | 0.1838 | 0.2372 | 0.6864 | 0.9276 |
| 34 | 2 | 0.1417 | 0.2371 | 0.6935 | 0.9309 |
| 41 | 2 | 0.1173 | 0.2330 | 0.6986 | 0.9300 |
| 44 | 2 | 0.1102 | 0.2340 | 0.6957 | 0.9303 |

Stage 1 最佳 proxy：

```text
mIoU = 0.6973 @ epoch 20
```

Stage 2 最佳 proxy：

```text
mIoU = 0.6986 @ epoch 41
```

最终保存的最佳 checkpoint：

```text
best proxy mIoU = 0.6986 @ epoch 41
```

该 checkpoint 在全部 1449 张验证图像上重新计算后得到：

```text
full-val mIoU = 0.7098
```

---

## 4. 最终评估结果

### 4.1 每类 IoU

| Class | IoU |
|---|---:|
| chair | 0.3505 |
| bicycle | 0.3964 |
| sofa | 0.4179 |
| diningtable | 0.4566 |
| pottedplant | 0.5878 |
| boat | 0.6318 |
| horse | 0.7003 |
| tvmonitor | 0.7069 |
| cow | 0.7177 |
| bottle | 0.7370 |
| motorbike | 0.7691 |
| train | 0.7903 |
| dog | 0.7970 |
| car | 0.8162 |
| sheep | 0.8205 |
| person | 0.8242 |
| aeroplane | 0.8479 |
| cat | 0.8501 |
| bird | 0.8523 |
| bus | 0.9056 |
| background | 0.9307 |

### 4.2 汇总指标

```text
mIoU       0.7098
pixel_acc  0.9338
mean_acc   0.8041
```

---

## 5. 三个 DeepLab 实验对比

| 模型 | ASPP / context 融合方式 | mIoU | pixel acc | mean acc |
|---|---|---:|---:|---:|
| DeepLab-v1-style | 单分支 LargeFOV，rate=12 | **0.7135** | **0.9350** | **0.8048** |
| v2-style experiment 1 | 四分支中间 feature sum | 0.6976 | 0.9307 | 0.8086 |
| **v2-style experiment 2** | **四分支独立分类，logits sum** | **0.7098** | **0.9338** | **0.8041** |

### 5.1 相对第一版 feature-sum ASPP

```text
mIoU: 0.6976 → 0.7098
提升: +0.0122
```

这说明多尺度分支的融合位置非常重要。第一版直接相加 128 维中间 feature，要求不同 dilation 分支自行学习完全一致的通道语义；新版先把每个分支变成21类 score map，再进行求和，使所有通道天然对齐。

### 5.2 相对 DeepLab-v1-style LargeFOV

```text
mIoU:      0.7135 → 0.7098  (-0.0037)
pixel_acc: 0.9350 → 0.9338  (-0.0012)
mean_acc:  0.8048 → 0.8041  (-0.0007)
```

差距只有 `0.0037 mIoU`。在单次训练、单个随机种子的条件下，更适合解释为：

> 两者性能处于相近水平，当前实验没有观察到 ASPP 相对 LargeFOV 的稳定优势。

---

## 6. 与 v1 的每类结果对比

### 6.1 提升类别

| Class | v1 | v2 logits-sum | Delta |
|---|---:|---:|---:|
| bus | 0.8760 | 0.9056 | +0.0296 |
| sheep | 0.7921 | 0.8205 | +0.0284 |
| pottedplant | 0.5726 | 0.5878 | +0.0152 |
| car | 0.8053 | 0.8162 | +0.0109 |
| bird | 0.8425 | 0.8523 | +0.0098 |
| boat | 0.6232 | 0.6318 | +0.0086 |
| bottle | 0.7305 | 0.7370 | +0.0065 |
| sofa | 0.4138 | 0.4179 | +0.0041 |
| train | 0.7876 | 0.7903 | +0.0027 |

### 6.2 下降类别

| Class | v1 | v2 logits-sum | Delta |
|---|---:|---:|---:|
| diningtable | 0.4892 | 0.4566 | -0.0326 |
| chair | 0.3808 | 0.3505 | -0.0303 |
| horse | 0.7297 | 0.7003 | -0.0294 |
| motorbike | 0.7879 | 0.7691 | -0.0188 |
| tvmonitor | 0.7240 | 0.7069 | -0.0171 |
| bicycle | 0.4114 | 0.3964 | -0.0150 |
| aeroplane | 0.8610 | 0.8479 | -0.0131 |
| cow | 0.7297 | 0.7177 | -0.0120 |
| person | 0.8343 | 0.8242 | -0.0101 |
| cat | 0.8599 | 0.8501 | -0.0098 |
| dog | 0.8022 | 0.7970 | -0.0052 |

本次结果不是“所有类别都没有变化”，而是：

```text
提高类别：9
下降类别：11
```

ASPP 对 `bus`、`sheep`、`pottedplant`、`car` 等类别产生了正向影响，但 `diningtable`、`chair`、`horse`、`motorbike`、`tvmonitor` 等类别出现下降，最终收益和损失基本抵消。

---

## 7. Stage 2 扰动分析

Stage 1 最后一轮：

```text
epoch 24
neck/head LR      = 4.28e-6
backbone-high LR  = 4.28e-7
train loss        = 0.1401
val loss          = 0.2330
mIoU              = 0.6951
```

Stage 2 第一轮：

```text
epoch 25
neck/head LR      = 1.00e-4
backbone-high LR  = 3.00e-5
backbone-low LR   = 1.00e-5
train loss        = 0.1917
val loss          = 0.2643
mIoU              = 0.6567
```

阶段切换同时发生：

1. neck/head LR 增大约 23 倍；
2. backbone-high LR 增大约 70 倍；
3. 创建新的 optimizer，Stage 1 的动量状态被重置；
4. 解冻 stem/layer1/layer2，使低层特征开始变化；
5. 冻结层的 BatchNorm 从固定状态重新进入训练状态；
6. Stage 2 从 Stage 1 最后一轮继续，而不是重新加载 Stage 1 最佳 checkpoint。

因此 Stage 2 的短暂下降属于优化状态不连续造成的扰动，而不是模型永久退化。

本次 Stage 2 的最终净收益较小：

```text
Stage 1 最佳 proxy = 0.6973
Stage 2 最佳 proxy = 0.6986
提升                 = +0.0013
```

说明大部分性能已经在 Stage 1 中获得，Stage 2 更接近轻量微调。

---

## 8. 实验结论

### 8.1 logits-sum 修复了 feature-sum 的主要问题

从 `0.6976 → 0.7098` 可以看出，让每个 ASPP 分支先完成类别预测，再对类别 logits 求和，比直接相加无固定语义的中间 feature 更合理。

### 8.2 ASPP 在当前 backbone 上没有明确超过 LargeFOV

当前 ResNet-34 backbone 已经让 `layer3/layer4` 在 OS8 上使用 dilation 1/2/4，进入 ASPP 的 `c5` 本身已包含较大的上下文。因此多个 ASPP 分支可能学到较多重复信息，新增的多尺度上下文只有有限边际收益。

### 8.3 v2 改变了类别分布，但没有提高总体上限

ASPP 对部分大目标或场景相关类别有效，但没有一致地改善所有尺度，也没有解决 `chair`、`bicycle` 等细结构类别的问题。

> 高层多尺度上下文不能替代低层边界和细节信息。

### 8.4 本次 v2 实验已经完成主要学习目标

该实验已经回答了两个问题：

1. 中间 feature sum 不适合作为本项目的 ASPP 融合方式；
2. 原版式 logits sum 可以恢复性能，但仍未明显超过单分支 LargeFOV。

因此没有必要继续围绕 v2 进行大量调参。

---

## 9. 下一步：DeepLab v3

下一步保留当前 dilated ResNet-34 backbone，只将 ASPP 更新为 v3-style enhanced ASPP：

```text
c5
├─ 1×1 branch
├─ atrous branch
├─ atrous branch
├─ atrous branch
└─ image-level pooling branch
        ↓
      concat
        ↓
  1×1 projection
        ↓
      dropout
        ↓
    classifier
```

v3 相对当前 v2 的核心变化：

- 不在每个分支中提前分类；
- 保留各尺度的高维 feature；
- 使用 concat 保留分支身份；
- 使用 1×1 projection 学习跨尺度融合；
- 使用 image pooling 提供显式全局上下文。

第一版 v3 应继续保持 backbone、dataset、augmentation、loss、training protocol 和 output stride 不变，以便单独验证 enhanced ASPP 的贡献。

---

## 10. 最终总结

1. 当前 logits-sum ASPP 在完整 VOC val 上达到 **0.7098 mIoU**；
2. 相比第一版 feature-sum ASPP 提升 **1.22 个百分点**；
3. 相比 v1 LargeFOV 低 **0.37 个百分点**，属于相近水平；
4. 更接近原版的 ASPP 融合明显更合理，但没有在当前 backbone 上表现出稳定优势；
5. Stage 2 的 LR、optimizer、解冻和 BN 状态同时重启，造成明显扰动；
6. Stage 2 对 proxy mIoU 的净提升仅约 `+0.0013`；
7. 下一步应进入 DeepLab v3，重点验证 feature concat、可学习 projection 和 image-level context。
