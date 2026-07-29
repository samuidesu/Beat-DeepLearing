# DeepLab-v3 on PASCAL VOC 2012

本项目在前面的 DeepLab-v1 / DeepLab-v2 实验基础上，实现了一个 **ResNet-34 + DeepLab-v3-style ASPP** 语义分割模型，并进一步测试了 `block4/layer4 Multi-Grid` 变体。

当前实验系列的核心目标不是严格复现论文的最终系统，而是在相同数据、相同 ResNet-34 backbone 和相近训练协议下，逐步比较：

```text
v1: 单一 LargeFOV context
v2: 多个 atrous branches，分类 logits 求和
v3: 多个 feature branches + image pooling + concat + learnable fusion
```

标准 v3 完整 VOC 2012 validation 结果：

```text
mIoU       0.7210
pixel_acc  0.9360
mean_acc   0.8163
```

Multi-Grid 变体结果：

```text
mIoU       0.7173
pixel_acc  0.9347
mean_acc   0.8148
```

> 当前最佳模型仍是标准 v3，完整验证集 `mIoU=0.7210`。激进的 ResNet-34 Multi-Grid `(4,8,16)` 没有继续提升结果，反而下降了 `0.0037` mIoU。

---

## 1. 数据集与实验设置

| 项目 | 配置 |
|---|---:|
| 训练集 | PASCAL VOC 2012 + SBD train_noval |
| 训练图像 | 7087 |
| 训练 batches / epoch | 442 |
| 验证集 | PASCAL VOC 2012 val |
| 验证图像 | 1449 |
| 类别数 | 21 |
| ignore index | 255 |
| crop size | 480 |
| random scale | 0.5 – 2.0 |
| batch size | 16 |
| random seed | 42 |
| device | cuda |

---

## 2. 标准 DeepLab-v3 结构

### 2.1 Dilated ResNet-34 backbone

输入图像经过 ImageNet 预训练 ResNet-34。

标准 v3 baseline 保持此前的 progressive dilation：

```text
layer1: output stride 4
layer2: output stride 8

layer3:
block 0      dilation 1
blocks 1–5   dilation 2
output stride 8

layer4:
block 0      dilation 2
blocks 1–2   dilation 4
output stride 8
```

最终：

```text
c5 = [B, 512, H/8, W/8]
```

### 2.2 v3 ASPP neck

配置：

```text
ASPP rates = (3, 6, 9)
hidden channels = 256
```

结构：

```text
c5
├─ 1×1 conv branch                         -> 256
├─ 3×3 atrous conv, dilation=3             -> 256
├─ 3×3 atrous conv, dilation=6             -> 256
├─ 3×3 atrous conv, dilation=9             -> 256
└─ global average pooling + 1×1 projection -> 256
        ↓
      concat
        ↓
[B, 1280, H/8, W/8]
```

与 v2 的关键区别：

```text
v2:
每个分支直接输出类别 logits
→ 固定相加

v3:
每个分支输出高维 feature
→ concat 保留各分支身份
→ fusion head 学习各尺度如何组合
```

### 2.3 Fusion head

```text
concat feature [B, 1280, H/8, W/8]
→ 1×1 fuse: 1280 → 256
→ BatchNorm + ReLU + Dropout
→ 1×1 classifier: 256 → 21
→ bilinear resize to [B, 21, H, W]
```

---

## 3. 标准 v3 训练过程

标准 v3 使用两阶段训练：

### Stage 1

冻结：

```text
stem
layer1
layer2
```

训练：

```text
layer3
layer4
ASPP-v3 neck
fusion head
```

### Stage 2

解冻完整 backbone，并使用分层学习率进行微调。

训练曲线显示，Stage 1 已经达到全程最优；Stage 2 开始后出现明显扰动，并且后续没有恢复到 Stage 1 的最佳水平。

### 3.1 Loss curve

![DeepLab-v3 loss](assets/v3_loss_curve.png)

### 3.2 Validation metrics

![DeepLab-v3 metrics](assets/v3_metrics_curve.png)

关键 epoch：

| Epoch | Stage | Train loss | Val loss | Proxy mIoU | Pixel acc |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.6671 | 0.3464 | 0.5699 | 0.8938 |
| 8 | 1 | 0.3649 | 0.3205 | 0.6067 | 0.8988 |
| 16 | 1 | 0.2685 | 0.2518 | 0.6705 | 0.9225 |
| 20 | 1 | 0.2161 | 0.2766 | 0.6546 | 0.9173 |
| 27 | 1 | 0.1488 | 0.2051 | 0.7264 | 0.9369 |
| 32 | 1 | 0.1372 | 0.2058 | 0.7222 | 0.9372 |
| 33 | 2 | 0.1834 | 0.2391 | 0.6820 | 0.9270 |
| 40 | 2 | 0.1595 | 0.2316 | 0.6875 | 0.9287 |
| 48 | 2 | 0.1328 | 0.2323 | 0.7033 | 0.9329 |
| 56 | 2 | 0.1129 | 0.2169 | 0.7174 | 0.9371 |
| 62 | 2 | 0.1069 | 0.2196 | 0.7156 | 0.9352 |

最佳 proxy checkpoint：

```text
mIoU = 0.7264 @ epoch 27
```

该 checkpoint 在完整 1449 张验证图像上评估为：

```text
mIoU = 0.7210
```

### 3.3 Stage 2 结论

Stage 2 没有提高最终结果：

```text
Stage 1 best proxy = 0.7264
Stage 2 best proxy = 0.7174
```

阶段切换时同时发生：

- 学习率重新升高；
- optimizer state 重置；
- stem/layer1/layer2 解冻；
- backbone BatchNorm 状态改变；
- 低层 feature distribution 开始变化。

因此当前 v3 最合理的正式 checkpoint 是 Stage 1 best，而不是训练最后一轮。

---

## 4. 标准 v3 最终结果

### 4.1 每类 IoU

| Class | IoU |
|---|---:|
| background | 0.9311 |
| aeroplane | 0.8397 |
| bicycle | 0.4025 |
| bird | 0.8411 |
| boat | 0.6448 |
| bottle | 0.7342 |
| bus | 0.9320 |
| car | 0.8171 |
| cat | 0.8624 |
| chair | 0.3507 |
| cow | 0.7510 |
| diningtable | 0.5076 |
| dog | 0.7984 |
| horse | 0.7458 |
| motorbike | 0.8160 |
| person | 0.8229 |
| pottedplant | 0.5134 |
| sheep | 0.8528 |
| sofa | 0.4753 |
| train | 0.8131 |
| tvmonitor | 0.6891 |

### 4.2 汇总指标

```text
mIoU       0.7210
pixel_acc  0.9360
mean_acc   0.8163
```

---

## 5. 与 v1 / v2 对比

| 模型 | Context / fusion | mIoU | Pixel acc | Mean acc |
|---|---|---:|---:|---:|
| DeepLab-v1-style | 单分支 LargeFOV，rate=12 | 0.7135 | 0.9350 | 0.8048 |
| DeepLab-v2-style | 多分支分类 logits 固定求和 | 0.7098 | 0.9338 | 0.8041 |
| **DeepLab-v3-style** | **feature concat + fusion + image pooling** | **0.7210** | **0.9360** | **0.8163** |

相对 v1：

```text
mIoU: 0.7135 → 0.7210
Delta: +0.0075
```

相对 v2：

```text
mIoU: 0.7098 → 0.7210
Delta: +0.0112
```

这说明：

> 在当前 ResNet-34 OS8 backbone 上，v3 的 feature-level多尺度融合和 image-level context 确实优于 v2 的固定 logits sum，但提升幅度仍然有限。

---

## 6. Multi-Grid 消融实验

### 6.1 修改内容

Multi-Grid 版本使用：

```text
layer3:
所有 6 个 BasicBlock 使用 dilation=2

layer4:
Multi-Grid unit rates = (1,2,4)
base dilation = 4
actual block dilations = (4,8,16)
```

在 ResNet-34 BasicBlock 中，每个 block 有两个 `3×3` 卷积；当前实现将该 block 的 Multi-Grid dilation 同时应用到两个卷积。

因此最后一个 block 实际为：

```text
3×3 dilation=16
→ 3×3 dilation=16
```

### 6.2 训练曲线

![DeepLab-v3 Multi-Grid loss](assets/v3_multigrid_loss_curve.png)

![DeepLab-v3 Multi-Grid metrics](assets/v3_multigrid_metrics_curve.png)

关键 epoch：

| Epoch | Train loss | Val loss | Proxy mIoU | Pixel acc |
|---:|---:|---:|---:|---:|
| 1 | 0.6681 | 0.3924 | 0.4857 | 0.8748 |
| 8 | 0.3450 | 0.3227 | 0.5941 | 0.8971 |
| 16 | 0.2342 | 0.2544 | 0.6739 | 0.9197 |
| 20 | 0.1853 | 0.2462 | 0.6856 | 0.9248 |
| 25 | 0.1507 | 0.2293 | 0.6826 | 0.9262 |
| 27 | 0.1426 | 0.2272 | 0.6911 | 0.9288 |
| 28 | 0.1404 | 0.2293 | 0.6895 | 0.9285 |

最佳 proxy：

```text
mIoU = 0.6911 @ epoch 27
```

完整验证集：

```text
mIoU       0.7173
pixel_acc  0.9347
mean_acc   0.8148
```

### 6.3 与标准 v3 对比

| 模型 | layer3 block dilation | layer4 block dilation | mIoU |
|---|---|---|---:|
| 标准 v3 | `(1,2,2,2,2,2)` | `(2,4,4)` | **0.7210** |
| v3 + Multi-Grid | `(2,2,2,2,2,2)` | `(4,8,16)` | 0.7173 |

变化：

```text
mIoU:      0.7210 → 0.7173  (-0.0037)
pixel_acc: 0.9360 → 0.9347  (-0.0013)
mean_acc:  0.8163 → 0.8148  (-0.0015)
```

当前结果说明：

> 激进的 ResNet-34 Multi-Grid `(4,8,16)` 没有提高总体性能。

但这不是严格的单变量 Multi-Grid 消融，因为同时改变了：

1. layer3 首个 block：`dilation 1 → 2`；
2. layer4：`(2,4,4) → (4,8,16)`；
3. Stage 1 训练长度：标准 v3 为32轮，MG版本为28轮；
4. cosine scheduler 的衰减速度随训练轮数一起改变。

因此该实验应记录为：

> `all-layer3-d2 + aggressive layer4 Multi-Grid` 没有优于标准 v3。

---

## 7. Multi-Grid 每类变化

### 7.1 提升类别

| Class | Standard v3 | Multi-Grid | Delta |
|---|---:|---:|---:|
| chair | 0.3507 | 0.3816 | +0.0309 |
| aeroplane | 0.8397 | 0.8628 | +0.0231 |
| bird | 0.8411 | 0.8511 | +0.0100 |
| boat | 0.6448 | 0.6544 | +0.0096 |
| car | 0.8171 | 0.8247 | +0.0076 |
| train | 0.8131 | 0.8204 | +0.0073 |
| bottle | 0.7342 | 0.7414 | +0.0072 |
| horse | 0.7458 | 0.7513 | +0.0055 |
| cow | 0.7510 | 0.7517 | +0.0007 |

### 7.2 下降类别

| Class | Standard v3 | Multi-Grid | Delta |
|---|---:|---:|---:|
| sheep | 0.8528 | 0.8125 | -0.0403 |
| sofa | 0.4753 | 0.4430 | -0.0323 |
| tvmonitor | 0.6891 | 0.6671 | -0.0220 |
| motorbike | 0.8160 | 0.7991 | -0.0169 |
| bicycle | 0.4025 | 0.3861 | -0.0164 |
| dog | 0.7984 | 0.7839 | -0.0145 |
| diningtable | 0.5076 | 0.4933 | -0.0143 |
| person | 0.8229 | 0.8127 | -0.0102 |
| pottedplant | 0.5134 | 0.5077 | -0.0057 |
| bus | 0.9320 | 0.9271 | -0.0049 |
| cat | 0.8624 | 0.8607 | -0.0017 |
| background | 0.9311 | 0.9306 | -0.0005 |

Multi-Grid 改变了类别间的性能分配：

- `chair`、`aeroplane` 等类别有所提高；
- `sheep`、`sofa`、`tvmonitor`、`motorbike` 等类别下降；
- 总收益与损失相抵后，mIoU 小幅下降。

---

## 8. 结果分析

### 8.1 为什么 v3 只比 v1/v2 小幅提高

主要原因：

1. backbone 本身已经是 OS8 dilated ResNet，输入 ASPP 的 `c5` 已拥有较大上下文；
2. v3 只升级 context neck，没有引入低层 decoder；
3. 最终仍是 OS8 logits 直接双线性上采样；
4. `chair`、`bicycle`、`pottedplant` 等类别仍受限于细结构与边界恢复；
5. ResNet-34、7087张训练图和当前训练协议限制了绝对上限。

### 8.2 为什么 Multi-Grid 没有继续提高

可能原因：

1. ResNet-34 layer4 只有3个 BasicBlock，结构比论文 ResNet-101 更短；
2. BasicBlock 中有两个空间 `3×3`，当前对两个卷积都使用相同大 dilation；
3. 最后连续两个 `dilation=16` 过于稀疏；
4. layer3 全部改成 dilation=2，失去首个 d1 block 的密集局部混合；
5. v3 ASPP 本身已经提供多尺度与全局上下文，backbone 中继续激进扩大 dilation 的边际收益有限；
6. 28轮 cosine schedule 比32轮更早衰减到接近零学习率。

---

## 9. 下一步建议

### 9.1 保留标准 v3 作为正式 baseline

当前正式最佳模型：

```text
DeepLab-v3-style
ResNet34
OS8
ASPP rates=(3,6,9)
mIoU=0.7210
```

Multi-Grid 版本应作为负面消融记录，而不是替换 baseline。

### 9.2 做更干净的 Multi-Grid 对照

保持 layer3 不变：

```text
layer3 = (1,2,2,2,2,2)
```

只比较 layer4：

```text
uniform:       (4,4,4)
mild MG:       (4,8,4)
aggressive MG: (4,8,16)
```

并统一：

```text
epochs-stage1 = 32
epochs-stage2 = 0
```

这样才能区分：

- Multi-Grid 本身是否有效；
- 还是最后的 dilation=16 对 ResNet-34 过于激进。

### 9.3 进入 DeepLab-v3+

如果目标是继续提高分割精度，更有价值的方向是加入低层 decoder：

```text
ASPP high-level feature
→ upsample to OS4

layer1 low-level feature
→ 1×1 projection

concat
→ 3×3 conv
→ 3×3 conv
→ classifier
→ upsample
```

这将直接针对当前仍然较弱的边界、细杆和小物体类别。

---

## 10. 最终结论

1. 标准 DeepLab-v3-style 达到 **0.7210 mIoU**，是当前 v1/v2/v3 系列最佳结果；
2. v3 相比 v1 提升 `+0.0075`，相比 v2 提升 `+0.0112`；
3. v3 的 concat、learnable fusion 和 image-level pooling 确实优于 v2 的固定 logits sum；
4. 当前 Stage 2 没有超过 Stage 1 best，因此正式结果应使用 Stage 1最佳 checkpoint；
5. 激进 Multi-Grid `(4,8,16)` 完整验证集只有 **0.7173 mIoU**；
6. Multi-Grid 相比标准 v3下降 `-0.0037`，没有带来收益；
7. 当前最合理的后续路线是保留标准 v3 baseline，随后实现 DeepLab-v3+ decoder。
