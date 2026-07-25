# U-Net on PASCAL VOC 2012(手写复现)

分割系列的第三个模型:从零手写的 **U-Net**(ResNet 编码器版)。与 [FCN(add 融合)](../FCN/)、[FCN_concat](../FCN_concat/) 用**同一份数据(VOC2012 train + SBD train_noval = 7087 张)、同一个 ResNet-34 编码器、同一套两阶段训练协议、同一个训练循环骨架**——对照的变量是"解码器整体设计":FPN 式 neck 换成经典 U-Net 的展开路径(concat 跳连 + 学习式上采样 + 镜像通道),并把输出分辨率推到 **stride-2**,顺便终结系列里悬着的问题:细结构差,到底是不是分辨率的锅?

- 结果:**mIoU 0.6894**(pixel_acc 0.9299 / mean_acc 0.7863)——与 add 版 0.6910、concat 版 0.6932 **同档打平**(三者极差 0.0038,单次训练噪声量级),而 decoder+head 参数只有 **3.14M**(add 6.12M / concat 4.13M)
- **stride-2 没有换来细结构提升**(bicycle 反而是三者最低)——"分辨率瓶颈"假设第三次落空,基本可以结案
- 背景知识(任务定义、与检测项目的逐点对照、数据协议、实验 1/2 的完整分析)见 [FCN 的 README](../FCN/README.md),不重复写

---

## 设计:与 FCN 系 neck 的三点差异(`model/`)

```
image [B,3,H,W]  (训练 480 随机裁剪;评估原尺寸 pad 到 /32)
  └─ encoder(ResNet-34,ImageNet 预训练,5 个 tap)
       c1 [64,H/2](stem 的 conv1+bn+relu 输出,maxpool 之前)   <- 比 FCN 系多 tap 的一层
       c2 [64,H/4]   c3 [128,H/8]   c4 [256,H/16]   c5 [512,H/32]
  └─ decoder(4 × UpBlock,自底向上爬回 stride-2)
       UpBlock = ConvTranspose2d(k=2,s=2) 上采样 → concat 同分辨率跳连 → DoubleConv(3×3+BN+ReLU ×2)
       通道镜像收缩:512 → 256 → 128 → 64 → 64,输出 p1 [B,64,H/2,W/2]
  └─ head(经典 OutConv:1×1 分类器 + 2× bilinear)  ──►  logits [B,21,H,W]
```

| | FCN 系 neck(../FCN、../FCN_concat) | U-Net decoder(本项目) |
|---|---|---|
| 跳连融合 | add(FCN)/ concat(FCN_concat) | **concat**(concat vs add 的单变量对照 FCN_concat 已做过:打平) |
| 上采样 | nearest(无参数) | **ConvTranspose2d(k=2,s=2),学习式**;k=s=2 无重叠,不会棋盘伪影 |
| 通道方案 | 恒定 256(add)/ 收窄到 128(concat) | **镜像编码器,一路减半到 64** |
| 最细层级 | stride-4(p2) | **stride-2(p1)**,末端只需 2× 上采样 |
| head | ConvSet 再融合一轮 + 1×1 | **只有 1×1**——每个 UpBlock 已自带 DoubleConv,head 无事可做 |
| decoder+head 参数 | 6.12M / 4.13M | **3.14M**(decoder 3.14M + head 0.001M;总参数 24.43M) |

两个"为什么":

- **C1 为什么在 maxpool 之前 tap**:maxpool 之后就是 stride-4(那里已有 C2);要爬回 stride-2,唯一的 stride-2 特征就是 stem 的 conv1 输出。它只过了一层卷积,语义极浅——但恢复边界要的恰恰是这种"where"信息,"what"由深层沿解码路径带上来。
- **与经典 U-Net 的差别**:编码器不从头训,直接用 ImageNet 预训练 ResNet-34(与系列其他项目同一预训练、同一冻结/解冻协议,保证对照干净);ResNet 的 layer4 就是 bottleneck,不再另加底块;其余(concat、双卷积、OutConv)按原味。

---

## 结果

一次训练:26 + 34 epoch(与 concat 版完全一致;add 版是 26+36,proxy 早已走平,差异可忽略),数据/协议同系列。best proxy 0.6708 @ epoch 59,全量评估 **mIoU 0.6894 / pixel_acc 0.9299 / mean_acc 0.7863**。三模型同台:

| | add(FCN 实验 2) | concat(FCN_concat) | U-Net(本项目) |
|---|---|---|---|
| 最细层级 / 上采样 | stride-4 / nearest | stride-4 / nearest | **stride-2 / 转置卷积** |
| decoder+head 参数 | 6.12M | 4.13M | **3.14M** |
| 每 epoch 耗时(s1 / s2) | ~124s / ~165s | ~95s / ~130s | ~98s / ~136s |
| stage1(冻结)proxy 峰值 | 0.582 | 0.566 | 0.560 |
| best proxy | 0.684 | 0.674 | 0.671 |
| **mIoU(全量)** | 0.6910 | **0.6932** | 0.6894 |
| pixel_acc | 0.9295 | 0.9292 | **0.9299** |
| mean_acc | 0.7811 | **0.7875** | 0.7863 |

三个结论:

**1. 三种解码器设计在 0.69 全部打平——解码器不是当前条件下的瓶颈。** 极差 0.0038,在单次训练的噪声量级内,排不出名次。U-Net 用**最少的 decoder 参数(3.1M)+ 最高的输出分辨率**拿到同档成绩,每 epoch 耗时与 concat 版相当、比 add 版快 ~20%。这个系列真正分出差距的变量至今只有两个:数据量(+0.11)和解冻 backbone(小数据 ×2.8 / 足数据 +0.11);换解码器三次,合计变化 < 0.004。

**2. stride-2 没有拯救细结构和家具——"分辨率瓶颈"假设第三次落空,结案。** 每类对照(add / concat / U-Net):

| 类 | add | concat | U-Net(stride-2) |
|---|---|---|---|
| chair | 0.3080 | 0.3125 | **0.3409**(三者最好,+0.03) |
| bicycle | 0.3943 | 0.3917 | **0.3811**(三者最低——分辨率最高反而最差) |
| sofa | 0.3534 | 0.4252 | 0.3969(居中) |
| diningtable | 0.4661 | 0.4506 | 0.4439 |
| pottedplant | 0.5538 | 0.5235 | 0.5128 |

从实验 1 → 实验 2 → 本项目,输出分辨率 stride-8 → 4 → 2 翻了两番,细结构类在 ±0.03 里抖、毫无单调改善;家具组(chair/sofa/diningtable)仍是全场最差。证据链已经三环:细结构/家具差的根因是**语义混淆与标注边界模糊**(chair vs sofa vs diningtable 在 VOC 里本来就定义含糊、遮挡镂空严重),不是解码分辨率。要再涨,方向是语义容量——更大 encoder、更强上下文(空洞卷积扩感受野)、更多数据——而不是更细的输出网格。

**3. 第三次复现:数据足够时,解冻 encoder 只值 +0.11。** stage1 冻结 0.560 → stage2 解冻 0.671,与 add(0.582→0.684)、concat(0.566→0.674)同一模式。顺带一个小观察:stage1 冻结成绩恰好随 decoder 参数量单调排序(6.1M→0.582,4.1M→0.566,3.1M→0.560)——方向符合"冻结时全靠 decoder 容量兜底",但差距只有 0.02,不足以下结论,记录备查。

<details>
<summary>完整 21 类 IoU(eval.py 原始输出)</summary>

```
class              IoU
----------------------
chair           0.3409
bicycle         0.3811
sofa            0.3969
diningtable     0.4439
pottedplant     0.5128
boat            0.6140
horse           0.6858
tvmonitor       0.6941
bottle          0.6972
cow             0.7031
motorbike       0.7620
train           0.7624
sheep           0.7747
dog             0.7784
car             0.8092
bird            0.8211
person          0.8234
cat             0.8345
aeroplane       0.8496
bus             0.8649
background      0.9285
----------------------
mIoU            0.6894
pixel_acc       0.9299
mean_acc        0.7863
```

</details>

---

## 快速开始

所有命令在本目录(`U-Net/`)下运行。

```bash
# 1. 数据:VOC2012(~2 GB)与 SBD(~1.4 GB,USE_SBD=True 默认开,需 scipy)缺了会自动下载;
#    若 YOLO3/FCOS 项目已有 VOC2012 会自动复用(见 config.DATA_ROOT),SBD 放在 DATA_ROOT/sbd。
python dataset/voc.py --download            # 手动预下载(可跳过,train.py 会自动下)

# 2. 训练(本 README 结果的复现命令;config 默认 20+60 epoch)
python train.py --epochs-stage1 26 --epochs-stage2 34

# 3. 全量评估(VOC2012 val 1449 张,输出 mIoU / pixel_acc / 每类 IoU,最差在前)
python eval.py
python eval.py --max-batches 100            # 快速抽查

# 4. 可视化
python segment/segment.py --voc-random 10   # overlay / pred / gt 三件套 -> segment/results/
python detect/detect.py --n 10              # 每图 <id>_pred.png + <id>_gt.png -> detect/results/
```

每个模块文件都带自测入口(离线,不需要数据集):

```bash
python model/encoder.py ; python model/decoder.py ; python model/head.py ; python model/unet.py
```

依赖同 ../FCN:`pip install torch torchvision numpy matplotlib tqdm pillow scipy`(scipy 供 SBD 读 .mat)。

文件结构与 ../FCN 一致(见其 README),差异只在 `model/`:`{backbone,neck,head,fcn}.py` 换成 `{encoder,decoder,head,unet}.py`;config 要点相同(`BACKBONE = "resnet34"`、`USE_SBD = True`、默认 epoch 20+60,本次 CLI 覆盖为 26+34)。

## 待办 / 实验计划

- [x] 与 FCN(add)/ FCN_concat 三方对照——**打平(极差 0.0038);"分辨率瓶颈"假设三连否定,见"结果"**
- [ ] 多种子重跑(×3),给三个解码器的 mIoU 置信区间,把"打平"从单次观察坐实成结论
- [ ] 混淆矩阵(`utils/metrics.py`)定量确认 chair/sofa/diningtable 组内互吞的像素流向
- [ ] 下一步涨点方向(语义容量,非分辨率):空洞卷积扩感受野(DeepLab 式)/ 更大 encoder(ResNet-50)——在本版(decoder 最便宜)上做
