# FCN_concat:FPN 融合用 concat 的对照变体

[FCN 主项目](../FCN/)的对照实验分支:代码与 FCN(stride-4 版,实验 2)完全同源,**唯一的设计差异在 neck 的自顶向下融合——add 换成 concat**(通道方案配套收窄,见下)。目的:把 "FPN 原文的 add" 与 "YOLOv3 route 层的 concat" 这两条融合路线,在同一任务、同一数据、同一训练协议下钉死对比。

- 训练/验证协议与 FCN 实验 2 完全一致:VOC2012 train + SBD train_noval = 7087 张训练;VOC2012 val 全量 1449 张、原尺寸评估
- 结果:**mIoU 0.6932**(pixel_acc 0.9292 / mean_acc 0.7875)——与 add 版的 0.6910 **打平**(+0.0022,噪声量级),但 neck+head 参数 6.1M → **4.1M**、每 epoch 快 ~20–25%
- 背景知识(FCN 思想、与检测项目的逐点对照、实验 1/实验 2 的完整分析)不重复写,见 [FCN 的 README](../FCN/README.md)

---

## 与 add 版的唯一差异(`model/neck.py`)

每步 top-down 合并从"逐元素相加"改成"沿通道拼接":

```
add 版(../FCN):                            concat 版(本项目):
m4 = lateral4(c4) + up(p5)                  m4 = concat(lateral4(c4), up(p5))
m3 = lateral3(c3) + up(p4)                  m3 = concat(lateral3(c3), up(p4))
m2 = lateral2(c2) + up(p3)                  m2 = concat(lateral2(c2), up(p3))

lateral / ConvSet / 输出全程 256ch          通道越往浅层越窄:lateral5/4 → 256,lateral3/2 → 128
p2 = [B, 256, H/4, W/4]                     smooth4=ConvSet(512→256) smooth3=ConvSet(384→128)
                                            smooth2=ConvSet(256→128)
                                            p2 = [B, 128, H/4, W/4]
```

- **为什么 concat 必须配收窄**:concat 让每个合并点的输入通道翻倍(256+256=512),若仍全程保持 256,越往浅层特征图越大,stride-4 上的 ConvSet 算力直接爆炸;所以金字塔越往下越窄(256 → 128),输出 p2 只有 128ch。add 天然保持通道数,才能全程 256。
- 这正是两条经典路线的分野:FPN 原文用 add(轻,信息做有损叠加),YOLOv3 用 concat(保留两路原始特征,让后面的 conv 自己学怎么混)。
- 代价对比:neck+head 参数 6.12M → **4.13M**(−33%);本机实测每 epoch stage1 ~124s → ~95s、stage2 ~165s → ~130s(省的主要是 stride-4 高分辨率上的 256ch ConvSet)。

其余一切与 ../FCN 相同:backbone(ResNet-34,tap c2–c5)、head(ConvSet + 1×1 分类器 + 4× bilinear 上采样,输入随 p2 变为 128ch)、逐像素 CE(ignore 255)、两阶段 finetune、混淆矩阵 mIoU。

---

## 结果

协议与 [FCN 实验 2](../FCN/README.md#结果) 一致,唯二区别:融合方式(本项目 concat)、stage2 epoch(34 vs 36,proxy 早已走平,影响可忽略)。best proxy 0.6742 @ epoch 51,全量评估:

| | concat(本项目) | add(FCN 实验 2) | Δ |
|---|---|---|---|
| **mIoU** | **0.6932** | 0.6910 | +0.0022 |
| pixel_acc | 0.9292 | 0.9295 | −0.0003 |
| mean_acc | 0.7875 | 0.7811 | +0.0064 |
| neck+head 参数 | **4.13M** | 6.12M | **−33%** |
| 每 epoch 耗时(s1 / s2) | ~95s / ~130s | ~124s / ~165s | 约 **−20–25%** |

三个结论:

**1. add vs concat 在精度上打平——差异在效率,不在 mIoU。** +0.0022 在单次训练的噪声量级内,不构成"concat 更准"的证据;但 concat 版用 2/3 的 neck+head 参数、约 3/4 的每 epoch 时间跑出了同样的精度。要说赢家,是 concat 的性价比,不是它的表达力。(严格说这不是纯算子对照:concat 配套了通道收窄,实际对比的是 "add + 全程 256" vs "concat + 收窄到 128" 两套完整方案——但这正是两条路线各自的标准打开方式。)

**2. 每类 IoU 基本原地互换,唯一显著的单类变化是 sofa +0.07。** sofa 0.3534 → 0.4252,把 add 版里"sofa 相对实验 1 倒退"补了回来(实验 1 为 0.4173);其余类在 ±0.03 内互有涨跌(pottedplant −0.03、diningtable −0.02、cow +0.02、boat +0.02)。家具组 chair 0.31 / sofa 0.43 / diningtable 0.45 仍是全场最差,bicycle 0.39 原地——**剩余短板与融合方式无关**,与 add 版结论一致(语义边界模糊 + 遮挡,不是分辨率或融合算子能解决的)。sofa 这 +0.07 更可能是家具组内像素分配的抖动而非 concat 的功劳,想坐实需要多种子重复。

**3. 复现了 FCN 实验 2 的修正结论:数据够时,解冻 backbone 只值 +0.11。** stage1 冻结 backbone 到 proxy 0.566,stage2 解冻到 0.674——与 add 版(0.582 → 0.684)同一模式,再次印证"解冻是 3 倍杠杆"只在 1464 张小数据时成立。

<details>
<summary>完整 21 类 IoU(eval.py 原始输出)</summary>

```
class              IoU
----------------------
chair           0.3125
bicycle         0.3917
sofa            0.4252
diningtable     0.4506
pottedplant     0.5235
boat            0.6293
tvmonitor       0.6735
bottle          0.7035
horse           0.7080
cow             0.7217
sheep           0.7670
motorbike       0.7716
train           0.7812
dog             0.7860
car             0.7944
person          0.8121
bird            0.8194
cat             0.8435
aeroplane       0.8533
bus             0.8631
background      0.9259
----------------------
mIoU            0.6932
pixel_acc       0.9292
mean_acc        0.7875
```

</details>

---

## 快速开始

所有命令在本目录(`FCN_concat/`)下运行。

```bash
# 1. 数据:VOC2012(~2 GB)与 SBD(~1.4 GB,USE_SBD=True 默认开,需 scipy)缺了会自动下载;
#    若 YOLO3/FCOS 项目已有 VOC2012 会自动复用(见 config.DATA_ROOT),SBD 放在 DATA_ROOT/sbd。
python dataset/voc.py --download            # 手动预下载(可跳过,train.py 会自动下)

# 2. 训练(本 README 结果的复现命令;config 默认 20+60 epoch)
python train.py --epochs-stage1 26 --epochs-stage2 34 --download

# 3. 全量评估(VOC2012 val 1449 张,输出 mIoU / pixel_acc / 每类 IoU,最差在前)
python eval.py
python eval.py --max-batches 100            # 快速抽查

# 4. 可视化
python segment/segment.py --voc-random 10   # overlay / pred / gt 三件套 -> segment/results/
python detect/detect.py --n 10              # 每图 <id>_pred.png + <id>_gt.png -> detect/results/
```

依赖同 ../FCN:`pip install torch torchvision numpy matplotlib tqdm pillow scipy`(scipy 供 SBD 读 .mat)。

文件结构与 ../FCN 完全一致(见其 README 的文件结构一节),唯一实质差异在 `model/neck.py`;config 要点也一致(`USE_SBD = True`,默认 epoch 20+60,本次 CLI 覆盖为 26+34)。

## 待办 / 实验计划

- [x] 与 add 版(FCN 实验 2)对照——**精度打平(+0.0022),效率 −33% 参数 / −20–25% 每 epoch 时间**,见"结果"
- [ ] 多种子重跑,确认 +0.0022 与 sofa +0.07 是否只是噪声
- [ ] FCN 待办里的公共消融(FCN-32s 对照、上采样方式)如果做,优先在本版上跑(更快)
