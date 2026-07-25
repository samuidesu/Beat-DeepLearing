# FCN on PASCAL VOC 2012(手写复现)

从零手写的类 **FCN**(Fully Convolutional Network)语义分割网络,在 PASCAL VOC 2012 分割集上训练。
与本仓库的 [YOLO3](../../[2]ObjectionDetection/YOLO3/PASCAL_VOC/) / [FCOS](../../[2]ObjectionDetection/FCOS/PASCAL_VOC/) 检测复现完全平行:**同一个 backbone(ResNet-18/34)、同一个 FPN neck、同一套两阶段训练协议、同一个训练循环骨架**,换掉的只有任务本身——从"预测框"变成"给每个像素分类"。目的:进入分割领域时,把"检测和分割到底差在哪"钉死在同一实验条件下。

- 训练集:VOC2012 Segmentation train(1464 张;实验 2 起开 SBD train_noval 增广 → 7087 张,为什么不是文献常说的 ~10.5k 见"实验 2")
- 验证集:VOC2012 Segmentation val(1449 张,标准协议,原始分辨率评估)
- 结果:实验 1(stride-8 输出,仅 VOC 1464 张)**mIoU 0.5813** → 实验 2(stride-4 输出 + SBD 增广)**mIoU 0.6910**(均为 VOC2012 val 全量 1449 张原尺寸评估;两次实验的完整分析与互相修正见"结果")

> 这是按 FCN *思想* 实现的(全卷积、逐像素分类、跳连融合多尺度、上采样输出)。原论文用 VGG16 + 逐层加 21 通道分数图再相加(FCN-8s);这里用 ResNet + FPN 在**特征层面**自顶向下融合再一次性预测——实验 1 融合到 stride-8(语义上等价 FCN-8s 的跳连),实验 2 多下探一级到 stride-4(相当于"FCN-4s"),表达力都比逐层加分数图更强。

---

## 结果

两次实验,同一评估协议(VOC2012 val 全量 1449 张、原始分辨率、best checkpoint):

| | 实验 1 | 实验 2 |
|---|---|---|
| 模型 | stride-8 输出(≈FCN-8s),ConvSet 5 卷积,neck+head 11.0M | **stride-4 输出(加 C2 tap,≈"FCN-4s")**,ConvSet 3 卷积、head 收窄 128ch,neck+head 6.1M |
| 训练数据 | VOC train 1464 张 | **+ SBD train_noval = 7087 张**(×4.8) |
| epoch(s1+s2) | 20 + 60 | 26 + 36 |
| **mIoU(全量)** | 0.5813 | **0.6910**(+0.1097) |
| pixel_acc / mean_acc | 0.9104 / 0.6763 | 0.9295 / 0.7811 |

实验 2 同时动了模型和数据两个变量,提升不能干净归因;证据倾向数据是大头,分析见实验 2 小节。

> 后续单变量对照:[FCN_concat](../FCN_concat/) 在实验 2 的基础上只把 neck 融合从 add 换成 concat(通道配套收窄)——mIoU 0.6932 与实验 2 打平,但 neck+head 参数 −33%、每 epoch 快 ~20–25%,详见其 README。
>
> 系列第三个模型 [U-Net](../U-Net/):解码器整体换成经典 U-Net 三件套(concat + 转置卷积上采样 + 镜像通道,爬到 stride-2)——mIoU 0.6894,三种解码器全部打平;输出分辨率 8→4→2 翻两番,细结构类毫无改善,"分辨率瓶颈"假设结案,详见其 README。

### 实验 1:stride-8 输出 + 仅 VOC 1464 张(mIoU 0.5813)

best checkpoint @ epoch 78:

| 指标 | 实验 1 | 备注 |
|---|---|---|
| mIoU | **0.5813** | 训练日志里逐 epoch 的 mIoU 是 300 张抽样 proxy(峰值 0.5569),全量评估略高 |
| pixel_acc | 0.9104 | 被 background(IoU 0.9223)主导,仅作 sanity check |
| mean_acc | 0.6763 | |

两个预期观察点都有了答案:

**1. "stage2 解冻 backbone 是最大杠杆"在分割上同样成立——解冻让 mIoU 翻了近 3 倍。**

| 阶段 | 训练部分 | mIoU(proxy) |
|---|---|---|
| Stage 1(20 ep) | 冻结 backbone,只训 neck+head | 0.196 |
| Stage 2(60 ep) | 全部解冻端到端 finetune | **0.557**(全量 0.5813) |

严格说两阶段 epoch 数不同(20 vs 60),但 stage1 后半段 mIoU 每个 epoch 只挪 0.01–0.02,而 stage2 从 0.18 一路爬到 0.56——差距不是再多训几个冻结 epoch 能补上的。解释与检测项目一致,但在分割上暴露得更彻底:逐像素分类要求特征**空间上**精确,ImageNet 分类预训练只回答"图里有什么"、不关心"在哪个像素",这个缺口冻结时靠 neck/head 补不回来。(实验 2 会修正这个结论的适用范围——它是小数据下的现象,见下。)

**2. 1464 张训练图确实不够,但短板不在预期的地方。** 每类 IoU 两端(完整表折叠在下面):

| 最差 5 类 | IoU | 最好 5 个前景类 | IoU |
|---|---|---|---|
| cow | **0.0298** | bird | 0.8178 |
| chair | 0.1070 | cat | 0.8136 |
| bicycle | 0.3060 | person | 0.7992 |
| horse | 0.3564 | bus | 0.7889 |
| pottedplant | 0.4001 | car | 0.7772 |

原本预判最差的是细结构类(bottle/bicycle/pottedplant,stride-8 分辨率极限),只对了一半:bicycle 0.31 / pottedplant 0.40 确实差,但 bottle 拿到 0.58 并不垫底。真正塌掉的是 cow 0.03 + horse 0.36,以及 chair 0.11 + sofa 0.42 + diningtable 0.46——**语义近邻(四足动物一组、家具一组)在互相吞并像素**,这是小数据下判别力不足,不是上采样分辨率不足。所以下一步优先 `USE_SBD = True` 补数据,而不是先换 DeepLab/U-Net 式解码。(cow 的像素具体流向哪些类,可用 `utils/metrics.py` 的混淆矩阵直接确认。)

<details>
<summary>完整 21 类 IoU(eval.py 原始输出)</summary>

```
class              IoU
----------------------
cow             0.0298
chair           0.1070
bicycle         0.3060
horse           0.3564
pottedplant     0.4001
sofa            0.4173
diningtable     0.4570
boat            0.5204
bottle          0.5765
sheep           0.6162
tvmonitor       0.6221
train           0.7001
motorbike       0.7095
dog             0.7266
aeroplane       0.7442
car             0.7772
bus             0.7889
person          0.7992
cat             0.8136
bird            0.8178
background      0.9223
----------------------
mIoU            0.5813
pixel_acc       0.9104
mean_acc        0.6763
```

</details>

### 实验 2:stride-4 输出("FCN-4s")+ SBD 增广(mIoU 0.6910)

对着实验 1 暴露的两个短板各动了一刀——**注意这次同时动了模型和数据两个变量**,归因讨论见本节末尾:

1. **模型:FPN 多下探一级,输出从 stride-8 提到 stride-4**(backbone 加 tap C2 [stride 4, 64ch],neck 融合到 p2,head 改 4× 上采样)——冲着实验 1 里"细结构受 stride-8 分辨率限制"的猜想去的。同时把 ConvSet 从 5 卷积减到 3 卷积、head 的 ConvSet 收窄到 128ch:分辨率翻倍多出的算力/显存用瘦身补回,neck+head 参数反而从 11.0M 降到 6.1M。
2. **数据:`USE_SBD = True`**。实际训练集 = VOC train 1464 + SBD `train_noval` 5623 = **7087 张**(×4.8),不是文献常说的 ~10.5k——10582 是 train_aug(把 SBD 的 val 部分也并入)的数字;本实现用 `train_noval`(SBD train 中剔除了全部 VOC2012 val 的图)保证 val 零泄漏,代价是少 ~3.5k 张。
3. epoch 改成 26 + 36:数据多了,每 epoch 从 91 批变成 442 批,总迭代反而从 7.3k 涨到 27.4k 批。

best proxy 0.6841 @ epoch 55;全量评估 **mIoU 0.6910 / pixel_acc 0.9295 / mean_acc 0.7811**,比实验 1 +0.1097。三个值得记下的观察:

**1. 实验 1 的"解冻是 3 倍杠杆"结论要修正:那是小数据下的现象。**

| 阶段 | 实验 1(1464 张) | 实验 2(7087 张) |
|---|---|---|
| Stage 1(冻结 backbone)proxy | 0.196 | **0.582**(峰值;结束时 0.574) |
| Stage 2(解冻)proxy 峰值 | 0.557 | 0.684 |
| 解冻的增益 | ≈ ×2.8 | **+0.10** |

实验 2 冻结 backbone、只训 neck+head 就到了 proxy 0.582——超过实验 1 解冻全训的最好成绩。所以"冻结的 ImageNet backbone 是瓶颈"这个从检测项目带来的结论,成立前提是数据少:1464 张时 neck/head 很快把数据背熟,只能动 backbone 挖潜;数据 ×4.8 后 neck/head 自己都还没吃饱,解冻的边际收益从近 3 倍缩到 +0.10。

**2. cow 0.03 → 0.70:实验 1 的"语义近邻互吞"诊断被证实。** cow 0.0298 → 0.6974,horse 0.3564 → 0.6939,sheep 0.6162 → 0.7756,四足动物组整体归位。模型虽然也换了,但 stride-4 针对的是细结构分辨率,对"把 cow 认成 horse"这种语义级混淆没有作用机制——这块修复基本可记在数据头上。

**3. 家具组仍是短板,sofa 还倒退了。** chair 0.11 → 0.31(仍垫底)、sofa 0.42 → **0.35(负增长)**、diningtable 0.46 → 0.47(原地):chair 涨、sofa 跌,像素多半只是在这组内部重新分配——VOC 里 chair/sofa/diningtable 语义边界本来就模糊、遮挡严重,不是数据量或分辨率能解决的。细结构组涨幅平平:bicycle 0.31 → 0.39、pottedplant 0.40 → 0.55,专门冲它们去的 stride-4 有没有净贡献,和 SBD 的贡献混在一起拆不出来。

**归因:+0.1097 无法干净拆分,但两条证据指向数据是大头**——(a) 冻结阶段(结构还瘦了、尚未解冻)就到 0.582;(b) 涨得最多的类恰好全是实验 1 判为"数据不足型混淆"的类。要拆干净需要单变量补跑(旧 stride-8 模型 + SBD),已列入待办。

<details>
<summary>实验 2 完整 21 类 IoU(eval.py 原始输出)</summary>

```
class              IoU
----------------------
chair           0.3080
sofa            0.3534
bicycle         0.3943
diningtable     0.4661
pottedplant     0.5538
boat            0.6055
tvmonitor       0.6864
horse           0.6939
cow             0.6974
bottle          0.7085
train           0.7691
motorbike       0.7710
sheep           0.7756
dog             0.7926
car             0.8162
bird            0.8172
person          0.8241
aeroplane       0.8407
cat             0.8412
bus             0.8703
background      0.9265
----------------------
mIoU            0.6910
pixel_acc       0.9295
mean_acc        0.7811
```

</details>

---

## 快速开始

所有命令在本目录(`FCN/`)下运行。

```bash
# 1. 数据:VOC2012 trainval 压缩包本身就带分割标注。
#    - 本地:若 YOLO3/FCOS 项目已下载过会自动复用(见 config.DATA_ROOT);
#    - 云上/新机器:train.py 检测到缺数据会【自动下载】,官方源挂了自动切
#      pjreddie 镜像、带 md5 校验、断点重跑不重下——所以这步可跳过;
#    - USE_SBD=True(实验 2 起为默认)时还会自动下 SBD(~1.4 GB,需 scipy;
#      官方源不稳,挂了按报错提示手动放置 benchmark.tgz);
#    也可以手动预下载:
python dataset/voc.py --download

# 2. 训练(两阶段 finetune;config 默认 20+60 epoch,复现实验 2 用 26+36)
#    云上首跑:直接这一条命令即可(含自动下载)
python train.py --epochs-stage1 26 --epochs-stage2 36
#   可调:--batch-size 8 --num-workers 8

# 3. 全量评估(VOC2012 val 1449 张,输出 mIoU / pixel_acc / 每类 IoU,最差在前)
python eval.py
python eval.py --max-batches 100         # 快速抽查

# 4. 可视化分割(overlay / pred / GT 三件套存到 segment/results/)
python segment/segment.py --voc-random 10   # 随机抽 10 张 VOC2012 val
python segment/segment.py --img path/to/image.jpg

# 5. 随机抽查 pred vs GT(每图 <id>_pred.png + <id>_gt.png 存到 detect/results/,
#    与检测项目的 detect.py 对应)
python detect/detect.py --n 10
```

每个模块文件都带自测入口,可以单独跑来验证形状/逻辑(全部离线,不需要数据集):

```bash
python model/backbone.py ; python model/neck.py ; python model/head.py ; python model/fcn.py
python dataset/transforms.py ; python dataset/voc.py
python losses/fcn_loss.py ; python utils/metrics.py ; python utils/viz.py
```

### 依赖

与检测项目相同(少了 torchmetrics/pycocotools——mIoU 是手写的混淆矩阵):

```bash
pip install torch torchvision numpy matplotlib tqdm pillow
# USE_SBD=True(实验 2 起为默认)额外需要: pip install scipy
```

---

## 架构

```
image [B,3,H,W]  (训练时 H=W=480 随机裁剪;评估时原尺寸 pad 到 /32)
  └─ backbone (ResNet-34, ImageNet 预训练)  ──►  c2,c3,c4,c5  (stride 4/8/16/32, 通道 64/128/256/512)
       └─ neck (FPN: lateral 1x1 + 自顶向下相加 + ConvSet)  ──►  只保留 p2 [B,256,H/4,W/4]
            └─ head (ConvSet 收窄到 128 + 1x1 分类器 + 4x bilinear 上采样)  ──►  logits [B,21,H,W]
```

以上是当前代码(实验 2)的 stride-4 版;实验 1 的 stride-8 版不 tap C2、融合止于 p3、8× 上采样、ConvSet 为 5 卷积,见 git 历史(`e7a54b2` 之前)。

- **21 类 = 20 个 VOC 类 + 显式 background(id 0)**:分割里每个像素必须有归属,background 不能像检测那样"不预测就是背景"。
- 标签 png 里 **255 = ignore**(物体边缘的白色轮廓);pad 出来的边也填 255——loss(`ignore_index`)和 mIoU(混淆矩阵前过滤)经同一机制自动跳过。
- 输出是 **raw logits**(不做 softmax),`nn.CrossEntropyLoss(ignore_index=255)` 内部做 log-softmax;推理时对 dim=1 取 argmax 得到每像素类别。
- 先出 21 通道 logits 再 4× bilinear 上采样(比先上采样 256 通道特征便宜 ~12×,与原版 FCN 一致);用 bilinear 不用 nearest,否则 mask 边界呈 4×4 色块。

### 两阶段 finetune(与检测项目完全一致)

| 阶段 | 做什么 | epoch(实验 1 / 实验 2) | LR |
|---|---|---|---|
| Stage 1 | 冻结整个 backbone,只训 neck+head | 20 / 26 | 1e-3 |
| Stage 2 | 解冻整个 backbone 端到端 finetune | 60 / 36 | head 1e-4 / backbone 3e-5 |

---

## 与检测项目的逐点对照(进入分割领域最值得看的表)

| | 检测(YOLO3 / FCOS) | 分割(本项目) |
|---|---|---|
| 预测目标 | 每个位置一个向量(框+分数) | **每个像素一个类别** |
| 标签形式 | 框列表(XML 解析、归一化) | **一张 png,像素值即类别 id** |
| background | 隐式("不预测"即背景) | **显式 class 0** |
| 正样本分配 | anchor 匹配 / 范围分层(loss 里最复杂的部分) | **不存在**——标签图天然逐像素对齐 |
| loss | 3 项(cls/box/obj 或 ctr),focal 抗失衡 | **1 项交叉熵**(每图上万前景像素,无需 focal) |
| 数据增强 | 图变换 + 框坐标重算 | 图和 mask **同参数联合变换**;mask 必须 NEAREST 插值(bilinear 会把类别 id 平均出无意义值) |
| 预处理 | 拉伸成 416×416 正方形 | 随机缩放+裁剪 480(训练)/ 原尺寸 pad /32(评估),**不拉伸** |
| collate | 自定义(每图框数不同) | **默认 collate**(裁剪后同尺寸);val 逐张(batch=1) |
| 后处理 | 解码 + 置信度过滤 + NMS | **argmax,没了** |
| 指标 | mAP(torchmetrics,重机器) | **mIoU(手写混淆矩阵,~40 行)**;每类 IoU 免费得到 |
| 可视化 | 画框 | **VOC 调色板上色 + overlay** |
| neck 输出 | 3 层金字塔(p3/p4/p5 各自预测) | **只留最细一层**(金字塔在 neck 内融合;实验 1 到 stride-8 p3 ≈ FCN-8s,实验 2 到 stride-4 p2 ≈ "FCN-4s") |

数据下载、两阶段训练循环、日志/曲线/checkpoint、设备选择等骨架与 FCOS 项目逐行相同——diff `train.py` 就能精确看到任务切换动了哪几行(答案:loss 调用和指标,其余全同)。

---

## 最终配置(`config.py` 要点)

```python
NUM_CLASSES = 21 ; IGNORE_INDEX = 255
CROP_SIZE = 480 ; SCALE_RANGE = (0.5, 2.0) ; SIZE_DIVISOR = 32
BACKBONE = "resnet34" ; FPN_CHANNELS = 256
BATCH_SIZE = 16 ; WEIGHT_DECAY = 1e-3
STAGE1_EPOCHS = 20 ; STAGE1_LR = 1e-3
STAGE2_EPOCHS = 60 ; STAGE2_LR_HEAD = 1e-4 ; STAGE2_LR_BACKBONE = 3e-5
STAGE2_UNFREEZE = "all"
EVAL_MAX_BATCHES = 300      # 每 epoch 的 mIoU proxy(全量评估留给 best.pt)
USE_SBD = True              # SBD train_noval 增广(1464 -> 7087 张),需 scipy;实验 2 起开
```

实验 2 的 26+36 epoch 是命令行覆盖的(`--epochs-stage1 26 --epochs-stage2 36`),config 默认仍是 20+60。

## 文件结构

```
FCN/
├─ config.py              # 所有超参 / 路径(单一配置入口;注意没有 anchor/NMS 相关项了)
├─ train.py               # 两阶段训练,日志/曲线/checkpoint 写到 outputs/
├─ eval.py                # 全量 mIoU + 每类 IoU(最差在前;无需单独的 eval_per_class)
├─ dataset/
│  ├─ voc.py              # VOCSegDataset(+可选 SBD)+ 下载;无自定义 collate
│  └─ transforms.py       # 图+mask 联合变换(mask 一律 NEAREST;pad 用 255)
├─ model/
│  ├─ fcn.py              # 组装 backbone→FPN→分割头 + finetune 解冻开关
│  ├─ backbone.py         # ResNet-18/34 backbone(与检测项目相同)
│  ├─ neck.py             # FPN 自顶向下融合,只输出 stride-8 的 p3
│  └─ head.py             # ConvSet + 1x1 分类器 + 8x bilinear 上采样
├─ losses/fcn_loss.py     # 逐像素交叉熵(ignore_index=255),就一项
├─ utils/
│  ├─ metrics.py          # 混淆矩阵 mIoU / pixel_acc / mean_acc + 每类表
│  └─ viz.py              # VOC 官方调色板上色 + overlay
├─ segment/segment.py     # 推理可视化(overlay / pred / gt 三件套)
├─ detect/detect.py       # 随机抽查 val:每图 <id>_pred.png + <id>_gt.png 写到 detect/results/
└─ outputs/               # best.pt / last.pt / training_log.json / 曲线 png(*.pt 被 .gitignore 排除,日志与曲线入库)
```

## 待办 / 实验计划

- [x] 跑通两阶段训练,记录 mIoU 曲线;确认"stage2 解冻 backbone"在分割上是否同样是最大杠杆——**实验 1 成立(0.196 → 0.5813,近 3 倍);实验 2 修正:是小数据现象,数据 ×4.8 后冻结 backbone 也能到 0.58,解冻只再 +0.10**
- [x] 开 `USE_SBD = True` 增广(train_noval,1464 → 7087 张)——实验 2:mIoU +0.1097、cow/horse/sheep 混淆消失;但与 stride-4 改模型同批变更,贡献没拆开
- [ ] 单变量补跑:旧 stride-8 模型 + SBD,把实验 2 的 +0.11 拆成"数据的"与"结构的"
- [ ] 消融:FPN 融合(等价 FCN-8s)vs 只用 c5 直接 32× 上采样(等价 FCN-32s),验证跳连的价值
- [ ] 消融:末端上采样(实验 2 为 4×)bilinear vs nearest vs 转置卷积
- [x] 观察每类 IoU:预判(bottle、bicycle、pottedplant 最差)只对一半——bicycle 0.31 / pottedplant 0.40 确实差,但 bottle 0.58 不垫底;最差是 cow 0.03 / chair 0.11,是语义近邻混淆(数据量问题)而非分辨率问题,见"结果"。stride-8 分辨率的真实代价、DeepLab(空洞卷积)/U-Net(更高分辨率解码)的对照,留到 SBD 补数据之后再做(实验 2 更新:SBD 后 cow 归位,证实混淆诊断;chair/sofa/bicycle 仍差,见"实验 2"观察 3。U-Net 对照已做:stride-2 解码也没救细结构,分辨率假设结案,见 ../U-Net)
