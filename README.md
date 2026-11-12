# DANet-Seg: Dual Attention for Scene Segmentation

> 项目二：注意力增强的编码器-解码器场景分割（Pascal VOC 2012 · PyTorch）
> 对应期中考核题目 **2. 场景分割**
> 任务：使用注意力机制增强的编码器-解码器架构进行场景分割
> 数据集：Pascal VOC / COCO　｜　注意力：DANet（位置注意力 PAM + 通道注意力 CAM，并行相加）

---

## 0. 一页速览

| 项目 | 内容 |
|---|---|
| 任务类型 | 稠密预测（逐像素分类） |
| 输入 / 输出 | RGB 图 → 同尺寸类别图（21 类） |
| 主干 | ResNet-50/101（ImageNet 预训练，dilation，输出 stride=8） |
| 注意力 | DANet：位置注意力 PAM + 通道注意力 CAM（并行相加） |
| 数据集 | Pascal VOC2012 seg（1464 train / 1449 val，21 类） |
| 主指标 | mIoU（辅：Pixel Acc、per-class IoU） |
| 必做实验 | 无注意力 baseline vs +PAM vs +CAM vs +PAM+CAM |
| 报告篇幅 | 约 3.5 页（8 页含封面，两个项目平分） |

**选型理由**：VOC 数据量小、单卡几小时可出结果、mIoU 对注意力模块敏感，最容易做出"有提升、有可视化、有分析"的完整闭环。COCO 作为可选扩展（数据量大 10 倍以上，仅建议在 VOC 跑通后考虑）。

---

## 1. 任务定义

对图像中每个像素 $p$ 预测类别 $c \in \{0,...,20\}$，输出 $H\times W$ 的标签图。
与目标检测的区别：不输出框，输出精确到像素的区域；与实例分割的区别：同类物体不区分个体。

难点集中在三处：**边界精度**（下采样丢失细节）、**多尺度**（近处大物体与远处小物体共存）、**长程依赖**（天空/道路这类大面积区域的像素需要互相"认同"）。注意力机制主要解决第三点，解码器的 skip connection 解决第一点。

---

## 2. 技术路线（架构）

```
输入 RGB (3×512×512)
      │
 ┌────▼──────────────────────────────────┐
 │ ResNet-50/101  (ImageNet 预训练)      │
 │ 移除最后两个下采样，dilation = 2 / 4  │
 │ 输出 stride=8 特征  (1024×64×64)      │
 └────┬──────────────────────────────────┘
      │ 3×3 conv, 降维到 512×64×64
      ├────────────────┬─────────────────┐
      ▼                ▼                 │
 ┌──────────┐   ┌───────────┐            │
 │ 位置注意力│   │ 通道注意力 │            │
 │   PAM    │   │    CAM    │            │
 └────┬─────┘   └─────┬─────┘            │
      └───────┬───────┘                  │
              ▼ 逐元素相加                │
        3×3 conv → 1×1 conv (21 类)      │
              ▼                          │
        8× 双线性上采样 → 512×512×21      │
              ▼                          │
        逐像素 argmax → 分割图
```

**两个注意力模块的作用（报告里要讲清楚，这是得分点）**

- **PAM（位置注意力）**：把特征图 reshape 成 $N\times C$（$N=H\times W$），算 $N\times N$ 的相似度矩阵，softmax 后作为权重对特征做加权聚合。效果是**任意两个像素直接建立联系**，远处同类物体互相增强，缓解感受野受限导致的区域碎片化。计算量 $O(N^2C)$，64×64 时可接受。
- **CAM（通道注意力）**：在通道维算 $C\times C$ 相似度，捕捉不同通道（≈不同语义响应）之间的依赖，相当于自适应的通道重标定，比 SE 的全局池化更细粒度。
- 两者输出维度对齐后**相加**（DANet 原文做法），再进分类头。
- **辅助损失**：PAM 分支和 CAM 分支各自还接一个 1×1 分类头，训练时也计入交叉熵（权重 0.4），推理时丢弃。这是 DANet 原文的训练策略，去掉会掉点；实现上就是 `DANetHead` 里的 `aux_loss` 开关。

**为什么要解码器**：stride=8 的特征直接上采样边界很糊。可选两种做法：① 直接 8× 双线性上采样（DANet 原版，简单）；② 级联上采样 + 融合 stride=4/2 的浅层特征（类似 U-Net/DeepLabv3+，边界更好但代码更多）。**建议先做 ①，跑通后再考虑 ②**。

---

## 3. 数据方案

| 项 | 设定 |
|---|---|
| 数据集 | VOC2012 segmentation（`SegmentationClass` + `ImageSets/Segmentation`） |
| 类别 | 21（20 类前景 + background） |
| 划分 | 官方 train 1464 / val 1449（**不做交叉验证**，报告里如实写明用官方划分 + 固定随机种子） |
| 忽略标签 | 边界像素标为 255，损失与指标中忽略 |
| 训练增强 | 随机缩放 0.5~2.0 → 随机裁剪 512×512 → 随机水平翻转 → 颜色抖动 |
| 验证 | **不缩放、不增强**，只把右下角 pad 到 8 的倍数（模型 stride=8 要求整除），pad 区 mask 填 255 并在指标中忽略；因此预测与 GT 同坐标系。各图尺寸不同，验证 batch_size=1 |
| 可选增强 | 加入 SBD 数据集（8498 张）可显著提点，但需额外下载与标签对齐，属于加分项 |

> 说明：模板里问"几折交叉验证"，本任务用官方 train/val 划分更规范；若老师坚持要 CV，可用 val 做 5 折，代价是训练时间 ×5，不建议。

---

## 4. 评估方案

**训练超参（初始值，据显存调整）**

| 超参 | 值 |
|---|---|
| optimizer | SGD, momentum 0.9, weight decay 1e-4 |
| lr | 0.001，poly 衰减 $(1-\frac{iter}{max})^0.9$ |
| batch size | 8（512 crop，AMP 混合精度） |
| epochs | 50 |
| loss | 逐像素交叉熵（ignore_index=255），可选加 class weight |
| 硬件 | 单卡 8~12GB 可跑；不足则 crop 降到 320、batch 降到 4 |

**定量指标**

- **mIoU**（主指标）：各类 IoU 的平均
- Pixel Accuracy（PA）、Mean Accuracy（MA）
- per-class IoU 表（21 行），用于分析哪些类被注意力改善、哪些类仍然差
- 可选：多尺度 + 水平翻转推理（TTA），通常 +1~2 mIoU

**定性图表**

1. 三联图：原图 / GT / 预测（选 4~6 组，含成功与失败）
2. **注意力热力图**：PAM 的 attention map、CAM 的通道权重，叠加到原图上
3. 失败案例：小目标漏检、相邻同类物体粘连、类别混淆（如"沙发/椅子"）

---

## 5. 实验矩阵（消融，报告的核心表格）

| # | 配置 | 目的 | 预期 mIoU（VOC val） |
|---|---|---|---|
| A | ResNet-50 + 直接上采样（无注意力） | baseline | 68~71 |
| B | A + PAM only | 验证位置注意力贡献 | 74~76 |
| C | A + CAM only | 验证通道注意力贡献 | 73~75 |
| D | A + PAM + CAM（完整 DANet） | 最终方案 | 75~78 |

> 只做 DANet 一条线：SE / CBAM 这类外部注意力对照不做了。B（PAM only）与 C（CAM only）是 DANet 自身的两个分支，代码里只是 `attention='pam'/'cam'` 一个开关，几乎零成本，保留下来正好支撑"位置注意力 vs 通道注意力"的消融分析。

> 参考：DANet 原文 VOC val 80.5（ResNet-101 + 多尺度测试 + COCO 预训练）。课程作业复现到 75~78 即为合理结果，**不要谎报**，如实写"低于原文，原因是未使用额外数据与 TTA"反而是加分项。

时间不够时的最小集：**A / D 两行**即可支撑"注意力有效"的结论。

---

## 6. 图表产出清单（直接对应报告章节）

| 文件 | 用途 |
|---|---|
| `figures/arch.png` | 架构图（建议 Visio/draw.io 重画，别用 ASCII） |
| `figures/qualitative.png` | 原图/GT/预测 三联图 |
| `figures/attention_pam.png` | PAM 注意力热力图 |
| `figures/attention_cam.png` | CAM 通道注意力可视化 |
| `figures/failure.png` | 失败案例 |
| `outputs/metrics.csv` | 各配置 mIoU/PA/MA |
| `outputs/per_class_iou.csv` | 逐类 IoU |
| `outputs/train_log.csv` | loss/mIoU 曲线数据 |

---

## 7. 里程碑

| 阶段 | 产出 | 验收标准 |
|---|---|---|
| M1 数据管线 | `src/dataset.py` + 可视化 3 张样本 | 图像与 mask 对齐、增强可复现 |
| M2 baseline | 配置 A 训练完成 | val mIoU ≥ 65，能出三联图 |
| M3 注意力 | 配置 D/E/F | 相对 A 有稳定提升 |
| M4 可视化与分析 | 热力图 + 逐类 IoU | 能解释"注意力改善了哪类错误" |
| M5 报告 | 3.5 页 PDF | 对照 `报告大纲.md` 逐节检查 |

---

## 8. 风险与降级

| 风险 | 触发信号 | 降级方案 |
|---|---|---|
| 显存不足 | OOM @ batch 8 / 512 | crop→320、batch→4、开 AMP、冻结 backbone 前两层 |
| 训练太慢 | 单 epoch > 40 min | 减到 30 epoch、backbone 换 ResNet-50、图像降采样 |
| 注意力无提升 | D 与 A 差距 < 0.5 mIoU | 检查学习率（注意力模块常需更大 lr）、确认 PAM 的 softmax 维度没写反、确认辅助损失（aux_weight=0.4）已开启、延长训练 |
| 边界太糊 | mIoU 尚可但视觉差 | 加上采样融合浅层特征（U-Net 式 skip） |

---

## 9. 目录说明

```
02-scene-segmentation/
├── README.md              # 本文件：任务框架
├── 报告大纲.md             # 对应学校模板的 8 页大纲
├── requirements.txt       # 依赖
├── LICENSE                # MIT
├── .gitignore             # 排除 data/ outputs/ 与权重
├── docs/                  # 论文笔记、实验记录
├── src/
│   ├── config.py          # 训练超参 dataclass
│   ├── losses.py          # 交叉熵 + DANet 辅助损失
│   ├── metrics.py         # 混淆矩阵 → mIoU / PA / MA / 逐类 IoU
│   ├── engine.py          # 一个 epoch 的训练与验证
│   ├── utils.py           # 种子、poly 学习率、CSV 日志、计时
│   ├── data/
│   │   ├── voc.py         # VOC2012 数据集与 collate
│   │   └── transforms.py  # 训练/验证增强（图像与 mask 同步）
│   └── models/
│       ├── attention.py   # PAM / CAM
│       ├── backbone.py    # Dilated ResNet
│       └── danet.py       # DANetHead + DANet + 配置
├── scripts/
│   ├── train.py           # 训练入口
│   ├── evaluate.py        # 验证并导出 metrics.csv / per_class_iou.csv
│   ├── visualize.py       # 三联图、PAM/CAM 热力图
│   └── smoke_test.py      # 不依赖数据的形状自检
├── data/                  # VOC2012 数据集（不入库）
├── outputs/               # 权重、日志、指标 CSV
└── figures/               # 报告用图
```

---

## 10. 数据准备（下载 VOC2012）

仓库里**不含数据集**（`data/` 为空，约 2 GB，不入库），需要自己下载。官方地址实测可用：

| 项 | 值 |
|---|---|
| 下载地址 | `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar` |
| 大小 | 1,999,639,040 字节（≈1.86 GiB） |
| 内容 | `VOCdevkit/VOC2012/` 下的 JPEGImages、SegmentationClass、SegmentationObject、ImageSets |

> ⚠️ 必须用 **http**：同一地址的 https 版本在本机 TLS 握手直接失败（`SSL: UNEXPECTED_EOF_WHILE_READING`）。
> 压缩包里已经是 `VOCdevkit/VOC2012/` 结构，解压到 `data/` 即可，代码会自动识别 `data/VOCdevkit/VOC2012` 或 `data/VOC2012`。

**Windows（PowerShell，系统自带 `curl.exe` 和 `tar.exe`）**

```powershell
# 以下命令都在项目根目录（含 README.md 的文件夹）执行，路径全部用相对路径
# 快捷方式：在资源管理器里进入该目录，地址栏输入 powershell 回车，即可直接在此处打开
New-Item -ItemType Directory -Force data | Out-Null
curl.exe -L -C - -o data\VOCtrainval_11-May-2012.tar http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf data\VOCtrainval_11-May-2012.tar -C data
```

**Linux / macOS**

```bash
mkdir -p data && cd data
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
cd ..
```

**校验（下载/解压后必做）**

```powershell
(Get-Content data\VOCdevkit\VOC2012\ImageSets\Segmentation\train.txt).Count  # 1464
(Get-Content data\VOCdevkit\VOC2012\ImageSets\Segmentation\val.txt).Count    # 1449
Test-Path data\VOCdevkit\VOC2012\SegmentationClass\2007_000032.png           # True
(Get-Item data\VOCtrainval_11-May-2012.tar).Length                           # 1999639040
```

```bash
wc -l data/VOCdevkit/VOC2012/ImageSets/Segmentation/{train,val}.txt   # 1464 / 1449
```

解压后的结构：

```
data/VOCdevkit/VOC2012/
├── JPEGImages/              17125 张
├── SegmentationClass/       2913 张（train 1464 + val 1449）
├── SegmentationObject/
└── ImageSets/Segmentation/{train,trainval,val}.txt
```

下载中断不用重来，上面的命令都带续传（`curl -C -` / `wget -c`）；确认无误后可以删掉 tar 包省 2 GB。

---

## 11. 运行方式

前置：先按 §10 下好 VOC2012。以下命令都在项目根目录（`02-scene-segmentation/`）执行。

### 11.1 环境安装

从 PyPI 直接 `pip install torch` 装到的是 **CPU 版**（版本号带 `+cpu`，`cuda.is_available()` 为 `False`），DANet 在 CPU 上训练不现实。**必须先确认自检输出里是 `+cu12x` 且第三项为 `True`。**

本机驱动 566.07 对应 CUDA 12.x，因此用 **cu126** 的轮子（cu126 索引里 cp312-win 的 torch 最新为 2.14.1；2.15.0 只在 cu132 提供，需要驱动 580+）。一次性把 `machinelearn` 换成 CUDA 版：

```bash
conda activate machinelearn
python -m pip uninstall -y torch torchvision
python -m pip install --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple "torch==2.14.1+cu126" "torchvision==0.29.1+cu126"
python -c "import torch, torchvision; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

期望输出 `2.14.1+cu126 12.6 True`（命令已写成一行，直接复制即可，不要拆行）。

再确认 GPU 真能算：

```bash
python -c "import torch; x=torch.randn(2000,2000,device='cuda'); print((x@x).sum().item())"
```

> 装完之后**不要**再跑 `pip install -r requirements.txt`，它会按 PyPI 把 torch 换回 CPU 版。
>
> 训练进度条用 `tqdm`（可选）：`python -m pip install tqdm`。不装也能跑，会自动退化成内置的简易进度条。

### 11.2 自检

不需要数据集，用随机张量确认形状、损失、反向传播都通：

```bash
python scripts/smoke_test.py --size 256 --batch 2
```

### 11.3 训练（四个消融配置）

四条命令互不干扰，各自写自己的输出目录：

| 配置 | 参数 | 输出目录 |
|---|---|---|
| A baseline | `--attention none` | `outputs/baseline_resnet50_none/` |
| B PAM only | `--attention pam` | `outputs/danet_resnet50_pam/` |
| C CAM only | `--attention cam` | `outputs/danet_resnet50_cam/` |
| D 完整 DANet | `--attention both` | `outputs/danet_resnet50_both/` |

```bash
python scripts/train.py --attention none  --epochs 50          # A baseline
python scripts/train.py --attention pam   --epochs 50          # B
python scripts/train.py --attention cam   --epochs 50          # C
python scripts/train.py --attention both  --epochs 50          # D 完整 DANet
```

- 显存不够：8 GB 卡（如 RTX 4060 Laptop）先试 `--batch-size 4`，仍 OOM 再 `--crop-size 384 --batch-size 4`；512 crop + batch 8 基本放不下（PAM 的 N×N 注意力在 512 输入时是 4096²，单这一项就几百 MB）
- 每个 epoch 有一条进度条（`epoch 3/50 [######----] 42.0% 2.31 it/s ETA 900s  loss=0.1234 lr=9.5e-04`），验证也有一条；不想要进度条就加 `--no-progress`
- 想对齐官方 DANet 的 2048 通道：加 `--use-layer4`（更吃显存，速度约慢一半）
- 中断了：`--resume outputs/danet_resnet50_both/ckpt_last.pth`
- 每个 epoch 打印一次 val 指标；best 按 val mIoU 保存，日志在 `train_log.csv`

### 11.4 评测

对四个 checkpoint 各跑一次，结果会追加到 `outputs/metrics_summary.csv`（正好是报告表 1 的四行）：

```bash
python scripts/evaluate.py --ckpt outputs/baseline_resnet50_none/ckpt_best.pth   --tag A_baseline
python scripts/evaluate.py --ckpt outputs/danet_resnet50_pam/ckpt_best.pth       --tag B_pam
python scripts/evaluate.py --ckpt outputs/danet_resnet50_cam/ckpt_best.pth       --tag C_cam
python scripts/evaluate.py --ckpt outputs/danet_resnet50_both/ckpt_best.pth      --tag D_danet
```

每个实验目录下还会写出 `metrics.json` 和 `per_class_iou.csv`（21 行，报告表 2 用）。

### 11.5 出图

```bash
# 三联图：原图 / GT / 预测（默认取 val 前 4 张）
python scripts/visualize.py --ckpt outputs/danet_resnet50_both/ckpt_best.pth --mode triple

# PAM 热力图 + CAM 通道亲和矩阵（指定具体图片）
python scripts/visualize.py --ckpt outputs/danet_resnet50_both/ckpt_best.pth --mode all --ids 2007_000033 2007_000129

# 单独模式
python scripts/visualize.py --ckpt outputs/danet_resnet50_both/ckpt_best.pth --mode pam --num 2
python scripts/visualize.py --ckpt outputs/danet_resnet50_both/ckpt_best.pth --mode cam --num 2
```

图片写到 `figures/`：`triple_<id>.png`、`pam_<id>.png`、`cam_<id>.png`（dpi 200）。

> `--ids` 只能填 val 里的图（想要 train 的图就加 `--split train`）；填了不在该 split 的 id 会直接提示并给出可用示例。
>
> 如果报 `OMP: Error #15 ... libiomp5md.dll already initialized`：Anaconda 的 `Library/bin` 和 torch 各带一份 OpenMP，画图时两份都被加载就会 abort。`visualize.py` 已在导入 torch 前设置 `KMP_DUPLICATE_LIB_OK=TRUE`，一般不会再出现；你自己写的新脚本遇到时，先在 PowerShell 里 `$env:KMP_DUPLICATE_LIB_OK="TRUE"` 再运行。

---

每个实验在 `outputs/<experiment_name>/` 下独立成套（`config.json`、`train_log.csv`、`metrics.json`、`ckpt_best.pth`、`ckpt_last.pth`），互不覆盖，正好对应报告里的四行消融。

---

## 12. 许可与引用

代码基于 **MIT License**（见 [LICENSE](LICENSE)）。若在报告或论文中引用：

- Fu J., Liu J., Tian H., et al. *Dual Attention Network for Scene Segmentation*. CVPR 2019.
- Everingham M., et al. *The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results*.

> 数据集版权归原作者，仓库不附带数据，仅提供下载脚本。
