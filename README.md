# vgg16-cifar10-separable-se

本项目基于 **PyTorch** 实现了一个改进版的 VGG16 模型，并在 **CIFAR-10** 图像分类数据集上进行对比实验。项目的主要目的是探索**深度可分离卷积** 和 **SE 注意力机制** 对模型分类性能、参数量以及计算量的影响。

# 项目背景与动机
传统 VGG16 模型虽然结构简单、特征提取能力强，但其全连接层和标准卷积层参数量巨大，计算成本极高。为了在保持模型精度的同时实现“轻量化”，本项目引入并测试了两种主流的网络优化策略：
1. **深度可分离卷积 (DW Conv)**：将标准卷积拆分为 Depthwise 和 Pointwise 两步，大幅降低计算量。
2. **SE 注意力机制 (SENet)**：通过学习的方式获取每个特征通道的重要程度，抑制无用特征，提升有用的特征权重。

# 模型结构对比

本项目提供了 4 种不同配置的模型结构，可通过命令行参数 `--model_type` 进行切换：

| 模型类型 (`--model_type`) | 深度可分离卷积 (DW) | SE 注意力 (SE) | 说明 |
| :--- | :---: | :---: | :--- |
| `baseline` | ❌ | ❌ | 标准 VGG16 模型（基准对照组） |
| `dw` | ✅ | ❌ | 仅使用深度可分离卷积 |
| `se` | ❌ | ✅ | 仅加入 SE 注意力模块 |
| `dw_se` | ✅ | ✅ | 同时使用 DW 卷积与 SE 模块 |

# 数据集准备 
本项目使用的是 CIFAR-10 数据集。[https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

数据划分：将官方 50,000 张训练集划分为 45,000 张训练集 和 5,000 张验证集。测试集使用官方 10,000 张图像。

数据增强：训练集使用了 RandomCrop(32, padding=4) 和 RandomHorizontalFlip()，并进行了标准化。

# 使用方法
在终端中运行 train.py 即可开始训练：

```text
# 训练同时包含 DW 卷积和 SE 注意力的模型
python train.py --model_type dw_se --epochs 50 --batch_size 128 --lr 0.001

# 训练基准 Baseline 模型
python train.py --model_type baseline --epochs 50
```

# 完整参数说明：
| 参数名 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--model_type` | str | baseline | 选择模型结构 (baseline, dw, se, dw_se) |
| `--epochs` | int | 10 | 训练的总轮数 |
| `--batch_size` | int | 128 | 训练和验证时的 Batch Size |
| `--lr` | float | 0.001 | 初始学习率 |
| `--seed` | int | 42 | 随机种子 |
| `--num_workers` | int | 2 | DataLoader 的工作线程数 |

# 输出结果说明
训练完成后，程序会自动在 results/{model_type}/ 目录下生成以下文件：

best_model.pth: 验证集上准确率最高的模型权重文件。

loss_curve.png: 训练集与验证集的 Loss 随 Epoch 变化的折线图。

accuracy_curve.png: 训练集与验证集的 Accuracy 随 Epoch 变化的折线图。

training_history.csv: 记录每个 Epoch 的 loss 和 accuracy 数据，方便导入 Excel 或 Pandas 分析。

classification_report.txt: 最佳模型在测试集上的分类报告（包含每个类别的 Precision, Recall, F1-score）。

confusion_matrix.png: 测试集上的混淆矩阵可视化图。

confusion_matrix.csv: 混淆矩阵的原始数据。

experiment_summary.txt: 实验摘要文件，包含模型类型、超参数、总参数量 (Parameters)、计算量 (FLOPs)、最佳验证集准确率、测试集准确率及总训练时间。

# 项目目录结构 
```text
vgg16-cifar10-separable-se/
│
├── model.py             # 核心模型定义
├── train.py             # 主训练脚本 (包含数据加载、训练循环、评估与绘图)
├── README.md            # 项目说明文档
├── .gitignore           
│
├── data/                # CIFAR-10 数据集 
└── results/             # 训练结果输出目录 
    ├── baseline/
    ├── dw/
    ├── se/
    └── dw_se/
