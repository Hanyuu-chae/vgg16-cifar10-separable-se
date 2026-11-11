import argparse
import os
import time
import random
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from model import ModifiedVGG

# CIFAR-10 类别
CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

# 1. 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 2. 创建模型
def build_model(model_type):

    if model_type == "baseline":
        return ModifiedVGG(
            use_separable=False,
            use_se=False
        )

    elif model_type == "dw":
        return ModifiedVGG(
            use_separable=True,
            use_se=False
        )

    elif model_type == "se":
        return ModifiedVGG(
            use_separable=False,
            use_se=True
        )

    elif model_type == "dw_se":
        return ModifiedVGG(
            use_separable=True,
            use_se=True
        )

    else:
        raise ValueError("Unsupported model type")

# 3. 统计参数量
def calculate_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

# 4. 统计 FLOPs
def calculate_flops(
    model,
    input_size=(1, 3, 32, 32),
    device="cpu"
):

    try:
        from thop import profile

        dummy_input = torch.randn(
            *input_size
        ).to(device)

        model.eval()

        flops, params = profile(
            model,
            inputs=(dummy_input,),
            verbose=False
        )

        return flops

    except ImportError:
        print("Warning: thop not installed. FLOPs skipped.")
        return None

    except Exception as e:
        print(f"Warning: FLOPs calculation failed: {e}")
        return None

# 5. 训练一个 Epoch
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs = inputs.to(device)
        targets = targets.to(device)

        # 梯度清零
        optimizer.zero_grad(set_to_none=True)

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 统计损失
        running_loss += loss.item() * inputs.size(0)

        # 统计准确率
        predicted = outputs.argmax(dim=1)

        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):

            current_acc = 100.0 * correct / total

            current_avg_loss = running_loss / total

            elapsed = time.time() - start_time

            print(
                f"Batch [{batch_idx + 1}/{len(loader)}] "
                f"Progress: {(batch_idx + 1) / len(loader) * 100:.1f}% | "
                f"Loss: {loss.item():.4f} | "
                f"Avg Loss: {current_avg_loss:.4f} | "
                f"Acc: {current_acc:.2f}% | "
                f"Time: {elapsed:.1f}s",
                flush=True
            )

    epoch_loss = running_loss / total

    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc

# 6. 验证和测试
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    all_preds = []

    all_targets = []

    with torch.no_grad():

        for inputs, targets in loader:

            inputs = inputs.to(device)

            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            predicted = outputs.argmax(dim=1)

            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

            all_preds.extend(
                predicted.cpu().numpy()
            )

            all_targets.extend(
                targets.cpu().numpy()
            )

    loss = running_loss / total

    accuracy = 100.0 * correct / total

    return (
        loss,
        accuracy,
        np.array(all_targets),
        np.array(all_preds)
    )

# 7. 保存训练曲线
def save_training_curves(history, output_dir):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    # Loss曲线
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training and Validation Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "loss_curve.png"
        ),
        dpi=300
    )

    plt.close()

    # Accuracy曲线
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_acc"],
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["val_acc"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.title("Training and Validation Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "accuracy_curve.png"
        ),
        dpi=300
    )

    plt.close()

# 8. 保存混淆矩阵
def save_confusion_matrix(
    targets,
    preds,
    output_dir
):

    cm = confusion_matrix(
        targets,
        preds,
        labels=list(range(10))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False
    )

    plt.title("CIFAR-10 Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()

    np.savetxt(
        os.path.join(
            output_dir,
            "confusion_matrix.csv"
        ),
        cm,
        delimiter=",",
        fmt="%d"
    )

# 9. 主程序
def main():

    parser = argparse.ArgumentParser(
        description="VGG16 CIFAR-10 Training"
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="dw_se",
        choices=[
            "baseline",
            "dw",
            "se",
            "dw_se"
        ]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2
    )

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 选择设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)

    print("VGG16 CIFAR-10 Classification")

    print("=" * 60)

    print(f"Device: {device}")

    print(f"Model: {args.model_type}")

    print(f"Epochs: {args.epochs}")

    print(f"Batch Size: {args.batch_size}")

    print(f"Learning Rate: {args.lr}")

    print(f"Random Seed: {args.seed}")

    # 创建结果目录
    output_dir = os.path.join(
        "results",
        args.model_type
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # 10. 数据预处理
    transform_train = transforms.Compose([

        transforms.RandomCrop(
            32,
            padding=4
        ),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )

    ])

    transform_eval = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )

    ])

    print("\nLoading CIFAR-10 dataset...")

    train_dataset_aug = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    train_dataset_eval = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_eval
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_eval
    )

    # 划分训练集和验证集
    generator = torch.Generator().manual_seed(
        args.seed
    )

    indices = torch.randperm(
        len(train_dataset_aug),
        generator=generator
    ).tolist()

    train_indices = indices[:45000]

    val_indices = indices[45000:]

    train_dataset = Subset(
        train_dataset_aug,
        train_indices
    )

    val_dataset = Subset(
        train_dataset_eval,
        val_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    print(f"Training Samples: {len(train_dataset)}")

    print(f"Validation Samples: {len(val_dataset)}")

    print(f"Testing Samples: {len(test_dataset)}")

    # 11. 初始化模型
    model = build_model(
        args.model_type
    ).to(device)

    total_params = calculate_parameters(model)

    print(
        f"\nTrainable Parameters: {total_params:,}"
    )

    flops = calculate_flops(
        model,
        device=device
    )

    if flops is not None:

        print(
            f"FLOPs: {flops / 1e9:.4f} GFLOPs"
        )

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    model_path = os.path.join(
        output_dir,
        "best_model.pth"
    )

    best_acc = -1.0

    # 训练历史
    history = {

        "train_loss": [],

        "train_acc": [],

        "val_loss": [],

        "val_acc": []

    }

    total_start = time.time()

    # 12. 开始训练
    print("\nStart Training...\n")

    for epoch in range(args.epochs):

        epoch_start = time.time()

        print("\n" + "=" * 60)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}]"
        )

        print("=" * 60)

        # 训练
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # 验证
        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(
            train_loss
        )

        history["train_acc"].append(
            train_acc
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_acc"].append(
            val_acc
        )

        print("\nEpoch Summary:")

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_acc:.2f}%"
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: {val_acc:.2f}%"
        )

        print(
            f"Epoch Time: {epoch_time:.2f}s"
        )

        # 保存验证集最优模型
        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": args.model_type,
                    "best_val_acc": best_acc,
                    "epoch": epoch + 1,
                    "seed": args.seed
                },
                model_path
            )

            print(
                f"\nSaved Best Model: {model_path}"
            )

    total_time = time.time() - total_start

    print("\n" + "=" * 60)

    print("Training Finished!")

    print(
        f"Best Validation Accuracy: {best_acc:.2f}%"
    )

    print(
        f"Total Training Time: {total_time:.2f}s"
    )

    print("=" * 60)

    # 13. 保存训练历史
    save_training_curves(
        history,
        output_dir
    )

    history_path = os.path.join(
        output_dir,
        "training_history.csv"
    )

    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc"
        ])

        for i in range(args.epochs):

            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i]
            ])

    # 14. 加载最佳模型
    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model weights not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # 15. 测试集评估
    print("\nEvaluating Best Model on Test Set...")

    test_loss, test_acc, all_targets, all_preds = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(
        f"\nTest Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_acc:.2f}%"
    )

    # 分类报告
    report = classification_report(
        all_targets,
        all_preds,
        labels=list(range(10)),
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report:")

    print(report)

    report_path = os.path.join(
        output_dir,
        "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    # 16. 混淆矩阵
    save_confusion_matrix(
        all_targets,
        all_preds,
        output_dir
    )

    # 17. 保存实验摘要
    summary_path = os.path.join(
        output_dir,
        "experiment_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("VGG16 CIFAR-10 Experiment Summary\n")

        f.write("=" * 40 + "\n")

        f.write(
            f"Model: {args.model_type}\n"
        )

        f.write(
            f"Epochs: {args.epochs}\n"
        )

        f.write(
            f"Batch Size: {args.batch_size}\n"
        )

        f.write(
            f"Learning Rate: {args.lr}\n"
        )

        f.write(
            f"Random Seed: {args.seed}\n"
        )

        f.write(
            f"Parameters: {total_params}\n"
        )

        if flops is not None:

            f.write(
                f"FLOPs: {flops}\n"
            )

        f.write(
            f"Best Validation Accuracy: {best_acc:.2f}%\n"
        )

        f.write(
            f"Test Loss: {test_loss:.4f}\n"
        )

        f.write(
            f"Test Accuracy: {test_acc:.2f}%\n"
        )

        f.write(
            f"Training Time: {total_time:.2f}s\n"
        )

    print("\nAll results saved to:")

    print(output_dir)

if __name__ == "__main__":

    main()