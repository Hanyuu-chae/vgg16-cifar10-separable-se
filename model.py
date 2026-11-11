
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 注意力模块
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        hidden_channels = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()

        y = self.pool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1, 1)

        return x * y


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积：
    Depthwise Conv + Pointwise Conv
    """
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ModifiedVGG(nn.Module):
    """
    VGG16 风格网络，适配 CIFAR-10 的 32x32 输入。

    use_separable:
        是否使用深度可分离卷积

    use_se:
        是否使用 SE 注意力模块
    """

    def __init__(
        self,
        num_classes=10,
        use_separable=False,
        use_se=False
    ):
        super(ModifiedVGG, self).__init__()

        self.use_separable = use_separable
        self.use_se = use_se

        # VGG16 配置：13 个卷积层 + 3 个全连接层
        cfg = [
            64, 64, "M",
            128, 128, "M",
            256, 256, 256, "M",
            512, 512, 512, "M",
            512, 512, 512, "M"
        ]

        layers = []
        in_channels = 3

        for item in cfg:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                if use_separable:
                    conv = DepthwiseSeparableConv(
                        in_channels,
                        item
                    )
                else:
                    conv = nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            item,
                            kernel_size=3,
                            padding=1,
                            bias=False
                        ),
                        nn.BatchNorm2d(item),
                        nn.ReLU(inplace=True)
                    )

                layers.append(conv)

                if use_se:
                    layers.append(SEBlock(item))

                in_channels = item

        self.features = nn.Sequential(*layers)

        # CIFAR-10 输入经过 5 次池化后为 1x1
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                # 只有存在 bias 时才初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(
                    m.weight,
                    0,
                    0.01
                )

                # 只有存在 bias 时才初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

