"""
宫颈癌分类的ResNet变体
包含ResNet50、ResNet101及可选注意力机制
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetCervicalModel(nn.Module):
    """ResNet系列分类模型（支持多种变体）"""

    def __init__(
            self,
            num_classes=7,
            pretrained=True,
            dropout_rate=0.5,
            model_type='resnet50',
            use_attention=False,
            attention_type=None
    ):
        super(ResNetCervicalModel, self).__init__()

        self.num_classes = num_classes
        self.model_type = model_type
        self.use_attention = use_attention
        self.attention_type = attention_type

        # 选择backbone
        if model_type == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
        elif model_type == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
        elif model_type == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
        elif model_type == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # 移除原始全连接层
        self.backbone.fc = nn.Identity()

        # 添加注意力机制
        if use_attention and attention_type:
            if attention_type == 'se':
                self.attention = SEBlock(in_features, reduction=16)
            elif attention_type == 'cbam':
                self.attention = CBAMBlock(in_features, reduction=16)
            else:
                self.attention = None
        else:
            self.attention = None

        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_features, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化分类层权重"""
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # Backbone特征提取
        x = self.backbone(x)  # (B, in_features)

        # 注意力机制
        if self.attention is not None:
            x = self.attention(x)

        # Dropout + 分类
        x = self.dropout(x)
        x = self.fc(x)

        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力模块"""

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Squeeze
        y = self.avg_pool(x).view(b, c)

        # Excitation
        y = self.fc(y).view(b, c, 1, 1)

        # Scale
        return x * y.expand_as(x)


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 通道注意力
        x = self.channel_attention(x) * x

        # 空间注意力
        x = self.spatial_attention(x) * x

        return x


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.conv = nn.Conv2d(
            2, 1, kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


# 便捷函数：快速创建不同配置的模型
def create_resnet50(num_classes=7, pretrained=True, use_attention=False):
    """创建ResNet50模型"""
    return ResNetCervicalModel(
        num_classes=num_classes,
        pretrained=pretrained,
        model_type='resnet50',
        use_attention=use_attention,
        attention_type='se' if use_attention else None
    )


def create_resnet101(num_classes=7, pretrained=True, use_attention=False):
    """创建ResNet101模型"""
    return ResNetCervicalModel(
        num_classes=num_classes,
        pretrained=pretrained,
        model_type='resnet101',
        use_attention=use_attention,
        attention_type='cbam' if use_attention else None
    )
