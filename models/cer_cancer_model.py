import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CervicalCancerModel(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, dropout_rate=0.5, model_type='resnet50',
                 use_attention=False, attention_type=None):
        super(CervicalCancerModel, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model_type = model_type
        self.use_attention = use_attention
        self.attention_type = attention_type

        # 根据model_type选择预训练模型
        if model_type == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            # 移除最后的全连接层
            self.backbone.fc = nn.Identity()
        elif model_type == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_type == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:  # 默认使用resnet50
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        # 添加注意力机制
        if use_attention and attention_type:
            if attention_type == 'se':
                self.attention = SEBlock(in_features)
            elif attention_type == 'cbam':
                self.attention = CBAM(in_features)
            else:
                self.attention = None
        else:
            self.attention = None

        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # 提取特征
        x = self.backbone(x)

        # 应用注意力机制
        if self.attention is not None:
            x = self.attention(x)

        # 分类
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 处理不同维度的输入
        if len(x.shape) == 4:  # 4D tensor (B, C, H, W)
            b, c, _, _ = x.size()
            y = torch.nn.functional.adaptive_avg_pool2d(x, 1).view(b, c)
            y = self.fc(y)
            y = y.view(b, c, 1, 1)
            return x * y.expand_as(x)
        elif len(x.shape) == 2:  # 2D tensor (B, C)
            b, c = x.size()
            y = self.fc(x)
            return x * y
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channels = channels
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        # 处理不同维度的输入
        if len(x.shape) == 4:  # 4D tensor (B, C, H, W)
            x = self.ca(x) * x
            x = self.sa(x) * x
            return x
        elif len(x.shape) == 2:  # 2D tensor (B, C)
            # 对于2D输入，只应用通道注意力
            x = self.ca(x) * x
            return x
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.channels = channels
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if len(x.shape) == 4:  # 4D tensor (B, C, H, W)
            b, c, _, _ = x.size()
            avg_out = F.adaptive_avg_pool2d(x, 1).view(b, c)
            max_out = F.adaptive_max_pool2d(x, 1).view(b, c)
            avg_out = self.fc(avg_out)
            max_out = self.fc(max_out)
            out = avg_out + max_out
            return self.sigmoid(out).view(b, c, 1, 1)
        elif len(x.shape) == 2:  # 2D tensor (B, C)
            b, c = x.size()
            # 2D 特征已经是 (B, C)，直接通过全连接处理
            avg_out = self.fc(x)
            max_out = self.fc(x)
            out = avg_out + max_out
            return self.sigmoid(out).view(b, c)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if len(x.shape) == 4:  # 4D tensor (B, C, H, W)
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            x = self.conv1(x)
            return self.sigmoid(x)
        else:
            # 对于非4D张量，直接返回原张量
            return x