import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import get_model as _tv_get_model

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
        elif model_type.startswith('efficientnet'):
            self.backbone = _tv_get_model(
                model_type, weights="DEFAULT" if pretrained else None)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(
                f"未实现的 model_type: {model_type!r}"
                f"（支持 resnet50/resnet101/efficientnet_b0..b7）")

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

    def _features_prepool(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回骨干网络输出的预池化特征
        """
        if self.model_type in ('resnet50', 'resnet101'):
            b = self.backbone
            x = b.conv1(x)
            x = b.bn1(x)
            x = b.relu(x)
            x = b.maxpool(x)
            x = b.layer1(x)
            x = b.layer2(x)
            x = b.layer3(x)
            x = b.layer4(x)
            return x
        if self.model_type.startswith('efficientnet'):
            return self.backbone.features(x)
        raise ValueError(f"CBAM 未支持的后端: {self.model_type!r}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """骨干 + 注意力，返回分类头之前的特征"""
        if self.attention is not None and self.attention_type == 'cbam':
            feat = self._features_prepool(x)
            feat = self.attention(feat)
            return F.adaptive_avg_pool2d(feat, 1).flatten(1) 
        x = self.backbone(x)
        if self.attention is not None:
            x = self.attention(x)
        return x

    def forward_head(self, features: torch.Tensor, dropout_active: bool = False) -> torch.Tensor:
        """分类头,dropout_active=True时开启dropout(用于 MC Dropout 采样)"""
        if dropout_active:
            features = F.dropout(features, p=self.dropout.p, training=True)
        else:
            features = self.dropout(features)
        return self.fc(features)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))

    @staticmethod
    def detect_model_type(state_dict) -> str:
        """从 state_dict 的键名中检测模型类型"""
        sd = state_dict if hasattr(state_dict, 'items') else {}
        keys = set(sd.keys())

        if any('patch_embed' in k or 'blocks.' in k or 'cls_token' in k for k in keys):
            return 'vit'

        if any(k.startswith('features.') for k in keys):
            w = next((v for k, v in sd.items() if k.endswith('classifier.1.weight')), None)
            if w is None:
                raise ValueError("EfficientNet 权重缺少 classifier.1.weight，无法判定型号")
            dim = int(w.shape[1])
            for name, d in (('efficientnet_b0', 1280), ('efficientnet_b2', 1408),
                            ('efficientnet_b3', 1536), ('efficientnet_b4', 1792)):
                if d == dim:
                    return name
            raise ValueError(f"无法识别的 EfficientNet 变体 (in_features={dim})")

        if any(k.startswith('layer1.') for k in keys):
            idx = set()
            for k in keys:
                if k.startswith('layer4.'):
                    parts = k.split('.')
                    try:
                        idx.add(int(parts[1]))
                    except (ValueError, IndexError):
                        continue
            return 'resnet101' if (max(idx) if idx else -1) >= 22 else 'resnet50'

        return 'resnet50'


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
    """
    Convolutional Block Attention Module(通道注意力 + 空间注意力)
    """

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
            # 兼容分支：CBAM 正常路径由 forward_features 传入 4D；
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