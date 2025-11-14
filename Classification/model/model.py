import torch
import torch.nn as nn
import torchvision


# 假设 PPEG 是你自定义的空间位置编码模块
class PPEG(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, dim, 1, 1))

    def forward(self, x):
        return x + self.pos_embed


# 视觉模态编码器：ResNet18 + Encoder + PPEG
class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 去掉FC层
        self.encoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            PPEG(256)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.encoder(x)
        return x


# 频域模态编码器：FFT + CNN + Encoder（不含PPEG）
class FrequencyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),  # 3通道 × (实部+虚部)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        fft_real = []
        fft_imag = []
        for c in range(x.shape[1]):
            x_fft = torch.fft.fft2(x[:, c, :, :])
            fft_real.append(x_fft.real.unsqueeze(1))
            fft_imag.append(x_fft.imag.unsqueeze(1))
        x_freq = torch.cat(fft_real + fft_imag, dim=1)  # [B, 6, H, W]
        x = self.cnn(x_freq)
        x = self.encoder(x)
        return x


# Cross Attention 融合模块
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)

    def forward(self, visual_feat, freq_feat):
        v = visual_feat.flatten(2).permute(0, 2, 1)
        f = freq_feat.flatten(2).permute(0, 2, 1)
        fused, _ = self.attn(v, f, f)
        return fused


# 解码器：视觉不含PPEG，频域含PPEG
class VisualDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.decoder(x)


class FrequencyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            PPEG(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.decoder(x)


# MLP Alignment + Fusion + Softmax
class AlignmentFusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.align_visual = nn.Linear(128, 64)
        self.align_freq = nn.Linear(128, 64)
        self.fusion = nn.Linear(128, 2)

    def forward(self, v_feat, f_feat):
        v = self.align_visual(v_feat.mean(dim=[2, 3]))
        f = self.align_freq(f_feat.mean(dim=[2, 3]))
        fused = torch.cat([v, f], dim=1)
        out = self.fusion(fused)
        return torch.softmax(out, dim=1)


# 总体模型封装
class StampClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = VisualEncoder()
        self.freq_encoder = FrequencyEncoder()
        self.cross_attention = CrossAttentionFusion(dim=256)
        self.visual_decoder = VisualDecoder()
        self.freq_decoder = FrequencyDecoder()
        self.head = AlignmentFusionHead()

    def forward(self, x):
        v_enc = self.visual_encoder(x)
        f_enc = self.freq_encoder(x)
        fused = self.cross_attention(v_enc, f_enc)
        v_dec = self.visual_decoder(v_enc)
        f_dec = self.freq_decoder(f_enc)
        out = self.head(v_dec, f_dec)
        return out
























