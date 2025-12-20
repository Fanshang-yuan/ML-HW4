import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#  类定义
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.conv(self.act(self.norm(x)))

class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.block1 = Block(in_ch, out_ch)
        self.block2 = Block(out_ch, out_ch)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        time_emb = self.mlp(t)
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.block2(h)
        return h + self.res_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(1, 2).view(B, C, H, W)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # Conv -> Norm -> Act
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels), 
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Sequential(
            ResnetBlock(in_channels, out_channels, time_dim),
            ResnetBlock(out_channels, out_channels, time_dim)
        )
    def forward(self, x, t):
        x = self.maxpool(x)
        x = self.conv[0](x, t)
        x = self.conv[1](x, t)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            ResnetBlock(in_channels, out_channels, time_dim),
            ResnetBlock(out_channels, out_channels, time_dim)
        )

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv[0](x, t)
        x = self.conv[1](x, t)
        return x


#  主网络定义(UNet)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=256):
        super().__init__()
        self.time_dim = time_emb_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128, time_emb_dim)
        self.down2 = Down(128, 256, time_emb_dim)
        self.sa1 = SelfAttention(256)
        self.down3 = Down(256, 256, time_emb_dim)
        self.sa2 = SelfAttention(256)

        # Bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # Decoder
        self.up1 = Up(512, 128, time_emb_dim)
        self.sa3 = SelfAttention(128)
        self.up2 = Up(256, 64, time_emb_dim)
        self.sa4 = SelfAttention(64)
        self.up3 = Up(128, 64, time_emb_dim)
        
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x3 = self.sa1(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa2(x4)
        
        # Bottleneck
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        
        # Decoder
        x = self.up1(x4, x3, t)
        x = self.sa3(x)
        x = self.up2(x, x2, t)
        x = self.sa4(x)
        x = self.up3(x, x1, t)
        
        output = self.outc(x)
        return output