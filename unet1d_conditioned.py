import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
import os

def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim % 2 == 0, "`dim` must be even for sinusoidal embedding"
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=timesteps.device) *
                      (-math.log(10000.0) / (half - 1)))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    print(f"[DEBUG] timestep_embedding: output shape={emb.shape}")
    return emb

class FiLM(nn.Module):
    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.scale_shift = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, x, cond):
        print(f"[DEBUG] FiLM input: x shape={x.shape}, cond shape={cond.shape}")
        scale_shift = self.scale_shift(cond)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        out = x * (1 + scale) + shift
        print(f"[DEBUG] FiLM output shape={out.shape}")
        return out

class DiscreteQuantizer(nn.Module):
    def __init__(self, levels=[-1.0, 1.0], temperature=0.01):
        super().__init__()
        self.register_buffer('levels', torch.tensor(levels, dtype=torch.float32))
        self.temperature = temperature
        print(f"[DEBUG] DiscreteQuantizer init levels={levels}")

    def forward(self, x, hard=True):
        print(f"[DEBUG] Quantizer input shape={x.shape}")
        quantized = torch.zeros_like(x)
        for i, level in enumerate(self.levels):
            if i == 0:
                mask = x <= (self.levels[0] + self.levels[1]) / 2
            else:
                mask = x > (self.levels[i-1] + self.levels[i]) / 2
            quantized[mask] = level
        print(f"[DEBUG] Quantizer output unique levels={torch.unique(quantized)}")
        return quantized


class SquareWaveDetector(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.edge_detector = nn.Conv1d(channels, channels, 5, padding=2)
        self.plateau_detector = nn.Conv1d(channels, channels, 15, padding=7)
        self.level_detector = nn.Conv1d(channels, channels, 7, padding=3)
        self.fusion = nn.Conv1d(channels * 3, channels, 1)
        print(f"[DEBUG] SquareWaveDetector init channels={channels}")

    def forward(self, x):
        edges = torch.tanh(self.edge_detector(x))
        plateaus = torch.sigmoid(self.plateau_detector(x))
        levels = torch.tanh(self.level_detector(x))
        combined = torch.cat([edges, plateaus, levels], dim=1)
        out = self.fusion(combined)
        print(f"[DEBUG] SquareWaveDetector output shape={out.shape}")
        return out

# -----------------------------------------------------------------------------
# EnhancedConvBlock with FiLM and SquareWaveDetector
# -----------------------------------------------------------------------------
class EnhancedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, kernel=3, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2 * dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(min(32, out_ch//4), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=kernel//2 * dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(min(32, out_ch//4), out_ch)
        self.film1 = FiLM(cond_dim, out_ch)
        self.film2 = FiLM(cond_dim, out_ch)
        self.activation = nn.SiLU()
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.square_detector = SquareWaveDetector(out_ch)
        print(f"[DEBUG] EnhancedConvBlock init in_ch={in_ch}, out_ch={out_ch}")

    def forward(self, x, cond):
        print(f"[DEBUG] EnhancedConvBlock input x shape={x.shape}")
        res = self.residual(x)
        h1 = self.activation(self.film1(self.norm1(self.conv1(x)), cond))
        h2 = self.activation(self.film2(self.norm2(self.conv2(h1)), cond))
        h3 = self.square_detector(h2)
        out = h3 + res
        print(f"[DEBUG] EnhancedConvBlock output shape={out.shape}")
        return out

# -----------------------------------------------------------------------------
# MultiScaleAttention for long-range dependencies
# -----------------------------------------------------------------------------
class MultiScaleAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(min(32, channels//4), channels)
        print(f"[DEBUG] MultiScaleAttention init channels={channels}, heads={num_heads}")

    def forward(self, x):
        B, C, L = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, L)
        q, k, v = qkv.unbind(1)
        # clamp to avoid extreme values
        q = torch.clamp(q, -10.0, 10.0)
        k = torch.clamp(k, -10.0, 10.0)
        v = torch.clamp(v, -10.0, 10.0)
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * scale
        attn = torch.clamp(attn, -10.0, 10.0)
        attn = F.softmax(attn, dim=-1)
        if torch.isnan(attn).any():
            attn = torch.eye(L, device=x.device, dtype=x.dtype)\
                   .unsqueeze(0).unsqueeze(0).repeat(B, self.num_heads, 1, 1)
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)
        out = out.reshape(B, C, L)
        out = self.proj(out)
        out = self.norm(out + x)
        print(f"[DEBUG] MultiScaleAttention output shape={out.shape}")
        return out

# -----------------------------------------------------------------------------
# UNet1DSquareWave with detailed debug
# -----------------------------------------------------------------------------
class UNet1DSquareWave(nn.Module):
    def __init__(self, cond_dim=8, emb_dim=128, base_channels=64):
        super().__init__()
        self.cond_dim, self.emb_dim, self.base_channels = cond_dim, emb_dim, base_channels
        print(f"[DEBUG] UNet1DSquareWave init cond_dim={cond_dim}, emb_dim={emb_dim}, base_channels={base_channels}")

        self.cond_proj = nn.Linear(cond_dim, emb_dim)
        self.time_proj = nn.Linear(emb_dim, emb_dim)
        self.input_conv = nn.Conv1d(1, base_channels, 7, padding=3)
        self.input_square_detector = SquareWaveDetector(base_channels)

        self.enc1 = EnhancedConvBlock(base_channels, base_channels*2, emb_dim)
        self.down1 = nn.AvgPool1d(2)
        self.enc2 = EnhancedConvBlock(base_channels*2, base_channels*4, emb_dim)
        self.down2 = nn.AvgPool1d(2)
        self.enc3 = EnhancedConvBlock(base_channels*4, base_channels*8, emb_dim)
        self.down3 = nn.AvgPool1d(2)

        self.bottleneck = nn.Sequential(
            EnhancedConvBlock(base_channels*8, base_channels*8, emb_dim),
            MultiScaleAttention(base_channels*8, num_heads=8),
            EnhancedConvBlock(base_channels*8, base_channels*8, emb_dim)
        )

        self.cond_to_bottleneck = nn.Linear(emb_dim, base_channels*8)
        self.cross_attn = nn.MultiheadAttention(embed_dim=base_channels*8, num_heads=8, batch_first=True)

        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec3 = EnhancedConvBlock(base_channels*16, base_channels*4, emb_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec2 = EnhancedConvBlock(base_channels*8, base_channels*2, emb_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec1 = EnhancedConvBlock(base_channels*4, base_channels, emb_dim)

        self.pre_output = nn.Sequential(
            nn.Conv1d(base_channels, base_channels//2, 3, padding=1),
            nn.GroupNorm(8, base_channels//2), nn.SiLU(),
            nn.Conv1d(base_channels//2, base_channels//4, 3, padding=1),
            nn.GroupNorm(4, base_channels//4), nn.SiLU()
        )
        self.output_conv = nn.Conv1d(base_channels//4, 1, 3, padding=1)
        self.quantizer = DiscreteQuantizer(levels=[-1.0, 1.0])

    def _get_combined_embedding(self, cond, t=None):
        print(f"[DEBUG] cond input shape={cond.shape}, t={t}")
        cond = torch.clamp(cond, -10, 10)
        cond_emb = self.cond_proj(cond)
        if t is None:
            t = torch.zeros(cond.size(0), dtype=torch.long, device=cond.device)
        t_emb = self.time_proj(timestep_embedding(t, self.emb_dim))
        emb = cond_emb + t_emb
        print(f"[DEBUG] combined embedding shape={emb.shape}")
        return emb

    def forward(self, x, cond, t=None, use_quantizer=True):
        print(f"[DEBUG] Forward start: x={x.shape}, cond={cond.shape}, t={t}")
        orig_len = x.size(-1)
        cond_emb = self._get_combined_embedding(cond, t)

        x1 = self.input_conv(x); print(f"[DEBUG] after input_conv: {x1.shape}")
        x1 = self.input_square_detector(x1)
        s1 = self.enc1(x1, cond_emb); x2 = self.down1(s1)
        s2 = self.enc2(x2, cond_emb); x3 = self.down2(s2)
        s3 = self.enc3(x3, cond_emb); x4 = self.down3(s3)

        y = x4
        for layer in self.bottleneck:
            y = layer(y, cond_emb) if isinstance(layer, EnhancedConvBlock) else layer(y)
        print(f"[DEBUG] after bottleneck: {y.shape}")

        proj = self.cond_to_bottleneck(cond_emb).unsqueeze(1)
        attn_out, _ = self.cross_attn(y.permute(0,2,1), proj, proj)
        y = (y.permute(0,2,1) + attn_out).permute(0,2,1)
        print(f"[DEBUG] after cross_attn: {y.shape}")

        u3 = self.up3(y)
        u3 = torch.cat([u3, F.interpolate(s3, size=u3.size(-1), mode='linear')], dim=1)
        d3 = self.dec3(u3, cond_emb)
        u2 = self.up2(d3)
        u2 = torch.cat([u2, F.interpolate(s2, size=u2.size(-1), mode='linear')], dim=1)
        d2 = self.dec2(u2, cond_emb)
        u1 = self.up1(d2)
        u1 = torch.cat([u1, F.interpolate(s1, size=u1.size(-1), mode='linear')], dim=1)
        d1 = self.dec1(u1, cond_emb)

        out = self.pre_output(d1)
        out = self.output_conv(out)
        # if use_quantizer and not self.training:
        #     out = self.quantizer(out)
        if out.size(-1) != orig_len:
            out = F.interpolate(out, size=orig_len, mode='linear')
        print(f"[DEBUG] Forward end output shape={out.shape}")
        return out

class UNet1DConditioned(UNet1DSquareWave):
    pass
