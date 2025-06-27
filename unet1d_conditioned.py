
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import DiffusionPipeline, DDPMScheduler
from transformers import PretrainedConfig
import os
# -----------------------------------------------------------------------------
# Utilities & Embeddings
# -----------------------------------------------------------------------------

def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim % 2 == 0, "`dim` must be even for sinusoidal embedding"
    half = dim // 2
    device = timesteps.device

    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)

    exponent = -math.log(10000.0) / (half - 1)
    freqs = torch.exp(torch.arange(half, device=device) * exponent)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    return emb

class FiLM(nn.Module):
    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.scale_shift = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, x, cond):
        scale_shift = self.scale_shift(cond)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        return x * (1 + scale) + shift

# -----------------------------------------------------------------------------
# Square Wave Specific Layers
# -----------------------------------------------------------------------------

class DiscreteQuantizer(nn.Module):
    def __init__(self, levels=[-1.0, -0.9, 0.8], temperature=0.01):
        super().__init__()
        self.register_buffer('levels', torch.tensor(levels, dtype=torch.float32))
        self.temperature = temperature

    def forward(self, x, hard=True):
        B, C, L = x.shape
        quantized = torch.zeros_like(x)
        for i, level in enumerate(self.levels):
            if i == 0:
                mask = x <= (self.levels[0] + self.levels[1]) / 2
            elif i == len(self.levels) - 1:
                mask = x > (self.levels[i-1] + self.levels[i]) / 2
            else:
                mask = (x > (self.levels[i-1] + self.levels[i]) / 2) & \
                       (x <= (self.levels[i] + self.levels[i+1]) / 2)
            quantized[mask] = level
        return quantized

class SquareWaveDetector(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.edge_detector = nn.Conv1d(channels, channels, 5, padding=2)
        self.plateau_detector = nn.Conv1d(channels, channels, 15, padding=7)
        self.level_detector = nn.Conv1d(channels, channels, 7, padding=3)
        self.fusion = nn.Conv1d(channels * 3, channels, 1)
        nn.init.xavier_normal_(self.edge_detector.weight)
        nn.init.xavier_normal_(self.plateau_detector.weight)

    def forward(self, x):
        edges = torch.tanh(self.edge_detector(x))
        plateaus = torch.sigmoid(self.plateau_detector(x))
        levels = torch.tanh(self.level_detector(x))
        combined = torch.cat([edges, plateaus, levels], dim=1)
        return self.fusion(combined)

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

    def forward(self, x, cond):
        residual = self.residual(x)
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.film1(h, cond)
        h = self.activation(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.film2(h, cond)
        h = self.activation(h)
        h = self.square_detector(h)
        return h + residual

class MultiScaleAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(min(32, channels//4), channels)

    def forward(self, x):
        B, C, L = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, L)
        q, k, v = qkv.unbind(1)
        q = torch.clamp(q, -10.0, 10.0)
        k = torch.clamp(k, -10.0, 10.0)
        v = torch.clamp(v, -10.0, 10.0)
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * scale
        attn = torch.clamp(attn, -10.0, 10.0)
        attn = F.softmax(attn, dim=-1)
        if torch.isnan(attn).any():
            attn = torch.eye(L, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0).repeat(B, self.num_heads, 1, 1)
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)
        out = out.reshape(B, C, L)
        out = self.proj(out)
        return self.norm(out + x)

# -----------------------------------------------------------------------------
# SPECIALIZED SQUARE WAVE U-NET ARCHITECTURE
# -----------------------------------------------------------------------------

class UNet1DSquareWave(nn.Module):
    def __init__(self, cond_dim=8, emb_dim=128, base_channels=64):
        super().__init__()
        self.emb_dim = emb_dim
        self.cond_dim = cond_dim
        self.base_channels = base_channels

        self.cond_proj = nn.Linear(cond_dim, emb_dim)
        self.time_proj = nn.Linear(emb_dim, emb_dim)
        nn.init.xavier_uniform_(self.cond_proj.weight, gain=0.02)
        nn.init.constant_(self.cond_proj.bias, 0)
        nn.init.xavier_uniform_(self.time_proj.weight, gain=0.02)
        nn.init.constant_(self.time_proj.bias, 0)

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
        self.dec3 = EnhancedConvBlock(base_channels*8 + base_channels*8, base_channels*4, emb_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec2 = EnhancedConvBlock(base_channels*4 + base_channels*4, base_channels*2, emb_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec1 = EnhancedConvBlock(base_channels*2 + base_channels*2, base_channels, emb_dim)

        self.pre_output = nn.Sequential(
            nn.Conv1d(base_channels, base_channels//2, 3, padding=1),
            nn.GroupNorm(8, base_channels//2),
            nn.SiLU(),
            nn.Conv1d(base_channels//2, base_channels//4, 3, padding=1),
            nn.GroupNorm(4, base_channels//4),
            nn.SiLU()
        )

        self.output_conv = nn.Conv1d(base_channels//4, 1, 3, padding=1)
        self.quantizer = DiscreteQuantizer(levels=[-1.0, -0.9, 0.8], temperature=0.01)

    def _get_combined_embedding(self, cond, t):
        if torch.isnan(cond).any():
            print("NaN detected in conditions, replacing with zeros")
            cond = torch.zeros_like(cond)
        cond = torch.clamp(cond, -10.0, 10.0)
        cond_emb = self.cond_proj(cond)
        if t is None:
            t = torch.zeros(cond.shape[0], dtype=torch.long, device=cond.device)
        t = torch.clamp(t, 0, 999)
        t_emb = timestep_embedding(t, self.emb_dim).to(cond.dtype)
        t_emb = self.time_proj(t_emb)
        if torch.isnan(cond_emb).any() or torch.isnan(t_emb).any():
            print("NaN in embeddings, using fallback")
            return torch.zeros_like(cond_emb)
        return cond_emb + t_emb
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def forward(self, x, cond, t=None, use_quantizer=True):
        original_length = x.shape[-1]
        cond_emb = self._get_combined_embedding(cond, t)
        x = self.input_conv(x)
        x = self.input_square_detector(x)
        skip1 = self.enc1(x, cond_emb)
        x = self.down1(skip1)
        skip2 = self.enc2(x, cond_emb)
        x = self.down2(skip2)
        skip3 = self.enc3(x, cond_emb)
        x = self.down3(skip3)

        for layer in self.bottleneck:
            if isinstance(layer, EnhancedConvBlock):
                x = layer(x, cond_emb)
            else:
                x = layer(x)

        cond_proj = self.cond_to_bottleneck(cond_emb).unsqueeze(1)
        x_perm = x.permute(0, 2, 1)
        x_attn, _ = self.cross_attn(x_perm, cond_proj, cond_proj)
        x = (x_perm + x_attn).permute(0, 2, 1)

        x = self.up3(x)
        if x.shape[-1] != skip3.shape[-1]:
            skip3 = F.interpolate(skip3, size=x.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec3(x, cond_emb)

        x = self.up2(x)
        if x.shape[-1] != skip2.shape[-1]:
            skip2 = F.interpolate(skip2, size=x.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2(x, cond_emb)

        x = self.up1(x)
        if x.shape[-1] != skip1.shape[-1]:
            skip1 = F.interpolate(skip1, size=x.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec1(x, cond_emb)

        x = self.pre_output(x)
        x = self.output_conv(x)

        if use_quantizer and not self.training:
            x = self.quantizer(x, hard=True)
            # Ensure output length matches input
        if x.shape[-1] != original_length:
            x = F.interpolate(x, size=original_length, mode='linear', align_corners=False)

        return x

class UNet1DSimple(UNet1DSquareWave):
    pass

class UNet1DConditioned(UNet1DSquareWave):
    pass


# class SignalDiffusionConfig(PretrainedConfig):
#     model_type = "signal-diffusion"

#     def __init__(self, cond_dim=8, emb_dim=128, base_channels=64, **kwargs):
#         super().__init__(**kwargs)
#         self.cond_dim = cond_dim
#         self.emb_dim = emb_dim
#         self.base_channels = base_channels

# class SignalDiffusionPipeline(DiffusionPipeline):
#     """
#     A pipeline for 1D-signal diffusion using your custom UNet1DConditioned.
#     """
#     config_class = SignalDiffusionConfig

#     def __init__(self, unet: UNet1DConditioned, scheduler: DDPMScheduler):
#         super().__init__()
#         self.register_modules(
#           unet=unet,
#           scheduler=scheduler,
#         )
#         self.unet = unet
#         self.scheduler = scheduler


#     @classmethod
#     def from_pretrained(
#         cls,
#         pretrained_model_name_or_path: str,
#         unet_class=UNet1DConditioned,
#         cond_dim: int = 8,
#         emb_dim: int = 128,
#         base_channels: int = 64,
#         scheduler_config=None,
#         **kwargs,
#     ):

#       # Load config
#         config = SignalDiffusionConfig.from_pretrained(
#           pretrained_model_name_or_path,
#           cond_dim=cond_dim,
#           emb_dim=emb_dim,
#           base_channels=base_channels,
#           local_files_only=True,  # ✅ THÊM DÒNG NÀY
#           **kwargs,
#       )


#         #  Instantiate UNet và load weights
#         unet = unet_class(
#             cond_dim=config.cond_dim,
#             emb_dim=config.emb_dim,
#             base_channels=config.base_channels,
#         )
#         unet_state = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
#         unet.load_state_dict(torch.load(unet_state, map_location="cpu"), strict=False)

#         # Scheduler
#         if scheduler_config is None:
#             scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path)
#         else:
#             scheduler = DDPMScheduler(**scheduler_config)

#         return cls(unet=unet, scheduler=scheduler)

#     def save_pretrained(self, save_directory: str):
#         os.makedirs(save_directory, exist_ok=True)
#         # --- Nếu muốn vẫn có file config.json, tự ghi bằng json.dump() ---
#         # cfg = {
#         #     "cond_dim": self.unet.cond_dim,
#         #     "emb_dim": self.unet.emb_dim,
#         #     "base_channels": self.unet.base_channels,
#         # }
#         # import json
#         # with open(os.path.join(save_directory, "config.json"), "w") as f:
#         #     json.dump(cfg, f, indent=2)

#         # Save UNet weights
#         torch.save(self.unet.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
#         # Save scheduler
#         self.scheduler.save_pretrained(save_directory)

# # — là đủ để khởi tạo lại model + scheduler.
# # # from unet1d_conditioned import SignalDiffusionPipeline
# # pipeline = SignalDiffusionPipeline.from_pretrained("path/to/output_dir")
