import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
from safetensors.torch import load_file
from Pipeline_ATC_sf import StableDiffusion1DPipeline
from ATC_thui.unet1d_conditioned import UNet1DSquareWave

SUPPORTED_EXT = [
    (".safetensors", load_file),
    (".sa",         load_file),
    (".pt",         torch.load),
    (".pth",        torch.load),
    (".bin",        torch.load),
    (".ptkl",       torch.load),
]

def find_checkpoint(path: str):
    # Nếu path là file, dùng luôn
    if os.path.isfile(path):
        return path
    # Nếu path là thư mục, scan file theo ưu tiên
    for ext, _ in SUPPORTED_EXT:
        for fname in os.listdir(path):
            if fname.endswith(ext):
                return os.path.join(path, fname)
    raise FileNotFoundError(f"No checkpoint with supported extensions in {path}")

def load_checkpoint(model, ckpt_path: str, device):
    ckpt_file = find_checkpoint(ckpt_path)
    ext = os.path.splitext(ckpt_file)[1]
    # Tìm loader tương ứng
    loader = None
    for e, fn in SUPPORTED_EXT:
        if e == ext:
            loader = fn
            break
    if loader is None:
        raise ValueError(f"Unsupported checkpoint extension: {ext}")
    # Load
    state = loader(ckpt_file, map_location=device) if loader is torch.load else loader(ckpt_file, device=device)
    # Nếu state là dict chứa state_dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model

def load_latents_from_csv(csv_path, signal_length=512):
    arr = pd.read_csv(csv_path, header=None).values.astype(np.float32)
    if arr.ndim == 1:
        arr = arr[:signal_length]
        lat_np = arr.reshape(1, 1, -1)
    elif arr.ndim == 2 and arr.shape[1] == signal_length:
        lat_np = arr.reshape(1, 1, signal_length)
    else:
        raise ValueError(f"CSV shape {arr.shape} invalid")
    return torch.from_numpy(lat_np)

def plot_signal(signal, title="Generated Signal", save_path=None):
    plt.figure(figsize=(8,3))
    plt.plot(signal.flatten(), linewidth=1)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def main():
    # 1. Config
    ckpt_folder   = "checkpoints/"           # Thư mục chứa .bin, .ptkl, .sa...
    csv_path      = "input_latents.csv"
    signal_length = 512
    num_steps     = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load latents
    latents = load_latents_from_csv(csv_path, signal_length).to(device)

    # 3. Init UNet & load ckpt
    unet = UNet1DSquareWave(cond_dim=8, emb_dim=128, base_channels=64)
    unet = load_checkpoint(unet, ckpt_folder, device)
    unet.to(device).eval()

    # 4. Scheduler & Pipeline
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    pipe = StableDiffusion1DPipeline(unet=unet, scheduler=scheduler).to(device)

    # 5. Inference
    with torch.no_grad():
        out = pipe(
            latents=latents,
            signal_length=signal_length,
            num_inference_steps=num_steps,
            timesteps=None,
            sigmas=None,
            output_type="np",
        )
    generated = out["signals"]

    # 6. Plot
    plot_signal(generated, title="Generated Square Wave", save_path="gen_sqwave.png")

if __name__ == "__main__":
    main()
