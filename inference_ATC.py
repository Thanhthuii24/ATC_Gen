import torch
import numpy as np
from diffusers import DDPMScheduler
from unet1d_conditioned import UNet1DConditioned
import matplotlib.pyplot as plt
import pandas as pd
import os

# ----------------------------
# Config
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/content/drive/MyDrive/ATC_/text_to_image/output_ATC/pytorch_model.bin"
csv_path   = "/content/drive/MyDrive/ATC_/text_to_image/dataset_0.1mm_6k_samples_v2_improved.csv"
signal_len = 512
steps      = 100

drive_output_dir = "/content/drive/MyDrive/outputs"
local_output_dir = "outputs"

# ----------------------------
# Load model
# ----------------------------
print("[INFO] Loading model and handling quantizer.levels mismatch...")
model = UNet1DConditioned().to(device)
checkpoint = torch.load(model_path, map_location=device)
if "quantizer.levels" in checkpoint:
    print("[WARNING] Removing 'quantizer.levels' from checkpoint to avoid size mismatch")
    checkpoint.pop("quantizer.levels")
missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
print(f"[INFO] Missing keys: {missing_keys}")
print(f"[INFO] Unexpected keys: {unexpected_keys}")
model.eval()

# ----------------------------
# Scheduler
# ----------------------------
scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
scheduler.set_timesteps(steps)

# ----------------------------
# Read CSV and prepare input
# ----------------------------
df = pd.read_csv(csv_path)
row = df.iloc[0]
cond_vals = row.iloc[:8].to_numpy(dtype=np.float32)
cond = torch.from_numpy(cond_vals).unsqueeze(0).to(device)
print(f"[DEBUG] cond_vals: {cond_vals}")
print(f"[DEBUG] cond.shape: {cond.shape}")

# ----------------------------
# Prepare output folders
# ----------------------------
os.makedirs(local_output_dir, exist_ok=True)
os.makedirs(drive_output_dir, exist_ok=True)

# ----------------------------
# Initial noise
# ----------------------------
x = torch.randn((1,1,signal_len), device=device)
x_initial = x.cpu().numpy()

# ----------------------------
# Denoising Loop
# ----------------------------
with torch.no_grad():
    for idx, t in enumerate(scheduler.timesteps):
        t_tensor = torch.tensor([t], device=device)
        noise_pred = model(x, cond, t_tensor, use_quantizer=False)
        x = scheduler.step(noise_pred, t, x).prev_sample
        if idx % 20 == 0 or idx == len(scheduler.timesteps)-1:
            print(f"[DEBUG] Step {idx+1}/{len(scheduler.timesteps)}, t={t}")

# ----------------------------
# Final Quantization
# ----------------------------
with torch.no_grad():
    x_quant = model(x, cond, t=None, use_quantizer=True)
    x_final = x_quant.cpu().numpy()

print(f"[INFO] Unique values in output: {np.unique(x_final)}")

# ----------------------------
# Save & Visualize
# ----------------------------
# Local
local_signal_path = os.path.join(local_output_dir, "generated_signal.npy")
np.save(local_signal_path, x_final)
# Drive
drive_signal_path = os.path.join(drive_output_dir, "generated_signal.npy")
np.save(drive_signal_path, x_final)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(x_initial[0][0])
plt.title("Noise Input")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(x_final[0][0])
plt.title("Generated Square Wave")
plt.grid(True)

plt.tight_layout()
# Local
plt.savefig(os.path.join(local_output_dir, "comparison.png"))
# Drive
plt.savefig(os.path.join(drive_output_dir, "comparison.png"))

plt.show()
