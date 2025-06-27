from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

class MySignalDataset(Dataset):
    def __init__(self, folder_path):   # đổi từ 'path' → 'folder_path'
        self.df = pd.read_csv(folder_path)
        # Chỉ lấy các cột bắt đầu bằng "signal_"
        self.signal_columns = [col for col in self.df.columns if col.startswith("signal_")]
        self.column_names = ["pixel_values", "text"]  # Cho Diffusers trainer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        signal = self.df.loc[idx, self.signal_columns].values.astype(np.float32)

        # Vẽ waveform thành ảnh
        fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
        ax.plot(signal, linewidth=2)
        ax.set_axis_off()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        buf.seek(0)
        image = Image.open(buf).convert("RGB")

        return {
            "pixel_values": image,
            "text": "a signal waveform"
        }
