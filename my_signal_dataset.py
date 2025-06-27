# my_signal_dataset.py

from torch.utils.data import Dataset
import pandas as pd
import torch

class MySignalDataset(Dataset):
    def __init__(self, csv_path, signal_prefix="signal_", cond_prefix="cond_"):
        self.df = pd.read_csv(csv_path)
        self.signal_columns = [col for col in self.df.columns if col.startswith(signal_prefix)]
        self.cond_columns = [col for col in self.df.columns if col.startswith(cond_prefix)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        signal = self.df.iloc[idx][self.signal_columns].values.astype("float32")
        cond = self.df.iloc[idx][self.cond_columns].values.astype("float32")

        return {
            "pixel_values": torch.from_numpy(signal),  
            "cond": torch.from_numpy(cond)            
        }
