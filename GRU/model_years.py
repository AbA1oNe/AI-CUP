# model_years.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path

class GRUClassifier(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])  # raw logits

def load_txt(path):
    seq = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 or line.strip() == "":
                continue
            nums = line.strip().split()
            if len(nums) >= 6:
                seq.append([int(n) for n in nums[:6]])
    return seq

def cut_sequence(seq, cuts):
    return [seq[cuts[i]:cuts[i+1]] for i in range(len(cuts) - 1)]

class YearsDataset(Dataset):
    def __init__(self, info_csv='train_info.csv', data_dir='train_data'):
        self.samples = []
        info = pd.read_csv(info_csv)
        for _, row in info.iterrows():
            uid = row['unique_id']
            try:
                label = int(row['play years'])  # already 0~2
                cuts = list(map(int, row['cut_point'].strip('[]').split()))
            except:
                continue
            path = Path(data_dir) / f"{uid}.txt"
            if not path.exists():
                continue
            seq = load_txt(path)
            swings = cut_sequence(seq, cuts)
            for s in swings:
                self.samples.append((torch.tensor(s, dtype=torch.float32), label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return x, torch.tensor(y)

def collate_fn(batch):
    xs, ys = zip(*batch)
    maxlen = max(len(x) for x in xs)
    padded = torch.zeros(len(xs), maxlen, 6)
    for i, x in enumerate(xs):
        padded[i, :len(x)] = x
    return padded, torch.tensor(ys)

def train():
    dataset = YearsDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    model = GRUClassifier(output_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        total = 0
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch} Loss: {total:.4f}")

    torch.save(model.state_dict(), "model_years.pth")
    print("✅ model_years.pth saved")

if __name__ == '__main__':
    train()
