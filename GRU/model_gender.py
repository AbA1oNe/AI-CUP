# model_gender.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path

class GRUClassifier(nn.Module):
    def __init__(self, input_size=6, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return torch.sigmoid(self.fc(h[-1]))

def load_txt(filepath):
    seq = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i == 0 or line.strip() == "":
                continue
            nums = line.strip().split()
            if len(nums) >= 6:
                seq.append([int(n) for n in nums[:6]])
    return seq

def cut_sequence(seq, cut_points):
    return [seq[cut_points[i]:cut_points[i+1]] for i in range(len(cut_points) - 1)]

class GenderDataset(Dataset):
    def __init__(self, info_csv='train_info.csv', data_dir='train_data'):
        self.samples = []
        info = pd.read_csv(info_csv)
        for _, row in info.iterrows():
            uid = row['unique_id']
            try:
                cuts = list(map(int, row['cut_point'].strip('[]').split()))
                gender = 1 if int(row['gender']) != 0 else 0

            except:
                continue
            path = Path(data_dir) / f"{uid}.txt"
            if not path.exists():
                continue
            seq = load_txt(path)
            swings = cut_sequence(seq, cuts)
            for swing in swings:
                self.samples.append((torch.tensor(swing, dtype=torch.float32), float(gender)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return x, torch.tensor(y, dtype=torch.float32)


def collate_fn(batch):
    x_list, y_list = zip(*batch)
    max_len = max(len(x) for x in x_list)
    padded = torch.zeros(len(x_list), max_len, 6)
    for i, x in enumerate(x_list):
        padded[i, :len(x), :] = x
    return padded, torch.tensor(y_list)

def train():
    dataset = GenderDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = GRUClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    for epoch in range(10):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            out = model(xb).squeeze()
            loss = loss_fn(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), 'model_gender.pth')
    print("✅ model_gender.pth 已儲存")

if __name__ == '__main__':
    train()
