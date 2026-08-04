"""Demo: run the full qualkit report on an INT8-quantized EuroSAT ResNet18.

Trains a quick reference model (or loads one), fake-quantizes it to 4 bits to
make the checks interesting, and writes qualkit_report.{json,md}.

Run:  python3 examples/eurosat_demo.py
"""
import copy
import sys

import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, ".")
from qualkit import report

BITS = 4  # deliberately aggressive so the report shows a failure mode
dev = "cuda" if torch.cuda.is_available() else "cpu"

norm = tv.transforms.Normalize([0.3444, 0.3803, 0.4078], [0.0914, 0.0651, 0.0552])
tf = tv.transforms.Compose([tv.transforms.ToTensor(), norm])
full = tv.datasets.EuroSAT(root="data", download=True)
n = len(full)
g = torch.Generator().manual_seed(0)
i_tr, i_cal, i_probe = random_split(range(n), [n - 6000, 4000, 2000], generator=g)


class Sub(torch.utils.data.Dataset):
    def __init__(self, idxs):
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, i):
        x, y = full[self.idxs[i]]
        return tf(x), y


model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 10)
model = model.to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
l_tr = DataLoader(Sub(i_tr), batch_size=256, shuffle=True, num_workers=4)
for epoch in range(3):  # quick demo training
    model.train()
    for x, y in l_tr:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()

deploy = copy.deepcopy(model)
with torch.no_grad():  # fake-quantize weights to BITS
    for m in deploy.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.data
            qm = 2 ** (BITS - 1) - 1
            sc = w.abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp(min=1e-8) / qm
            m.weight.data = torch.round(w / sc).clamp(-qm - 1, qm) * sc

l_cal = DataLoader(Sub(i_cal), batch_size=256, num_workers=4)
l_probe = DataLoader(Sub(i_probe), batch_size=256, num_workers=4)
r = report.run_all(model.cpu(), deploy.cpu(), l_cal, l_probe,
                   alpha=0.10, seu_bits=BITS, device=dev)
pj, pm = report.save(r)
print(open(pm).read())
