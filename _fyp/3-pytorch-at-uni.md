---
title: "Running PyTorch At University"
excerpt: "Getting PyTorch up and running on the lab PCs"
collection: fyp
permalink: /fyp/pytorch-at-uni
---

In order to run PyTorch on the University Lab PCs, you will first need to log into Linux. Be sure to use your network username (not your email). See the following commands to see how to make a Python virtual environment (you need to use a custom script).

We provide a script `mkvenv` that will create the environment for you. You just have to provide a name for the environment. Below, we use the convention of calling it `.venv`.

```bash
# Create directory
cd ~/Documents/
mkdir pytorch-test
cd pytorch-test

# Create a virtual environment
mkvenv .venv

# Activate virtual environment
source .venv/bin/activate

# Install Pytorch
python3 -m pip install torch torchvision
```

Now that the python environment is set up, we want to check that Pytorch is working. Copy the following Python code and run it with the environment activated.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)   # output 6x24x24 for 28x28 input
        self.pool  = nn.AvgPool2d(kernel_size=2, stride=2)                 # avg pool as original
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)  # output 16x8x8
        self.fc1   = nn.Linear(256, 120)   # note: depending on input size adjust flatten size
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool(x)
        x = F.tanh(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    print(f"Device={DEVICE}")
    x = torch.tensor([1, 2, 3]).to(DEVICE)
    y = torch.tensor([9, 8, 7]).to(DEVICE)

    z = x + y
    print(z)

    model = LeNet(10).to(DEVICE)
    input = torch.randn(64, 1, 28, 28).to(DEVICE)
    output = model(input)
    print(output.shape, output.device)
```
If all works as intended, you should see the following output:
```text
Device=cuda
tensor([10, 10, 10], device='cuda:0')
torch.Size([64, 10]) cuda:0
```
