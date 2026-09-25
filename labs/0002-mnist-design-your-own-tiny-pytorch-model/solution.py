import torch, torch.nn as nn

def build_model() -> nn.Module:
    """
    Return a tiny nn.Module for MNIST classification (10 classes).
    IMPORTANT: If total trainable params > 2048, final accuracy will be set to 0.
    Tip: Consider very small convs, global average pooling, and tiny linear head.
    """
    class TinyNet(nn.Module):
         def __init__(self):
             super().__init__()
             # input N x 1 x 28 x 28
             self.model = nn.Sequential(
                nn.Conv2d(1,2, kernel_size=3, padding=1), # N x 8 x 28 x 28
                nn.ReLU(),
                nn.MaxPool2d(2),  # N x 8 x 14 x 14
                nn.Conv2d(2,4, kernel_size=3, padding=1), # N x 4 x 14 x 14
                nn.ReLU(),
                nn.MaxPool2d(2),  # N x 4 x 7 x 7
                nn.Flatten(),  # N x 784
                nn.Linear(196, 10) # N x 10
             )
         def forward(self, x):
            return self.model(x)
    return TinyNet()
