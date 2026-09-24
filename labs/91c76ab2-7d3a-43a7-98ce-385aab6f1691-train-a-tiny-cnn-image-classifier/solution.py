import torch
import torch.nn as nn
import torch.optim as optim


class TinyCNN(nn.Module):
    """Small CNN: Conv2d -> ReLU -> pool -> (optional extras) -> Linear."""

    def __init__(self, img_size=8, n_classes=2):
        super().__init__()
        # TODO: define conv, activation, pooling, and classifier layers
        # Input shape: (N, 1, img_size, img_size)
        self.model = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1), # N, 10, S, S
            nn.ReLU(),                                  # N, 10, S, S
            nn.AvgPool2d(2),                            # N, 10, S//2, S//2 
            nn.Conv2d(10, 30, kernel_size=3, padding=1),# N, 30, S//2, S//2 
            nn.ReLU(),                                  # N, 30, S//2, S//2 
            nn.AvgPool2d(2),                            # N, 30, S//4, S//4 
            nn.Flatten(),                               # N, 30 * S//4 * S//4 
            nn.Linear(30 * (img_size // 4) * (img_size // 4), n_classes)
        )

    def forward(self, x):
        # TODO: implement forward pass; return logits of shape (N, n_classes)
        return self.model(x)


def build_model(img_size=8, n_classes=2):
    """Return an instance of your TinyCNN (or equivalent nn.Module)."""
    # TODO: return TinyCNN(img_size=img_size, n_classes=n_classes)
    return TinyCNN(img_size=img_size, n_classes=n_classes)


def train_model(model, train_x, train_y, epochs=15, lr=0.01, batch_size=32, seed=0):
    """Train model on train_x/train_y and return the trained model.

    Args:
        model: nn.Module from build_model
        train_x: FloatTensor (N, 1, H, W)
        train_y: LongTensor (N,)
        epochs: number of full passes over the data
        lr: optimizer learning rate
        batch_size: mini-batch size
        seed: RNG seed for shuffling / init determinism

    Returns:
        Trained model (same instance is fine).
    """
    # TODO:
    # - torch.manual_seed(seed)
    # - CrossEntropyLoss + Adam (or SGD)
    # - mini-batch loop for `epochs` epochs
    # - model.train(); zero_grad -> forward -> loss -> backward -> step
    torch.manual_seed(seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(train_x.shape[0])
        x = train_x[perm]
        y = train_y[perm]
        for i in range(0, train_x.shape[0], batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    return model


