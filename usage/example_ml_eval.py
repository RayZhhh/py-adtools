import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x8x8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class Trainer:
    def __init__(
        self,
        data_root: str = "./data",
        batch_size: int = 128,
        epochs: int = 5,
        lr: float = 1e-3,
        num_workers: int = 4,
    ):
        mps_available = torch.mps.is_available()
        cuda_available = torch.cuda.is_available()
        self.device = torch.device(
            "mps" if mps_available else "cuda" if cuda_available else "cpu"
        )
        self.epochs = epochs

        # 1️⃣ build model
        self.model = SimpleCNN()

        # 2️⃣ build dataset & dataloader
        self.train_loader = self._build_dataloader(data_root, batch_size, num_workers)

        # 3️⃣ loss & optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def _build_dataloader(
        self,
        data_root: str,
        batch_size: int,
        num_workers: int,
    ) -> DataLoader:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transform,
        )
        dataset.data = dataset.data[:50]
        dataset.targets = dataset.targets[:50]

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def train_one_epoch(self, epoch: int):
        self.model.to(self.device)
        self.model.train()

        total_loss = 0.0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch [{epoch}/{self.epochs}] - Loss: {avg_loss:.4f}")

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.train_one_epoch(epoch)


if __name__ == "__main__":
    from adtools.sandbox import SandboxExecutor, SandboxExecutorRay

    trainer = SandboxExecutor(Trainer(), debug_mode=True)
    res = trainer.secure_execute("train", timeout_seconds=20)
    print(res)

    trainer = SandboxExecutorRay(Trainer(), debug_mode=True)
    res = trainer.secure_execute("train", timeout_seconds=20)
    print(res)
