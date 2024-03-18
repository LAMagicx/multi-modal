import torch
import time
from torch import nn
from tqdm import tqdm

def get_default_device():
    """Pick GPU if available, else CPU"""
    """ 3 things:
    1. Connected to Nvidia GPU
    2. Cuda drivers
    3. Pytorch suitable to GPU version
    then torch.cuda.is_available is True
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_epoch(model, train_loader, optimizer, lr_scheduler, step, device):
    loss_meter = AvgMeter()
    tqdm_obj = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_obj:
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        try:
            loss = model(batch)
        except Exception as e:
            print(str(e))
            print(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 'batch':
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_obj.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter

def train(model: nn.Module, dl: torch.utils.data.DataLoader, epochs: int = 5, model_name: str = "model.pt", lr: float = 1e-3, wd: float = 1e-3):
    device = get_default_device()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    step = 'epoch'

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = train_epoch(model, dl, optimizer, lr_scheduler, step, device)
        model.eval()
        epoch_loss = train_loss.avg
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

    torch.save(model.state_dict(), model_name)
