import os
import math
import torch
from torch import nn
try:
    import wandb
except Exception:
    wandb = None

# 从我们拆分的模块导入
from dataset.dataloader import get_dataloaders
from utils.checkpoint import save_checkpoint

class _NoOpWandb:
    def init(self, *_args, **_kwargs):
        del _args, _kwargs
        return None

    def define_metric(self, *_args, **_kwargs):
        del _args, _kwargs
        return None

    def log(self, *_args, **_kwargs):
        del _args, _kwargs
        return None

    def finish(self, *_args, **_kwargs):
        del _args, _kwargs
        return None


def _get_wandb():
    if wandb is None:
        return _NoOpWandb()

    def has_existing_login():
        if os.environ.get("WANDB_API_KEY"):
            return True
        netrc_path = os.path.expanduser("~/.netrc")
        if os.path.exists(netrc_path):
            try:
                with open(netrc_path, "r", encoding="utf-8") as f:
                    if "api.wandb.ai" in f.read():
                        return True
            except Exception:
                pass
        settings_path = os.path.expanduser("~/.config/wandb/settings")
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if "api_key" in content:
                    return True
            except Exception:
                pass
        return False

    if os.environ.get("WANDB_MODE") is None and not has_existing_login():
        os.environ["WANDB_MODE"] = "offline"
    return wandb


def train_loop(dataloader, model, loss_fn, optimizer, device, wandb_run, max_steps=0, lr_schedule=None, global_step=0):
    size = len(dataloader.dataset)
    model.train()

    seen = 0
    loss_sum = 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        if lr_schedule is not None:
            lr = lr_schedule(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = X.shape[0]
        seen += batch_size
        loss_sum += loss.item() * batch_size
        
        if batch % 100 == 0:
            loss_val = loss.item()
            current = batch * len(X)
            print(f'loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]')
            wandb_run.log({"step": global_step, "train_loss": loss_val, "lr": optimizer.param_groups[0]["lr"]})
        if max_steps and (batch + 1) >= max_steps:
            break
        global_step += 1
    return loss_sum / max(seen, 1), global_step


def eval_loop(dataloader, model, loss_fn, device):
    model.eval()
    seen = 0
    loss_sum = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            batch_size = X.shape[0]
            seen += batch_size
            loss_sum += loss.item() * batch_size

    return loss_sum / max(seen, 1)

def main():
    # 初始化 Wandb (确保你在终端或 Colab 已经跑过 wandb.login())
    wandb_run = _get_wandb()
    wandb_run.init(project='faimdl-lab3')
    wandb_run.define_metric("train_loss", step_metric="step")
    wandb_run.define_metric("lr", step_metric="step")
    wandb_run.define_metric("val_loss", step_metric="step")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # 使用封装好的 DataLoader
    train_loader, eval_loader = get_dataloaders(batch_size=64)

    model_name = os.environ.get("MODEL", "alexnet").lower()
    if model_name == "alexnet":
        from models.alexnet import AlexNet

        model = AlexNet(num_classes=200).to(device)
    elif model_name in {"resnet", "resnet18"}:
        from models.resnet import ResNet18

        model = ResNet18(num_classes=200).to(device)
    elif model_name in {"custom", "customcnn"}:
        from models.customnet import CustomCNN

        model = CustomCNN().to(device)
    else:
        raise ValueError(f"Unknown MODEL={model_name}")
    loss_fn = nn.CrossEntropyLoss()
    
    # 获取环境变量中的超参数
    lr = float(os.environ.get("LR", "1e-3")) # 降低默认学习率到 1e-3
    
    # TinyImageNet 分类任务推荐使用带 momentum 和 weight_decay 的 SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    epochs = int(os.environ.get("EPOCHS", "5"))
    max_steps = int(os.environ.get("MAX_STEPS", "0"))

    warmup_epochs = int(os.environ.get("WARMUP_EPOCHS", "1"))
    min_lr = float(os.environ.get("MIN_LR", "1e-5"))

    steps_per_epoch = len(train_loader)
    if max_steps:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)

    def warmup_cosine(step):
        if warmup_steps > 0 and step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        if total_steps <= warmup_steps:
            return min_lr
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (lr - min_lr) * cosine

    global_step = 0
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}\n-------------------------------')
        train_loss, global_step = train_loop(
            train_loader,
            model,
            loss_fn,
            optimizer,
            device,
            wandb_run,
            max_steps=max_steps,
            lr_schedule=warmup_cosine,
            global_step=global_step,
        )
        val_loss = eval_loop(eval_loader, model, loss_fn, device)
        print(f'val_loss: {val_loss:>7f}')
        wandb_run.log({"step": global_step, "epoch": epoch + 1, "train_loss_epoch": train_loss, "val_loss": val_loss})
        
        # 每一个 Epoch 结束后保存一次 Checkpoint
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        save_checkpoint(checkpoint_state, filename=f'{model_name}_epoch_{epoch+1}.pth')

    # 保存最终模型
    save_checkpoint({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filename=f'{model_name}_final.pth')
    wandb_run.finish()

if __name__ == '__main__':
    main()
