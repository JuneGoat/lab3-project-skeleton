import os
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


def train_loop(dataloader, model, loss_fn, optimizer, device, wandb_run, max_steps=0):
    size = len(dataloader.dataset)
    model.train() # 确保模型处于训练模式
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss_val = loss.item()
            current = batch * len(X)
            print(f'loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]')
            # 记录到 Wandb
            wandb_run.log({"train_loss": loss_val})
        if max_steps and (batch + 1) >= max_steps:
            break

def main():
    # 初始化 Wandb (确保你在终端或 Colab 已经跑过 wandb.login())
    wandb_run = _get_wandb()
    wandb_run.init(project='faimdl-lab3')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # 使用封装好的 DataLoader
    train_loader, _ = get_dataloaders(batch_size=64)

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
    lr = float(os.environ.get("LR", "0.01"))
    
    # 推荐使用 Adam 或带有 Momentum 的 SGD
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = int(os.environ.get("EPOCHS", "5"))
    max_steps = int(os.environ.get("MAX_STEPS", "0"))
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}\n-------------------------------')
        train_loop(train_loader, model, loss_fn, optimizer, device, wandb_run, max_steps=max_steps)
        
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
