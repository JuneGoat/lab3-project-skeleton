import os
import torch
from torch import nn
try:
    import wandb
except Exception:
    wandb = None

# 从我们拆分的模块导入
from models.customnet import CustomCNN
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
    if os.environ.get("WANDB_MODE") is None and os.environ.get("WANDB_API_KEY") is None:
        os.environ["WANDB_MODE"] = "offline"
    return wandb


def train_loop(dataloader, model, loss_fn, optimizer, device, wandb_run):
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

def main():
    # 初始化 Wandb (确保你在终端或 Colab 已经跑过 wandb.login())
    wandb_run = _get_wandb()
    wandb_run.init(project='faimdl-lab3')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # 使用封装好的 DataLoader
    train_loader, _ = get_dataloaders(batch_size=64)

    model = CustomCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}\n-------------------------------')
        train_loop(train_loader, model, loss_fn, optimizer, device, wandb_run)
        
        # 每一个 Epoch 结束后保存一次 Checkpoint
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        save_checkpoint(checkpoint_state, filename=f'customcnn_epoch_{epoch+1}.pth')

    # 保存最终模型
    save_checkpoint({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filename='customcnn_final.pth')
    
    wandb_run.finish()

if __name__ == '__main__':
    main()
