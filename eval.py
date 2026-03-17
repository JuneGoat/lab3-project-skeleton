import torch
from torch import nn

# 从我们拆分的模块导入
from models.customnet import CustomCNN
from dataset.dataloader import get_dataloaders
from utils.checkpoint import load_checkpoint

def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    model.eval() # 设置为评估模式
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # 获取评估集的 DataLoader
    _, eval_loader = get_dataloaders(batch_size=64)

    # 实例化空模型
    model = CustomCNN().to(device)
    
    # 使用分离的工具函数加载 Checkpoint
    checkpoint_path = 'checkpoints/customcnn_final.pth'
    load_checkpoint(checkpoint_path, model, device=device)

    loss_fn = nn.CrossEntropyLoss()
    
    print("开始执行评估...")
    test_loop(eval_loader, model, loss_fn, device)

if __name__ == '__main__':
    main()