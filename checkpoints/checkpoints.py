import os
import torch

def save_checkpoint(state, checkpoint_dir='checkpoints', filename='checkpoint.pth'):
    """
    保存模型、优化器状态和当前 Epoch
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint 已保存至: {filepath}")

def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    加载模型和优化器状态
    """
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        print(f"成功加载 Checkpoint: {filepath}")
        return checkpoint.get('epoch', 0)
    else:
        print(f"警告: 找不到 Checkpoint 文件 {filepath}")
        return 0