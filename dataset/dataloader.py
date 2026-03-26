import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(data_root='./data/tiny-imagenet-200', batch_size=256): # 增大 batch_size
    """
    构建并返回训练集和评估集的 DataLoader
    """
    transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_root = os.environ.get('DATA_ROOT', data_root)
    candidate_roots = [data_root]
    if data_root == './data/tiny-imagenet-200':
        candidate_roots.append('./tiny-imagenet-200')

    # 针对 Lab 演示，由于 TinyImageNet val 目录结构特殊，此处暂时复用 train 作为 eval 示例。
    # 实际项目中，你应该指向真实的验证集路径 (如 'val') 并在必要时自定义 Dataset。
    train_dir = None
    eval_dir = None
    for root in candidate_roots:
        candidate_train_dir = os.path.join(root, 'train')
        if os.path.exists(candidate_train_dir):
            train_dir = candidate_train_dir
            eval_dir = os.path.join(root, 'train')
            break

    if train_dir is None:
        if os.environ.get('USE_FAKE_DATA') in {'1', 'true', 'True', 'yes', 'YES'}:
            train_dataset = datasets.FakeData(
                size=512,
                image_size=(3, 64, 64),
                num_classes=200,
                transform=eval_transform,
            )
            eval_dataset = datasets.FakeData(
                size=128,
                image_size=(3, 64, 64),
                num_classes=200,
                transform=transform,
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
            return train_loader, eval_loader

        expected_train_dir = os.path.join(data_root, 'train')
        raise FileNotFoundError(
            f"找不到数据目录: {expected_train_dir}。请先下载并解压数据集，或设置环境变量 DATA_ROOT。"
        )

    # 构建 Dataset
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    eval_dataset = datasets.ImageFolder(root=eval_dir, transform=eval_transform)

    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader
