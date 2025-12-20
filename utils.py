import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
# import logging


def setup_logging(run_name):
    """
    创建存放模型、日志和结果的文件夹
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def save_images(images, path, **kwargs):
    """
    保存图片
    Args:
        images: 形状为 (B, C, H, W) 的 Tensor，范围 [-1, 1]
        path: 保存路径
    """
    grid = torchvision.utils.make_grid(images, **kwargs)
    # 将 [-1, 1] 转换回 [0, 1] 
    ndarr = grid.permute(1, 2, 0).cpu().numpy()
    im = (ndarr + 1) / 2.0  # 反归一化
    im = (im * 255).clip(0, 255).astype('uint8')
    image = Image.fromarray(im)
    image.save(path)

def get_data(args):
    """
    加载 CIFAR-10 数据集
    Args:
        args: 配置对象 (包含 batch_size 等)
    """
    transforms_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(), # 数据增强
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 关键：归一化到 [-1, 1]
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', 
                                           train=True,
                                           download=True, 
                                           transform=transforms_train)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=12,
                                             pin_memory=True,
                                             persistent_workers=True)
    
    return dataloader


def save_checkpoint(model, optimizer, epoch, loss, run_name, filename):
    """
    保存模型权重
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    save_path = os.path.join("models", run_name, filename)
    torch.save(checkpoint, save_path)
    #logging.info(f"Saved checkpoint: {save_path}") 


def plot_loss_curve(losses, run_name):
    """
    绘制 Loss 曲线
    """
    plt.figure(figsize=(10, 6))
    
    # 设置字体为 Times New Roman 
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    
    plt.plot(losses, label='Training Loss', color='black', linewidth=1.5)
    
    plt.title('Training Loss Convergence', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MSE Loss', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存
    plt.savefig(os.path.join("results", run_name, "loss_curve.png"), dpi=600, bbox_inches='tight')
    plt.close()


# EMA 平滑类
class EMA:
    def __init__(self, beta=0.995):
        self.beta = beta
        self.step = 0

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        """
        ema_model: 平滑后的模型
        model: 正在训练的模型
        step_start_ema: 前多少步不进行平滑
        """
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        
        self.update_model_average(ema_model, model)
        self.step += 1

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old_weight, up_weight = ema_params.data, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

