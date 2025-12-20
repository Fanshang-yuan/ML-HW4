import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm 
import logging
import os
from utils import *
from model import UNet 
from diffusion import Diffusion
from config import Config

# 设置日志格式
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    
    # 1. 初始化主模型 (训练用)
    model = UNet(time_emb_dim=256).to(device)
    
    # 2. 初始化 EMA 模型 (生成用，不参与梯度更新)
    ema_model = UNet(time_emb_dim=256).to(device)
    ema_model.load_state_dict(model.state_dict()) # 同步初始权重
    for param in ema_model.parameters():
        param.requires_grad = False # 冻结 EMA 参数

    # 3. 初始化优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.8)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    
    # 初始化 EMA 工具类 
    ema = EMA(beta=0.995) 
    
    # 初始化混合精度训练 (提速)
    scaler = torch.amp.GradScaler('cuda')

    # 记录 Loss
    epoch_losses = []
    best_loss = float('inf')
    
    logging.info(f"Starting training on {device} with EMA & AMP enabled.")

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, ncols=80, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        batch_losses = [] 
        
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0])
            
            optimizer.zero_grad()
            
            # --- 混合精度上下文 ---
            with torch.amp.autocast('cuda'):
                x_t, noise = diffusion.noise_images(images, t)
                predicted_noise = model(x_t, t)
                loss = mse(noise, predicted_noise)
            
            # --- 反向传播 (使用 Scaler) ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # --- 更新 EMA 模型 ---
            ema.step_ema(ema_model, model)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            batch_losses.append(loss.item())
        
        # --- Epoch 后处理 ---
        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        
        # 1. 更新学习率
        scheduler.step()
        
        # 2. 绘制曲线
        plot_loss_curve(epoch_losses, args.run_name)
        
        # 3. 保存最新权重
        save_checkpoint(ema_model, optimizer, epoch, avg_loss, args.run_name, filename="last.pth")
        
        # 4. 保存最佳权重 (Best Model)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(ema_model, optimizer, epoch, avg_loss, args.run_name, filename="best.pth")
            tqdm.write(f"New best model found at epoch {epoch+1} with loss {avg_loss:.5f}")

        # 5. 定期存档 (每20轮)
        if epoch > 0 and epoch % 20 == 0:
            save_filename = f"epoch_{epoch}.pth"
            save_checkpoint(ema_model, optimizer, epoch, avg_loss, args.run_name, filename=save_filename)
        
        # 6. 生成预览图 (每5轮，生成32张)
        if epoch % 10 == 0:
            sampled_images = diffusion.sample(ema_model, n=32)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))

if __name__ == '__main__':
    cfg = Config()
    train(cfg)




"""
# 测试数据加载和预处理是否正确
from config import Config
from utils import get_data, save_images, setup_logging
import torch

def test_data_loading():
    # 1. 初始化配置
    cfg = Config()
    setup_logging(cfg.run_name)
    
    # 2. 加载数据
    print("正在加载数据...")
    dataloader = get_data(cfg)
    
    # 3. 获取一个 Batch
    images, labels = next(iter(dataloader))
    print(f"数据加载成功！")
    print(f"图片形状: {images.shape}") # 应该是 [64, 3, 32, 32]
    print(f"数值范围: Min={images.min():.2f}, Max={images.max():.2f}") # 应该接近 -1 和 1
    
    # 4. 尝试保存图片（验证反归一化是否正确）
    save_path = f"results/{cfg.run_name}/test_data.jpg"
    save_images(images, save_path)
    print(f"测试图片已保存至: {save_path}")

if __name__ == '__main__':
    test_data_loading()
"""


"""
# 扩散效果测试
from config import Config
from utils import get_data, save_images, setup_logging
from diffusion import Diffusion # 导入刚才写的类
import torch

def test_diffusion_process():
    # 1. 初始化
    cfg = Config()
    setup_logging(cfg.run_name)
    diffusion = Diffusion(img_size=cfg.image_size, device=cfg.device)
    
    # 2. 拿一个 batch 的数据
    dataloader = get_data(cfg)
    images, _ = next(iter(dataloader))
    images = images.to(cfg.device)
    
    # 3. 设定我们要观察的时间点：原图 -> 加了一点噪 -> 加了很多噪 -> 纯噪
    # 比如观察第 0步, 100步, 500步, 999步
    t_steps = [0, 100, 500, 999] 
    
    noisy_images_list = []
    
    print("开始生成噪声演示图...")
    for t_val in t_steps:
        # 创建一个全是 t_val 的时间步张量
        t = torch.full((images.shape[0],), t_val, device=cfg.device, dtype=torch.long)
        
        # 调用 noise_images 加噪
        # 注意：如果是第0步，我们直接用原图
        if t_val == 0:
            x_t = images
        else:
            x_t, _ = diffusion.noise_images(images, t)
            
        # 取第一张图放进列表里展示
        noisy_images_list.append(x_t[0]) 

    # 4. 拼成一张大图保存 (4张图排成一排)
    result = torch.stack(noisy_images_list, dim=0).unsqueeze(0) # 增加 batch 维度以适配 grid
    # 此时 result shape 是 [1, 4, 3, 32, 32]，我们需要把它展平成 [4, 3, 32, 32]
    result = result.view(-1, 3, 32, 32)
    
    save_path = f"results/{cfg.run_name}/diffusion_process.jpg"
    save_images(result, save_path, nrow=4) # 一行4张
    print(f"扩散过程演示图已保存至: {save_path}")

if __name__ == '__main__':
    test_diffusion_process()
"""

