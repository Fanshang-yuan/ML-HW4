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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=6e-5)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)

    # 从断点恢复训练 
    start_epoch = 0
    resume_path = os.path.join("models", args.run_name, "last.pth") 
    if os.path.exists(resume_path):
        logging.info(f"Resuming training from {resume_path}")
        checkpoint = torch.load(resume_path)
        # 处理可能存在的 module 前缀
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        ema_model.load_state_dict(new_state_dict) # EMA 同步加载
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            
        logging.info(f"Loaded successfully! Starting from Epoch {start_epoch}")
    else:
        logging.info("No checkpoint found. Starting from scratch.")
    
    # 初始化 EMA 工具类 
    ema = EMA(beta=0.995) 
    
    # 初始化混合精度训练 (提速)
    scaler = torch.amp.GradScaler('cuda')

    # 记录 Loss
    epoch_losses = []
    best_loss = float('inf')
    
    logging.info(f"Starting training on {device} with EMA & AMP enabled.")

    for epoch in range(start_epoch, args.epochs):
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
        
        # 4. 保存最佳权重
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(ema_model, optimizer, epoch, avg_loss, args.run_name, filename="best.pth")
            tqdm.write(f"New best model found at epoch {epoch+1} with loss {avg_loss:.5f}")

        # 5. 定期存档 
        if epoch > 0 and epoch % 200 == 0:
            save_filename = f"epoch_{epoch}.pth"
            save_checkpoint(ema_model, optimizer, epoch, avg_loss, args.run_name, filename=save_filename)
        
        # 6. 生成预览图 
        if epoch % 200 == 0:
            sampled_images = diffusion.sample(ema_model, n=32)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))

if __name__ == '__main__':
    cfg = Config()
    train(cfg)
