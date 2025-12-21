import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm 
import logging
import os
import numpy as np
from utils import * 
from model import UNet 
from diffusion import Diffusion
from config import Config  

# 设置设备可见性
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(args):
    # 1. 初始化日志 
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    
    logging.info(f"显卡模式: {torch.cuda.get_device_name(0)}")
    
    # 2. 初始化主模型
    model = UNet(time_emb_dim=256, num_classes=args.num_classes, device=device).to(device)
    
    # 3. 初始化 EMA 模型 (用于推理)
    ema_model = UNet(time_emb_dim=256, num_classes=args.num_classes, device=device).to(device)
    ema_model.load_state_dict(model.state_dict())
    for param in ema_model.parameters():
        param.requires_grad = False 

    # 优化器与损失函数
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-5)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)

    # === 4. 权重加载与断点恢复逻辑 ===
    resume_path = os.path.join("models", args.run_name, "last.pth")
    start_epoch = 0
    loss_history = [] # 

    if os.path.exists(resume_path):
        logging.info(f"发现存档，尝试加载: {resume_path}")
        checkpoint = torch.load(resume_path)
        
        # 兼容性处理：去除 'module.' 前缀
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        try:
            # 加载主模型
            model.load_state_dict(new_state_dict, strict=True)
            
            # 恢复优化器
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载 EMA 模型
            if 'ema_model_state_dict' in checkpoint:
                ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
                logging.info("EMA 权重加载成功")
            else:
                logging.warning("未发现 EMA 权重，使用主模型权重初始化")
                ema_model.load_state_dict(new_state_dict)

            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                
            logging.info(f"成功恢复至 Epoch {start_epoch}")
            
        except Exception as e:
            logging.error(f"权重加载失败: {e}")
            logging.error("建议重新开始")
            return 
    else:
        logging.info("无存档，从头开始训练。")

    ema = EMA(beta=0.995) 
    scaler = torch.amp.GradScaler('cuda')

    logging.info("开始训练...")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(dataloader, ncols=100, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0 # 记录当前轮次的总 loss
        
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            t = diffusion.sample_timesteps(images.shape[0])
            
            # Label Dropping (CFG): 10% 概率无条件训练
            if np.random.random() < 0.1: 
                labels = None 
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                x_t, noise = diffusion.noise_images(images, t)
                predicted_noise = model(x_t, t, labels)
                loss = mse(noise, predicted_noise)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            ema.step_ema(ema_model, model)
            
            loss_val = loss.item()
            epoch_loss += loss_val
            pbar.set_postfix(loss=f"{loss_val:.4f}")
        
        # === Epoch 结束处理 ===
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        scheduler.step()
        
        # 1. 绘制并保存 Loss 曲线
        plot_loss_curve(loss_history, args.run_name)
        
        # 2. 保存权重
        # 保存 last.pth
        save_checkpoint(model, optimizer, epoch, avg_loss, args.run_name, "last.pth", ema_model=ema_model)
        
        # 备份
        save_interval = getattr(args, 'save_interval', 200)
        if (epoch + 1) % save_interval == 0:
            filename = f"ckpt_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch, avg_loss, args.run_name, filename, ema_model=ema_model)
            logging.info(f"权重已备份: {filename}")
        
        # 3. 采样测试
        sample_interval = getattr(args, 'sample_interval', 5)
        if (epoch + 1) % sample_interval == 0:
            # 生成 0-9 类，每类生成 3 张 
            n_per_class = 4
            labels = torch.arange(args.num_classes).repeat_interleave(n_per_class).to(device)
            n_samples = args.num_classes * n_per_class
            
            # sample 返回 [-1, 1] 的 Tensor，由 utils.save_images 处理归一化
            sampled_images = diffusion.sample(ema_model, n=n_samples, labels=labels, cfg_scale=3.0)
            
            save_path = os.path.join("results", args.run_name, f"{epoch+1}.jpg")
            save_images(sampled_images, save_path, nrow=n_per_class)

if __name__ == '__main__':
    cfg = Config()
    train(cfg)