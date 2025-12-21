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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def train(args):
    # 1. 初始化日志
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    
    # 打印显卡信息，确认双卡识别
    gpu_count = torch.cuda.device_count()
    logging.info(f"检测到 {gpu_count} 张显卡: {[torch.cuda.get_device_name(i) for i in range(gpu_count)]}")
    if gpu_count > 1:
        logging.info("多卡 DataParallel 训练模式")
    
    # 2. 初始化主模型
    model = UNet(time_emb_dim=256, num_classes=args.num_classes, device=device).to(device)
    
    # 3. 如果有多张卡，使用 DataParallel 包裹 model
   
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 3. 初始化 EMA 模型
    ema_model = UNet(time_emb_dim=256, num_classes=args.num_classes, device=device).to(device)
    # 4. 初始化 EMA 权重时，要从 model.module (如果是多卡) 或 model (单卡) 复制
    source_model = model.module if isinstance(model, nn.DataParallel) else model
    ema_model.load_state_dict(source_model.state_dict())
    
    for param in ema_model.parameters():
        param.requires_grad = False 

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)

    # === 4. 权重加载 ===
    resume_path = os.path.join("models", args.run_name, "last.pth")
    start_epoch = 0
    loss_history = [] 

    if os.path.exists(resume_path):
        logging.info(f"发现存档，尝试加载: {resume_path}")
        checkpoint = torch.load(resume_path)
        
        # 兼容性处理
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        try:
            # 5. 加载权重时的策略
            # 如果当前 model 是 DataParallel，它期望 key 有 'module.' 前缀
            # 但为了通用性，我们建议总是加载到 '底层模型' (model.module)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(new_state_dict, strict=True)
            else:
                model.load_state_dict(new_state_dict, strict=True)
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
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
            return 
    else:
        logging.info("⚠️ 无存档，从头开始训练。")

    ema = EMA(beta=0.995) 
    scaler = torch.amp.GradScaler('cuda')

    logging.info("开始训练...")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(dataloader, ncols=100, desc=f"Ep {epoch+1}/{args.epochs}")
        epoch_loss = 0 
        
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            t = diffusion.sample_timesteps(images.shape[0])
            
            if np.random.random() < 0.1: 
                labels = None 
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                x_t, noise = diffusion.noise_images(images, t)
                # DataParallel 会自动切分 batch 分配到两张卡上
                predicted_noise = model(x_t, t, labels)
                loss = mse(noise, predicted_noise)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 6. EMA 更新
            # 必须传入底层的 model (即去掉 DataParallel 壳子的)，否则参数名对不上
            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            ema.step_ema(ema_model, actual_model)
            
            loss_val = loss.item()
            epoch_loss += loss_val
            pbar.set_postfix(loss=f"{loss_val:.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        scheduler.step()
        
        plot_loss_curve(loss_history, args.run_name)
        
        # 7. 保存权重
        # 始终保存 model.module (纯净版权重)
        save_model_ref = model.module if isinstance(model, nn.DataParallel) else model
        
        save_checkpoint(save_model_ref, optimizer, epoch, avg_loss, args.run_name, "last.pth", ema_model=ema_model)
        
        save_interval = getattr(args, 'save_interval', 200)
        if (epoch + 1) % save_interval == 0:
            filename = f"ckpt_epoch_{epoch+1}.pth"
            save_checkpoint(save_model_ref, optimizer, epoch, avg_loss, args.run_name, filename, ema_model=ema_model)
            logging.info(f"权重已备份: {filename}")
        
        sample_interval = getattr(args, 'sample_interval', 5)
        if (epoch + 1) % sample_interval == 0:
            n_per_class = 4
            labels = torch.arange(args.num_classes).repeat_interleave(n_per_class).to(device)
            n_samples = args.num_classes * n_per_class
        
            sampled_images = diffusion.sample(ema_model, n=n_samples, labels=labels, cfg_scale=3.0)
            
            save_path = os.path.join("results", args.run_name, f"{epoch+1}.jpg")
            save_images(sampled_images, save_path, nrow=n_per_class)

if __name__ == '__main__':
    cfg = Config()
    
    if hasattr(cfg, 'batch_size'):
        cfg.batch_size = 256
    
    train(cfg)