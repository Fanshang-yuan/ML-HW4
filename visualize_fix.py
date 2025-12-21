import torch
import os
from model import UNet
from utils import save_images
from config import Config


args = Config()
CHECKPOINT_PATH = os.path.join("models", args.run_name, "last.pth") 
DEVICE = "cuda"

def adaptive_sampling():
    print(f"正在加载模型进行自适应采样: {CHECKPOINT_PATH}")
    device = torch.device(DEVICE)
    model = UNet(time_emb_dim=256).to(device)
    
    # 加载权重
    ckpt = torch.load(CHECKPOINT_PATH)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    
    # 采样参数 
    n = 32
    img_size = 32
    noise_steps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    
    beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    
    x = torch.randn((n, 3, img_size, img_size)).to(device)
    
    print("开始采样 (Steps: 1000)...")
    with torch.no_grad():
        for i in reversed(range(1, noise_steps)):
            t = (torch.ones(n) * i).long().to(device)
            predicted_noise = model(x, t)
            
            alpha_t = alpha[t][:, None, None, None]
            alpha_hat_t = alpha_hat[t][:, None, None, None]
            beta_t = beta[t][:, None, None, None]
            
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            # DDPM 采样公式
            x = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / (torch.sqrt(1 - alpha_hat_t))) * predicted_noise) + torch.sqrt(beta_t) * noise

    print(f"采样结束。原始数值范围: Min={x.min():.2f}, Max={x.max():.2f}")
    # 根据实际范围拉伸到 0-1
    # 还原清晰的图像
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-5)
    
    save_path = "adaptive_result4.jpg"
    from torchvision.utils import save_image
    save_image(x_norm, save_path)
    
    print(f"图片已保存至: {save_path}")
    print("if:fig清晰->模型没问题->数值范围偏移")

if __name__ == '__main__':
    adaptive_sampling()