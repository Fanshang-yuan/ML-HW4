import torch
import logging

class Diffusion:

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # 1. 准备噪声调度表 beta 
        self.beta = self.prepare_noise_schedule().to(device)
        
        # 2. 计算 alpha 及其相关参数 
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) # alpha_hat 是 alpha 的累乘

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        前向加噪过程 q(x_t | x_0)
        公式: x_t = sqrt(alpha_hat) * x_0 + sqrt(1 - alpha_hat) * epsilon
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        
        epsilon = torch.randn_like(x) # 随机噪声
        
        # 返回: 加噪后的图 x_t, 以及刚才加进去的噪声 epsilon
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        """
        随机采样时间步 t，用于训练
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)

    # --- 反向生成 ---
    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # 1. 从纯噪声开始
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            
            # 2. 从 T 走到 1
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                
                # 模型预测噪声
                predicted_noise = model(x, t)
                
                # 根据公式减去预测的噪声（DDPM采样公式）
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        model.train()
        # 把值限制回 [-1, 1] 之间
        x = (x.clamp(-1, 1) + 1) / 2 
        x = (x * 255).type(torch.uint8)
        return x