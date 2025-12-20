import torch

# 配置，包含训练和模型的超参数
class Config:
    def __init__(self):

        # 训练参数
        self.run_name = "DDPM_Uncond_CIFAR10"
        self.epochs = 2000
        self.batch_size = 512
        self.image_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = 3e-4
        
        # Diffusion specific
        self.T = 1000  # 总时间步数
        self.beta_start = 1e-4
        self.beta_end = 0.02


