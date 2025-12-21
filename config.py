import torch

# 配置，包含训练和模型的超参数
class Config:
    def __init__(self):

        # # 训练参数 DDPM_Uncond_CIFAR10
        # self.run_name = "DDPM_Uncond_CIFAR10"
        # self.epochs = 20000
        # self.batch_size = 512
        # self.image_size = 32
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.lr = 3e-4
        
        # # Diffusion specific
        # self.T = 1000  # 总时间步数
        # self.beta_start = 1e-4
        # self.beta_end = 0.02

        # 训练参数 DDPM_Cond_CIFAR10
        self.run_name = "DDPM_Cond_CIFAR10" # 
        self.epochs = 20000 # 
        self.batch_size = 128 # 
        self.image_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = 3e-4

        self.T = 1000  # 总时间步数
        self.beta_start = 1e-4
        self.beta_end = 0.02
        
        self.num_classes = 10      # 0-9 是真实类别
        self.cfg_dropout = 0.1     # 10% 的概率丢弃标签

        self.save_interval = 200  # 每多少个 epoch 保存一次备份
        self.sample_interval = 50  # 每多少个 epoch 采样一次图片


