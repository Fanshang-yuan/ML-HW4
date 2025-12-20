import torch
import os
import shutil
from model import UNet
from diffusion import Diffusion
from config import Config
from utils import save_images
from tqdm import tqdm
from torch_fidelity import calculate_metrics

def generate_samples_for_eval(cfg, n_samples=2000, batch_size=50):
    """
    生成用于评测的图片
    """
    device = cfg.device
    model = UNet(time_emb_dim=256).to(device)
    
    # 1. 加载最佳权重
    ckpt_path = os.path.join("models", cfg.run_name, "best.pth")
    if not os.path.exists(ckpt_path):
        print(f"⚠️ 警告: 未找到 {ckpt_path}，将尝试加载 last.pth")
        ckpt_path = os.path.join("models", cfg.run_name, "last.pth")
    
    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    diffusion = Diffusion(img_size=cfg.image_size, device=device)
    
    # 2. 准备保存路径
    # 也就是把生成的几千张图存在这里
    eval_folder = os.path.join("results", cfg.run_name, "eval_images")
    if os.path.exists(eval_folder):
        shutil.rmtree(eval_folder) # 清空旧数据
    os.makedirs(eval_folder)

    print(f"开始生成 {n_samples} 张图片用于评测...")
    
    # 3. 批量生成
    n_generated = 0
    # 为了加快速度，我们分批生成
    num_batches = int(n_samples / batch_size)
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches)):
            # 这里的 sample 需要返回原始张量 [-1, 1] 或者 [0, 1] 都可以
            # 我们直接用 diffusion.sample 得到最终结果
            sampled_images = diffusion.sample(model, n=batch_size)
            
            # 一张张保存
            for idx, img in enumerate(sampled_images):
                # 构造唯一文件名
                file_name = f"{n_generated + idx}.png"
                save_path = os.path.join(eval_folder, file_name)
                
                # 这里我们需要手动处理单张图片的保存，不能直接用 utils.save_images
                # 因为 utils.save_images 是拼图，我们需要单张图
                from torchvision.utils import save_image
                # 反归一化已经在 diffusion.sample 里做过了吗？
                # 检查你的 diffusion.sample 返回的是什么。
                # 如果是上一轮修改过的版本，它返回的是 [0, 255] 的 uint8
                # 为了 save_image 方便，我们最好让它转回 [0, 1] float
                img_float = img.float() / 255.0 
                save_image(img_float, save_path)
            
            n_generated += batch_size

    print(f"生成完毕！图片保存在: {eval_folder}")
    return eval_folder

def calc_metrics(eval_folder, device):
    print("开始计算 FID 和 IS 指标...")
    print("注意：这可能需要下载 CIFAR-10 统计数据，请保持网络连接。")
    
    # input1: 'cifar10-train' 是库内置的关键字，自动对比 CIFAR10 训练集
    # input2: 我们刚才生成的文件夹
    metrics_dict = calculate_metrics(
        input1='cifar10-train', 
        input2=eval_folder, 
        cuda=True if device == 'cuda' else False,
        isc=True, # 计算 Inception Score
        fid=True, # 计算 FID
        verbose=False,
    )
    
    print("\n" + "="*40)
    print(f" >> Inception Score (IS): {metrics_dict['inception_score_mean']:.4f} ± {metrics_dict['inception_score_std']:.4f}")
    print(f" >> FID Score:           {metrics_dict['frechet_inception_distance']:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    cfg = Config()
    
    # 1. 生成图片 
    img_folder = generate_samples_for_eval(cfg, n_samples=2000, batch_size=64)
    
    # 2. 计算指标
    calc_metrics(img_folder, cfg.device)