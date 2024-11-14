import os
import csv
from tqdm import tqdm
import torch
import argparse
# from tools.dataset import Dataset_train, Dataset_val
from torch.utils.data import Dataset, DataLoader
import cv2
from tools.utils import valid,train
from tools.patch_generate import divide
import matplotlib.pyplot as plt
from model.auto_vit import VAE
import imutils
import pandas as pd 
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

min_loss = 1000.0

# img_path = r".\image\splicing-01.png"
# noise_path = r".\noise\splicing-01.mat"
# save_path = r".\results\splicing-01.png"
# mask_path = r""


class DeepfakeDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 이미지 경로 가져오기
        img_path = self.df.loc[idx, 'path']  # loc 사용
        noise_path = img_path.replace("DFWild-Cup/", "DFWild-Cup/noiseprint/")
        
        # data_idx 처리 (확장자 제거)
        valid_extensions = ['.png', '.jpg', '.jpeg']  # 유효한 확장자 리스트
        for ext in valid_extensions:
            if img_path.endswith(ext):
                data_idx = img_path[img_path.find("DFWild-Cup/") + len("DFWild-Cup/"):].replace(ext, '')
                break
        else:
            raise ValueError(f"Unsupported file extension in path: {img_path}")

        # noise_path 확장자 변경
        noise_path = os.path.splitext(noise_path)[0] + '.mat'

        # 레이블 가져오기
        label = self.df.loc[idx, 'label']  # loc 사용

        # 이미지 읽기 및 크기 확인
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        [m, n, c] = img_cv.shape

        # 패치 크기 및 스트라이드 설정
        patch_size = 64
        if m * n > 1000 * 1000:
            patch_stride = 16
            # print(f"Input image size [{m} x {n}] large")
        else:
            patch_stride = 8
            # print(f"Input image size [{m} x {n}] small")
        
        # 데이터 분할
        data_all, qf = divide(img_path, noise_path, patch_size, patch_stride)
        noise_list = data_all[0]
        data = noise_list[0]

        # 텐서 변환
        noise = torch.Tensor(data)
        return data_idx, noise, label

def main():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--loss_interval', type=int, default=2)
    parser.add_argument('--stop_loss', type=float, default=0.0001, help='Stop training if loss is below this threshold.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for anomaly detection.')
    parser.add_argument('--coef', type=float, nargs='+', default=[0.05, 0.05, 0.9], help='Coefficients for loss terms. alpha, beta, gamma')
    args = parser.parse_args()

    # Get local rank from environment variable
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Initialize distributed process group
    dist.init_process_group(backend="nccl")

    # Set the device for this rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Prepare datasets and samplers
    train_df = pd.read_csv('./data/DFWild-Cup_train.csv')
    valid_df = pd.read_csv('./data/valid.csv')

    train_dataset = DeepfakeDataset(train_df)  # Replace with your dataset class
    valid_dataset = DeepfakeDataset(valid_df)  # Replace with your dataset class

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler)

    # Initialize the model and optimizer
    model = VAE().to(device)  # Replace with your model class
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3, T_mult=2, eta_min=1e-4, last_epoch=-1
    )

    exp_name = "DFWild-Cup_ViTVAE"

    # Initialize WandB on rank 0 only
    if rank == 0:
        wandb.init(project=exp_name, name="auto_vit")
        wandb.config.update(args)
        wandb.watch(model)

    os.makedirs("results", exist_ok=True)
    os.makedirs(f"results/{exp_name}", exist_ok=True)

    # Training and validation loop
    for epoch in tqdm(range(args.epoch), desc="Training Epochs"):
        train_sampler.set_epoch(epoch)  # Ensures proper shuffling for each epoch
        model.train()
        train_loss = train(train_dataloader, optimizer, model, device, epoch, rank, coef=args.coef)

        model.eval()
        val_loss, val_metrics = valid(valid_dataloader, optimizer, model, device, exp_name, epoch, rank, coef=args.coef)

        # Log results only on rank 0
        if rank == 0:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_auc": val_metrics["auc"],
                "val_precision": val_metrics["precision"],
                "val_f1": val_metrics["f1"],
                "val_recall": val_metrics["recall"],
                "val_eer": val_metrics["eer"],
            })

    # Cleanup the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
