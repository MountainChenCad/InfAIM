# 文件: main_finetune_video.py (版本 8 - 修复可视化采样偏差)
# 核心改动:
# 1. (BUG FIX) 修复了可视化采样仅从第一个场景中选择的问题。
# 2. (FEATURE) 现在，代码会随机地从所有训练场景的全部序列中挑选10个进行可视化，
#    从而能够公正地反映模型在整个训练集上的表现。

import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import util.misc as misc
from util.misc import load_model
from dataset_video import VideoDetectionDataset
from models_aim import AIM_InfMAE_Detector
from engine_finetune import train_one_epoch, evaluate, visualize_epoch
import math
import random  # <--- 核心修改点 1


def collate_fn_train(batch):
    clips, targets = zip(*batch)
    return torch.stack(clips, 0), list(targets)


def collate_fn_val(batch):
    return batch[0]


def get_args_parser():
    parser = argparse.ArgumentParser('InfMAE Video Detection Fine-tuning', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--pretrained_weights', default='./checkpoints/infmae_pretrained_weights.pth', type=str)
    parser.add_argument('--num_classes', default=6, type=int)
    parser.add_argument('--clip_length', default=8, type=int)
    parser.add_argument('--data_path', default='./dataset', type=str,
                        help='Path to the new dataset root (containing scene_X folders)')
    parser.add_argument('--output_dir', default='./output_finetune', help='path where to save')
    parser.add_argument('--device', default='cuda', help='device to use for training')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--train_data_fraction', type=float, default=0.1,
                        help='Fraction of training data to use (e.g. 0.2 for 20%)')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True
    # 设置随机种子以确保可复现的随机可视化采样
    random.seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.425, 0.425, 0.425], std=[0.200, 0.200, 0.200])
    ])

    base_path = args.data_path
    train_scene_paths = [os.path.join(base_path, d) for d in ['scene_1', 'scene_2', 'scene_3']]
    val_scene_paths = [os.path.join(base_path, 'contest_1')]

    dataset_train = VideoDetectionDataset(scene_paths=train_scene_paths, clip_length=args.clip_length,
                                          transform=transform, return_clips=True)

    if args.train_data_fraction < 1.0:
        num_train_samples = math.ceil(len(dataset_train) * args.train_data_fraction)
        train_indices = list(range(num_train_samples))
        dataset_train = Subset(dataset_train, train_indices)
        print(f"Using {args.train_data_fraction * 100:.1f}% of the training data: {len(dataset_train)} samples.")

    sampler_train = torch.utils.data.DistributedSampler(dataset_train,
                                                        shuffle=True) if args.distributed else torch.utils.data.RandomSampler(
        dataset_train)
    data_loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size,
                                   num_workers=args.num_workers, collate_fn=collate_fn_train, drop_last=True)

    dataset_val = VideoDetectionDataset(scene_paths=val_scene_paths, transform=transform, return_clips=False)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=1, num_workers=args.num_workers,
                                 collate_fn=collate_fn_val)

    data_loader_val_vis = data_loader_val

    full_dataset_train_vis = VideoDetectionDataset(scene_paths=train_scene_paths, transform=transform,
                                                   return_clips=False)

    # ========================= 核心修改点 2: 随机采样可视化序列 =========================
    all_vis_indices = list(range(len(full_dataset_train_vis)))
    random.shuffle(all_vis_indices)  # 打乱所有可用于可视化的训练序列的索引
    vis_indices = all_vis_indices[:min(10, len(full_dataset_train_vis))]  # 从打乱后的列表中取前10个
    print(f"Randomly selected {len(vis_indices)} training sequences for visualization.")
    # =================================================================================

    subset_train_vis = Subset(full_dataset_train_vis, vis_indices)
    sampler_train_vis = torch.utils.data.SequentialSampler(subset_train_vis)
    data_loader_train_vis = DataLoader(subset_train_vis, sampler=sampler_train_vis, batch_size=1,
                                       num_workers=args.num_workers, collate_fn=collate_fn_val)

    model = AIM_InfMAE_Detector(pretrained_infmae_path=args.pretrained_weights, num_classes=args.num_classes,
                                clip_length=args.clip_length).to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_update = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_update, lr=args.lr, weight_decay=args.weight_decay)

    load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=None)

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, args=args)

        eval_stats = evaluate(model_without_ddp, data_loader_val, device, args.output_dir, epoch)

        print(f"--- Epoch {epoch} Evaluation ---")
        print(f"  目标识别召回率:      {eval_stats['avg_recall']:.4f}")
        print(f"  目标识别虚警率:      {eval_stats['avg_false_alarm']:.4f}")
        print(f"  时空序列稳定性:      {eval_stats['spatial_stability']:.4f}")
        print("-----------------------------")

        if args.output_dir and misc.is_main_process():
            visualize_epoch(model_without_ddp, data_loader_train_vis, device, args.output_dir, epoch, 'train_visuals')
            visualize_epoch(model_without_ddp, data_loader_val_vis, device, args.output_dir, epoch, 'val_visuals')

            torch.save({
                'epoch': epoch,
                'model_state_dict': model_without_ddp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)