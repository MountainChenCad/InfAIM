# 文件: main_finetune_video.py (版本 11 - 支持阈值分析)
# 核心改动:
# 1. (FEATURE) 更新了评估结果的打印逻辑，以处理 `evaluate` 函数返回的性能列表。
# 2. (FEATURE) 现在会打印一个清晰的表格，显示在不同置信度阈值下的召回率、
#    虚警率和稳定性，便于进行详细的性能分析。

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
import random


# ... (collate_fn and get_args_parser remain unchanged) ...
def collate_fn_train(batch):
    clips, targets_list_of_lists = zip(*batch);
    clips_batch = torch.stack(clips, 0);
    collated_targets = {}
    if len(targets_list_of_lists) > 0 and len(targets_list_of_lists[0]) > 0:
        keys = targets_list_of_lists[0][0].keys()
        for key in keys:
            tensors_for_key = []
            for clip_targets in targets_list_of_lists:
                clip_tensor_for_key = torch.stack([frame[key] for frame in clip_targets], dim=0);
                tensors_for_key.append(clip_tensor_for_key)
            collated_targets[key] = torch.stack(tensors_for_key, dim=0)
    return clips_batch, collated_targets


def collate_fn_val(batch): return batch[0]


def get_args_parser():
    parser = argparse.ArgumentParser('InfMAE Video Detection Fine-tuning', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int);
    parser.add_argument('--epochs', default=40, type=int)
    # ========================= 核心修改点 =========================
    # 1. 进一步降低学习率
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate (default: 1e-6 for stable full fine-tuning)')
    # 2. 添加梯度裁剪参数
    parser.add_argument('--clip_grad', type=float, default=0.1, help='gradient clipping max norm')
    # ==========================================================
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--pretrained_weights', default='./checkpoints/infmae_pretrained_weights.pth', type=str)
    parser.add_argument('--num_classes', default=6, type=int);
    parser.add_argument('--clip_length', default=8, type=int)
    parser.add_argument('--data_path', default='./dataset', type=str,
                        help='Path to the new dataset root (containing scene_X folders)')
    parser.add_argument('--output_dir', default='./output_finetune', help='path where to save')
    parser.add_argument('--device', default='cuda', help='device to use for training');
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint');  # 清空resume
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=0, type=int);
    parser.add_argument('--train_data_fraction', type=float, default=1,
                        help='Fraction of training data to use (e.g. 0.2 for 20%)')
    parser.add_argument('--world_size', default=1, type=int);
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true');
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    misc.init_distributed_mode(args);
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))));
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device);
    cudnn.benchmark = True;
    random.seed(args.seed)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.425, 0.425, 0.425], std=[0.200, 0.200, 0.200])])
    base_path = args.data_path
    train_scene_paths = [os.path.join(base_path, d) for d in ['scene_1']];

    val_scene_paths = [os.path.join(base_path, 'contest_1_scene_1')]
    # val_scene_paths = [os.path.join(base_path, 'contest_1')]
    dataset_train = VideoDetectionDataset(scene_paths=train_scene_paths, clip_length=args.clip_length,
                                          transform=transform, return_clips=True, num_classes=args.num_classes)
    if args.train_data_fraction < 1.0:
        num_train_samples = math.ceil(len(dataset_train) * args.train_data_fraction)
        dataset_train = Subset(dataset_train, list(range(num_train_samples)))
        print(f"Using {args.train_data_fraction * 100:.1f}% of the training data: {len(dataset_train)} samples.")
    sampler_train = torch.utils.data.RandomSampler(
        dataset_train) if not args.distributed else torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
    data_loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size,
                                   num_workers=args.num_workers, collate_fn=collate_fn_train, drop_last=True)
    dataset_val = VideoDetectionDataset(scene_paths=val_scene_paths, transform=transform, return_clips=False,
                                        num_classes=args.num_classes)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=1, num_workers=args.num_workers,
                                 collate_fn=collate_fn_val)
    data_loader_val_vis = data_loader_val
    full_dataset_train_vis = VideoDetectionDataset(scene_paths=train_scene_paths, transform=transform,
                                                   return_clips=False, num_classes=args.num_classes)
    all_vis_indices = list(range(len(full_dataset_train_vis)));
    random.shuffle(all_vis_indices)
    vis_indices = all_vis_indices[:min(10, len(full_dataset_train_vis))];
    print(f"Randomly selected {len(vis_indices)} training sequences for visualization.")
    subset_train_vis = Subset(full_dataset_train_vis, vis_indices)
    sampler_train_vis = torch.utils.data.SequentialSampler(subset_train_vis)
    data_loader_train_vis = DataLoader(subset_train_vis, sampler=sampler_train_vis, batch_size=1,
                                       num_workers=args.num_workers, collate_fn=collate_fn_val)
    model = AIM_InfMAE_Detector(pretrained_infmae_path=args.pretrained_weights, num_classes=args.num_classes,
                                clip_length=args.clip_length).to(device)
    model_without_ddp = model
    if args.distributed: model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[
        args.gpu]); model_without_ddp = model.module
    params_to_update = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_update, lr=args.lr, weight_decay=args.weight_decay)
    # Use a dummy loss scaler if not using AMP
    loss_scaler = misc.NativeScalerWithGradNormCount() if 'amp' in args and args.amp else None
    load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed: data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, log_writer=None, args=args)

        # NEW: The evaluate function now returns a list of stats for each threshold
        eval_results_list = evaluate(model_without_ddp, data_loader_val, device, args.output_dir, epoch)

        # NEW: Print a formatted table with the results
        print(f"--- Epoch {epoch} Evaluation Analysis ---")
        print("=" * 80)
        print(
            f"{'Threshold':>10s} | {'Recall':>10s} | {'False Alarm':>12s} | {'Stability':>10s} | {'TP':>8s} | {'FP':>8s} | {'FN':>8s}")
        print("-" * 80)
        for stats in eval_results_list:
            print(
                f"{stats['threshold']:>10.2f} | {stats['avg_recall']:>10.4f} | {stats['avg_false_alarm']:>12.4f} | {stats['spatial_stability']:>10.4f} | {stats['tp']:>8d} | {stats['fp']:>8d} | {stats['fn']:>8d}")
        print("=" * 80)

        if args.output_dir and misc.is_main_process():
            # Saving logic, visualization etc.
            print("--- Starting Visualization ---")
            visualize_epoch(model_without_ddp, data_loader_val_vis, device, args.output_dir, epoch, 'val_visuals')
            visualize_epoch(model_without_ddp, data_loader_train_vis, device, args.output_dir, epoch, 'train_visuals')
            print("--- Finished Visualization ---")

            save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            to_save = {
                'epoch': epoch,
                'model_state_dict': model_without_ddp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if loss_scaler is not None:
                to_save['scaler_state_dict'] = loss_scaler.state_dict()
            torch.save(to_save, save_path)


if __name__ == '__main__':
    args = get_args_parser();
    args = args.parse_args()
    if args.output_dir: os.makedirs(args.output_dir, exist_ok=True)
    main(args)