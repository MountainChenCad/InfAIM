# 文件: main_finetune_video.py (诊断版本)
# 核心改动:
# 1. 将 epoch 和 output_dir 传递给 evaluate 函数，以便可视化函数可以创建带有纪元编号的文件。

import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
import util.misc as misc
from dataset_video import VideoDetectionDataset
from models_aim import AIM_InfMAE_Detector
from engine_finetune import train_one_epoch, evaluate


def collate_fn_train(batch):
    clips, targets = zip(*batch)
    clips_batch = torch.stack(clips, 0)
    return clips_batch, list(targets)


def collate_fn_val(batch):
    return batch[0]


def get_args_parser():
    parser = argparse.ArgumentParser('InfMAE Video Detection Fine-tuning', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for fine-tuning')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--pretrained_weights', default='./checkpoints/infmae_pretrained_weights.pth', type=str,
                        help='Path to the pretrained InfMAE weights.')
    parser.add_argument('--num_classes', default=6, type=int)
    parser.add_argument('--clip_length', default=8, type=int, help='Number of frames per video clip')
    parser.add_argument('--data_path', default='./dataset', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./output_finetune', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.425, 0.425, 0.425], std=[0.200, 0.200, 0.200])
    ])

    dataset_train = VideoDetectionDataset(data_root=args.data_path, clip_length=args.clip_length, transform=transform,
                                          mode='train')
    dataset_val = VideoDetectionDataset(data_root=args.data_path, transform=transform, mode='val')

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn_train,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=collate_fn_val,
    )

    model = AIM_InfMAE_Detector(
        pretrained_infmae_path=args.pretrained_weights,
        num_classes=args.num_classes,
        clip_length=args.clip_length
    )
    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_update = [p for p in model_without_ddp.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in params_to_update)}")
    optimizer = torch.optim.AdamW(params_to_update, lr=args.lr, weight_decay=args.weight_decay)

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, args=args
        )

        # ========================= 核心修改点 =========================
        # 将 epoch 和 output_dir 传递给 evaluate 函数
        eval_stats = evaluate(model_without_ddp, data_loader_val, device, args.output_dir, epoch)
        # ==============================================================

        print(f"--- Epoch {epoch} Evaluation ---")
        print(f"  目标识别召回率:      {eval_stats['avg_recall']:.4f}")
        print(f"  目标识别虚警率:      {eval_stats['avg_false_alarm']:.4f}")
        print(f"  时空序列稳定性:      {eval_stats['spatial_stability']:.4f}")
        print("-----------------------------")

        if args.output_dir and misc.is_main_process():
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