import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import shlex
import textwrap
import subprocess
import random
from pathlib import Path

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import pandas as pd
import torch
import torch.jit
import torch.distributed
import torch.nn as nn
import torch.utils
import torch.utils.data
import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.model_selection import KFold

import models
import dataset


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-df",
        type=str,
        default="./data/features_metadata.csv",
        help="path to train df",
    )
    parser.add_argument(
        "--train-images-dir",
        type=str,
        help="path to train df",
        default="./data/train_features",
    )
    parser.add_argument(
        "--train-labels-dir",
        type=str,
        help="path to train df",
        default="./data/train_agbm",
    )
    parser.add_argument(
        "--test-df",
        type=str,
        default=None,
        help="path to test df",
    )
    parser.add_argument(
        "--test-images-dir",
        type=str,
        help="path to test df",
        default=None,
    )
    parser.add_argument("--checkpoint-dir", type=str, default="logs")

    parser.add_argument("--backbone", type=str, default="tf_efficientnet_b5_ns")
    parser.add_argument("--pretrained", type=str, default="nvidia/segformer-b3-finetuned-ade-512-512")
    parser.add_argument("--dec-attn-type", type=str, default=None)

    parser.add_argument("--loss", type=str, default="rmse", choices=["rmse", "nrmse", "mae", "nmae", "huber", "smoothl1", "rmsefocal"])
    parser.add_argument("--loss-nonan", action="store_true", help="loss without outlyers")
    parser.add_argument("--in-channels", type=int, default=15)
    parser.add_argument("--out-indices", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--dec-channels", type=int, nargs="+", default=[256, 240, 224, 208, 192])
    parser.add_argument("--n-classes", type=int, default=1)

    parser.add_argument("--optim", type=str, default="adamw", help="optimizer name")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lr-decay-scale", type=float, default=1e-2)
    parser.add_argument("--warmup-steps-ratio", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)

    parser.add_argument("--mix-beta", type=float, default=1)

    parser.add_argument("--scheduler", type=str, default="cosa", help="scheduler name")
    parser.add_argument("--scheduler-mode", type=str, default="epoch", choices=["step", "epoch"], help="scheduler mode")
    parser.add_argument("--T-max", type=int, default=440)
    parser.add_argument("--eta-min", type=int, default=0)

    parser.add_argument("--mixup", type=float, default=0)
    parser.add_argument("--cutmix", type=float, default=0)
    parser.add_argument("--mix-y", action="store_true", help="mix target")

    parser.add_argument("--augs", action="store_true", help="augmentations")

    parser.add_argument(
        "--num-workers", type=int, help="number of data loader workers", default=8,
    )
    parser.add_argument(
        "--num-epochs", type=int, help="number of epochs to train", default=440,
    )
    parser.add_argument("--batch-size", type=int, help="batch size", default=32)
    parser.add_argument(
        "--random-state",
        type=int,
        help="random seed",
        default=314159,
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        help="number of folds",
        default=10,
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="fold",
        default=0,
    )

    parser.add_argument(
        "--distributed", action="store_true", help="distributed training"
    )
    parser.add_argument("--syncbn", action="store_true", help="sync batchnorm")
    parser.add_argument(
        "--deterministic", action="store_true", help="deterministic training"
    )
    parser.add_argument(
        "--load", type=str, default="", help="path to pretrained model weights"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="path to pretrained model to resume training",
    )
    parser.add_argument("--fp16", action="store_true", help="fp16 training")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--img-size", type=int, nargs=2, default=dataset.IMG_SIZE)
    parser.add_argument("--ft", action="store_true", help="finetune")

    args = parser.parse_args(args=args)

    return args


def notify(title, summary, value=-1, server="sun"):
    print(title, summary)
    return
    cmd = textwrap.dedent(
        f"""
            ssh {server} \
                '\
                    export DISPLAY=:0 \
                    && dunstify -t 0 -h int:value:{value} "{title}" "{summary}" \
                '
        """
    )
    cmds = shlex.split(cmd)
    with subprocess.Popen(cmds, start_new_session=True):
        pass


def epoch_step_train(loader, desc, model, criterion, optimizer, scaler, fp16=False, grad_accum=1, local_rank=0):
    model.train()

    if local_rank == 0:
        pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=False, mininterval=2)

    loc_loss = n = 0
    for images, mask, target in loader:
        images = images.cuda(local_rank, non_blocking=True)
        mask = mask.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=fp16):
            logits = model(images, mask)
            loss = criterion(logits, target)
            # loss = loss / grad_accum

        scaler.scale(loss).backward()

        #if i % grad_accum == 0 or i == len(loader):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
            # error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        bs = target.size(0)
        loc_loss += loss.item() * bs# * grad_accum
        n += bs

        torch.cuda.synchronize()

        if local_rank == 0:
            postfix = {
                "loss": f"{loc_loss / n:.3f}",
            }
            pbar.set_postfix(**postfix)
            pbar.update()

        if np.isnan(loc_loss) or np.isinf(loc_loss):
            break

    if local_rank == 0:
        pbar.close()

    return loc_loss, n


@torch.no_grad()
def epoch_step_val(loader, desc, model, criterion, local_rank=0):
    model.eval()

    if local_rank == 0:
        pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=False, mininterval=2)

    loc_loss = n = 0
    loc_loss_tta = {f"tta_{i}": 0 for i in range(1, 4 + 1)}
    for images, mask, target in loader:
        images = images.cuda(local_rank, non_blocking=True)
        mask = mask.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        logits = model(images, mask)
        loss = criterion(logits, target)
        bs = target.size(0)
        n += bs
        loc_loss += loss.item() * bs
        loc_loss_tta["tta_1"] += loss.item() * bs

        logits += torch.flip(model(torch.flip(images, dims=[-1]), mask), dims=[-1])
        loss = criterion(logits / 2, target)
        loc_loss_tta["tta_2"] += loss.item() * bs

        logits += torch.flip(model(torch.flip(images, dims=[-2]), mask), dims=[-2])
        loss = criterion(logits / 3, target)
        loc_loss_tta["tta_3"] += loss.item() * bs

        logits += torch.flip(model(torch.flip(images, dims=[-2, -1]), mask), dims=[-2, -1])
        loss = criterion(logits / 4, target)
        loc_loss_tta["tta_4"] += loss.item() * bs

        torch.cuda.synchronize()

        if local_rank == 0:
            postfix = {
                "loss": f"{loc_loss / n:.3f}",
            }
            pbar.set_postfix(**postfix)
            pbar.update()

    if local_rank == 0:
        pbar.close()

    #grid = torchvision.utils.make_grid(images[0, :, [6, 5, 4]], nrow=4)
    #summary_writer.add_image('dev/images', grid, epoch)
    #summary_writer.add_image('dev/target', (target[0] - target[0].min()) / (target[0].max() - target[0].min()), epoch, dataformats="HW")
    #summary_writer.add_image('dev/preds', (logits[0, 0] - logits[0, 0].min()) / (logits[0, 0].max() - logits[0, 0].min()), epoch, dataformats="HW")

    #fig = plt.figure(figsize=(12, 12))
    #ax = fig.add_subplot(1, 1, 1)
    #x = target[0].cpu().numpy().ravel()
    #y = logits[0, 0].detach().cpu().numpy().ravel()
    #ax.set_title(f"RMSE: {np.sqrt(((x - y) ** 2).mean()):.3f}")
    #ax.plot(np.arange(x.max()), np.arange(x.max()))
    #ax.scatter(x, y)
    #summary_writer.add_figure('dev/hist', fig, epoch)
    #plt.close(fig)

    return loc_loss, n, loc_loss_tta


def train_dev_split(df, args):
    df["fold"] = None

    n_col = len(df.columns) - 1
    skf = KFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.random_state
    )
    for fold, (_, dev_index) in enumerate(skf.split(df)):
        df.iloc[dev_index, n_col] = fold

    train, dev = (
        df[df.fold != args.fold].copy(),
        df[df.fold == args.fold].copy(),
    )

    return train, dev


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_dist(args):
    # to autotune convolutions and other algorithms
    # to pick the best for current configuration
    torch.backends.cudnn.benchmark = True

    if args.deterministic:
        set_seed(args.random_state)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_printoptions(precision=10)

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    args.world_size = 1
    if args.distributed:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."


def add_weight_decay(args, model, weight_decay=1e-5, skip_list=()):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    params = [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]

    return params


def save_jit(model, args, model_path):
    if hasattr(model, "module"):
        model = model.module

    if args.backbone.startswith("efficientnet"):
        model.encoder.set_swish(memory_efficient=False)

    model.eval()
    inp = torch.rand(1, 12, args.in_channels, args.img_size[0], args.img_size[1]).cuda(int(os.environ.get("LOCAL_RANK", 0)))

    with torch.no_grad():
        traced_model = torch.jit.trace(model, inp)

    traced_model = torch.jit.freeze(traced_model)

    traced_model.save(model_path)

    if args.backbone.startswith("efficientnet"):
        model.encoder.set_swish(memory_efficient=True)


def all_gather(value, n, is_dist):
    if is_dist:
        vals = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(vals, value)
        ns = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(ns, n)

        n = sum(ns)
        if isinstance(value, dict):
            val = {
                k: sum(val[k] for val in vals) / n
                for k in value
            }
        else:
            val = sum(vals) / n
    elif isinstance(value, dict):
        val = {k: v / n for k, v in value.items()}
    else:
        val = value / n

    return val


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        diff = logits.squeeze(1) - target
        diff2 = torch.square(diff)
        diff2m = diff2.mean((-1, -2))
        diff2msqrt = torch.sqrt(diff2m)

        return diff2msqrt.mean(0)


class NoNaNRMSE(nn.Module):
    def __init__(self, threshold=400):
        super().__init__()

        self.threshold = threshold

    def forward(self, logits, target):
        not_nan = target < self.threshold

        logits = logits.squeeze(1)
        diff = logits - target
        diff[~not_nan] = 0
        diff2 = torch.square(diff)
        diff2m = (diff2 / not_nan.sum((-1, -2), keepdim=True)).sum((-1, -2))
        diff2msqrt = torch.sqrt(diff2m)

        rmse = diff2msqrt.mean(0)

        return rmse


class NoNaNMAE(nn.Module):
    def __init__(self, threshold=400):
        super().__init__()

        self.threshold = threshold
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, logits, target):
        is_nan = target >= self.threshold

        logits = logits.squeeze(1)
        loss = self.loss(logits, target)
        loss[is_nan] = 0
        loss = loss.sum((-1, -2)) / torch.sqrt((~is_nan).sum((-1, -2)))
        loss = loss.mean(0)

        return loss


class NoNanLoss(nn.Module):
    def __init__(self, loss, threshold=1000):
        super().__init__()

        self.loss = loss
        self.threshold = threshold

    def forward(self, logits, target):
        not_nan = target < self.threshold
        logits = logits.squeeze(1)
        logits = logits.masked_select(not_nan)
        target = target.masked_select(not_nan)
        loss = self.loss(logits, target)

        return loss


def weighted_focal_mse_loss(inputs, targets, activate="sigmoid", beta=.2, gamma=1, weights=None):
    diff = inputs - targets
    loss = diff ** 2
    adiff = torch.abs(diff)
    badiff = beta * adiff
    if activate == "tanh":
        loss *= torch.tanh(badiff)
    else:
        loss *= 2 * torch.sigmoid(badiff) - 1

    loss = loss ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)

    loss = torch.mean(loss)

    return loss


class RMSEFocal(nn.Module):
    def __init__(self, beta=0.2, w=0.05):
        super().__init__()

        self.beta = beta
        self.w = w

    def forward(self, logits, target):
        diff = logits.squeeze(1) - target
        diff2 = torch.square(diff)
        diff2m = diff2.mean((-1, -2))
        diff2msqrt = torch.sqrt(diff2m)
        rmse = diff2msqrt.mean(0)

        adiff = torch.abs(diff)
        badiff = self.beta * adiff
        focal = diff2 * (2 * torch.sigmoid(badiff) - 1)
        focal = torch.mean(focal)

        return rmse + self.w * focal


def train(args):
    init_dist(args)

    torch.backends.cudnn.benchmark = True

    checkpoint_dir = Path(args.checkpoint_dir)
    summary_writer = None
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(args)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        summary_writer = SummaryWriter(checkpoint_dir / "logs")

        notify(
            checkpoint_dir.name,
            f"start training",
        )

    model = build_model(args)
    model = model.cuda(local_rank)

    checkpoint = None
    if args.load:
        path_to_resume = Path(args.load).expanduser()
        if path_to_resume.is_file():
            print(f"=> loading resume checkpoint '{path_to_resume}'")
            checkpoint = torch.load(
                path_to_resume,
                map_location="cpu",
            )

            nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint["state_dict"], "module.")
            model.load_state_dict(checkpoint["state_dict"])
            print(
                f"=> resume from checkpoint '{path_to_resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            print(f"=> no checkpoint found at '{path_to_resume}'")

    model = model.to(memory_format=torch.channels_last)

    weight_decay = args.weight_decay
    if weight_decay > 0:  # and filter_bias_and_bn:
        skip = {}
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

        parameters = add_weight_decay(args, model, weight_decay, skip)
        weight_decay = 0.0
    else:
        parameters = model.parameters()

    optimizer = build_optimizer(parameters, args)

    if args.distributed:
        if args.syncbn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    df: pd.DataFrame = pd.read_csv(args.train_df)
    test_df = df[df.split == "test"].copy()
    df = df[df.split == "train"].copy()
    df = df.groupby("chip_id").agg(list).reset_index()

    train_df, dev_df = train_dev_split(df, args)

    if args.ft:
        train_df = pd.concat([train_df, dev_df])

    if local_rank == 0:
        print(train_df)
        print(dev_df)

    train_images_dir = Path(args.train_images_dir)
    train_labels_dir = Path(args.train_labels_dir)
    train_dataset = dataset.DS(
        df=train_df,
        dir_features=train_images_dir,
        dir_labels=train_labels_dir,
        augs=args.augs,
    )

    if args.test_df is not None:
        test_dataset =dataset.DS(
            df=test_df,
            dir_features=train_images_dir,
            dir_labels=train_labels_dir,
        )
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    val_dataset = dataset.DS(
        df=dev_df,
        dir_features=train_images_dir,
        dir_labels=train_labels_dir,
    )

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    args.num_workers = min(args.batch_size, 2)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=None,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True,
    )
    val_batch_size = args.batch_size
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=None,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=False,
    )

    scheduler = build_scheduler(optimizer, args, n=len(train_loader) if args.scheduler_mode == "step" else 1)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    criterion = build_criterion(args)
    metric = RMSE()

    def saver(path, score):
        torch.save(
            {
                "epoch": epoch,
                "best_score": best_score,
                "score": score,
                "state_dict": model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
                "sched_state_dict": scheduler.state_dict()
                if scheduler is not None
                else None,
                "scaler": scaler.state_dict(),
                "args": args,
            },
            path,
        )

    res = 0
    start_epoch = 0
    best_score = float("+inf")
    if args.resume and checkpoint is not None:
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        if checkpoint["sched_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["sched_state_dict"])

        optimizer.load_state_dict(checkpoint["opt_state_dict"])
        scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(start_epoch, args.num_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        desc = f"{epoch}/{args.num_epochs}"

        train_loss, n = epoch_step_train(
            train_loader,
            desc,
            model,
            criterion,
            optimizer,
            scaler,
            fp16=args.fp16,
            grad_accum=args.grad_accum,
            local_rank=local_rank,
        )

        train_loss = all_gather(train_loss, n, args.distributed)
        if np.isnan(train_loss) or np.isinf(train_loss):
            res = 1
            break

        dev_loss, n, dev_losses = epoch_step_val(
            val_loader,
            desc,
            model,
            metric,
            local_rank=local_rank,
        )

        dev_loss = all_gather(dev_loss, n, args.distributed)
        dev_losses = all_gather(dev_losses, n, args.distributed)

        if scheduler is not None and args.scheduler_mode == "epoch":
            scheduler.step()

        if local_rank == 0:
            for idx, param_group in enumerate(optimizer.param_groups):
                lr = param_group["lr"]
                summary_writer.add_scalar(
                    "group{}/lr".format(idx), float(lr), global_step=epoch
                )

            summary_writer.add_scalar("loss/train_loss", train_loss, global_step=epoch)
            score = min(dev_losses.values())
            summary_writer.add_scalar("loss/dev_loss", score, global_step=epoch)
            summary_writer.add_scalars("loss/dev_losses", dev_losses, global_step=epoch)

            if score < best_score:
                notify(
                    checkpoint_dir.name,
                    f"epoch {epoch}: new score {score:.3f} (old {best_score:.3f}, diff {abs(score - best_score):.3f})",
                    int(100 * (epoch / args.num_epochs)),
                )
                best_score = score

                saver(checkpoint_dir / "model_best.pth", best_score)
                # save_jit(model, args, checkpoint_dir / f"model_best.pt")
                if hasattr(model, "module"):
                    torch.save(model.module, checkpoint_dir / f"modelo_best.pth")
                else:
                    torch.save(model, checkpoint_dir / f"modelo_best.pth")

            saver(checkpoint_dir / "model_last.pth", score)
            # save_jit(model, args, checkpoint_dir / "model_last.pt")

            if epoch % (2 * args.T_max) == (args.T_max - 1):
                saver(checkpoint_dir / f"model_last_{epoch + 1}.pth", score)
                # save_jit(model, args, checkpoint_dir / f"model_last_{epoch + 1}.pt")

        torch.cuda.empty_cache()

    if local_rank == 0:
        summary_writer.close()

        notify(
            checkpoint_dir.name,
            f"finished training with score {score:.3f} (best {best_score:.3f}) on epoch {epoch}",
        )

    return res


def build_model(args):
    model = models.UnetVFLOW(args)

    return model


def build_criterion(args):
    if args.loss == "rmse":
        criterion = RMSE()
    elif args.loss == "nrmse":
        criterion = NoNaNRMSE()
    elif args.loss == "rmsefocal":
        criterion = RMSEFocal()
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    elif args.loss == "nmae":
        criterion = NoNaNMAE()
    elif args.loss == "huber":
        criterion = nn.HuberLoss()
    elif args.loss == "smoothl1":
        criterion = nn.SmoothL1Loss()

    if args.loss_nonan:
        criterion = NoNanLoss(criterion)

    return criterion


def build_optimizer(parameters, args):
    if args.optim.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optim.lower() == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError(f"not yet implemented {args.optim}")

    return optimizer


def build_scheduler(optimizer, args, n=1):
    scheduler = None

    if args.scheduler.lower() == "cosa":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.T_max * n,
            eta_min=args.eta_min if args.eta_min > 0 else max(args.learning_rate * 1e-1, 5e-5),
        )
    elif args.scheduler.lower() == "cosawr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.T_max,
            T_mult=1,
            eta_min=args.eta_min if args.eta_min > 0 else max(args.learning_rate * 1e-1, 5e-5),
        )
    else:
        print("No scheduler")

    return scheduler


def main():
    args = parse_args()

    train(args)


if __name__ == "__main__":
    main()
