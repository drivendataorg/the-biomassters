import os
import torch
import pandas as pd

from ..models import define_model
from .losses import RecLoss, SimLoss
from .scheduler import WarmupCosineAnnealingLR


class BMBaseTrainer(object):

    def __init__(self, configs, exp_dir, resume=False):

        # creates dirs for saving outputs
        self.configs    = configs
        self.exp_dir    = exp_dir
        self.ckpts_dir  = os.path.join(self.exp_dir, 'ckpts')
        self.logs_path  = os.path.join(self.exp_dir, 'logs.csv')
        self.model_path = os.path.join(self.exp_dir, 'model.pth')

        os.makedirs(self.ckpts_dir, exist_ok=True)
        if os.path.isfile(self.logs_path) and (not resume):
            os.remove(self.logs_path)

        # trainer parameters
        self.start_epoch = 1
        self.epochs      = configs.trainer.epochs
        self.ckpt_freq   = configs.trainer.ckpt_freq
        self.accum_iter  = configs.trainer.accum_iter
        self.print_freq  = configs.trainer.print_freq
        self.device      = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.early_stop  = configs.trainer.get('early_stop', None)

        # instantiates model
        self.model = define_model(configs.model)
        self.model = self.model.to(self.device)

        # instantiates reconstructrion loss
        rec_kwargs = configs.loss.get('rec', None)
        self.rec_loss_func = None
        if rec_kwargs is not None:
            self.rec_loss_func = RecLoss(**rec_kwargs)

        # instantiates similarity loss
        sim_kwargs = configs.loss.get('sim', None)
        self.sim_loss_func = None
        if sim_kwargs is not None:
            self.sim_loss_func = SimLoss(**sim_kwargs)
            self.sim_loss_func.to(self.device)

        # instantiates optimizer
        if configs.optimizer.mode == 'adamw':
            optimizer = torch.optim.AdamW
        elif configs.optimizer.mode == 'adam':
            optimizer = torch.optim.Adam
        else:
            raise ValueError('unknown optimizer mode')

        lr           = configs.optimizer.lr
        betas        = configs.optimizer.betas
        amsgrad      = configs.optimizer.amsgrad
        weight_decay = configs.optimizer.weight_decay
        self.optimizer = optimizer(
            self.model.parameters(), lr=lr, betas=betas,
            amsgrad=amsgrad, weight_decay=weight_decay
        )

        # instantiates scheduler
        min_lr = configs.scheduler.min_lr
        warmup = configs.scheduler.warmup
        self.scheduler = WarmupCosineAnnealingLR(
            self.optimizer, max_lr=lr, min_lr=min_lr,
            total=self.epochs, warmup=warmup
        )

        # loads checkpoint
        self._load_checkpoint(resume)

    def _load_checkpoint(self, resume):
        if not resume:
            return

        # find checkpoint from ckpt_dir
        ckpt_files = os.listdir(self.ckpts_dir)
        ckpt_files = [f for f in ckpt_files if f.startswith('ckpt')]
        if len(ckpt_files) > 0:
            ckpt_files.sort()
            ckpt_file = ckpt_files[-1]
            ckpt_path = os.path.join(self.ckpts_dir, ckpt_file)
        else:
            print('>>> No checkpoint was found, ignore', '\n')
            return

        try:
            print('>>> Resume checkpoint from:', ckpt_path, '\n')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            self.start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception:
            print('>>> Faild to resume checkpoint')

    def _save_checkpoint(self, epoch):
        ckpt_file = f'ckpt-{epoch:06d}.ckpt'
        ckpt_path = os.path.join(self.ckpts_dir, ckpt_file)
        ckpt = {
            'epoch':     epoch,
            'model':     self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(ckpt, ckpt_path)

    def _save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def _save_logs(self, epoch, train_metrics, val_metrics=None):
        log_stats = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        df = pd.DataFrame(log_stats, index=[0])

        if not os.path.isfile(self.logs_path):
            df.to_csv(self.logs_path, index=False)
        else:
            df.to_csv(self.logs_path, index=False, mode='a', header=False)
