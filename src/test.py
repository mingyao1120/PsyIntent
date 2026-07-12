"""
Single-GPU evaluation for PsyIntent (Intentonomy and MET-MeMe).

Loads a trained checkpoint and reports mAP / Micro-F1 / Macro-F1 / Samples-F1 on
both the validation and the test splits. The MLLM psychological analysis is
loaded from the ``*_janus7b_psy.json`` annotation files (Intentonomy and
MET-MeMe) and encoded by BERT; all metrics use a 0.5 sigmoid decision threshold.
"""

import argparse
import os, sys
import random
import datetime
import time
from typing import List
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import _init_paths

from utils.logger import setup_logger
import models
import models.aslloss
from models.query2label import build_q2l
from utils.metric import voc_mAP
from utils.misc import clean_state_dict

from data_utils.get_dataset_new import get_datasets
from data_utils.metrics import validate_f1


def parser_args():
    parser = argparse.ArgumentParser(description='PsyIntent Testing (Single GPU)')
    available_models = ['R101-448', 'R101-576', 'R50']
    parser.add_argument('--dataname', help='dataname', default='intentonomy', choices=['intentonomy', 'METMEME'])
    parser.add_argument('--dataset_dir', help='dir of dataset', default='')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--img_size_hight', default=224, type=int)
    parser.add_argument('--img_size_weight', default=224, type=int)
    parser.add_argument('--output', metavar='DIR', default='test_output')
    parser.add_argument('--resume', default='checkpoint/model_best.pth.tar', type=str, metavar='PATH')
    parser.add_argument('--num_class', default=28, type=int)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=False)
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'])
    parser.add_argument('-a', '--arch', metavar='ARCH', default='R101-448', choices=available_models)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--eps', default=1e-5, type=float)
    parser.add_argument('--dtgfl', action='store_true', default=False)
    parser.add_argument('--gamma_pos', default=0, type=float)
    parser.add_argument('--gamma_neg', default=2, type=float)
    parser.add_argument('--loss_dev', default=-1, type=float)
    parser.add_argument('--loss_clip', default=0.0, type=float)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--val_interval', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int)
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', default=True, action='store_true')
    parser.add_argument('--ema-decay', default=0.9997, type=float)
    parser.add_argument('--ema-epoch', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--cutout', action='store_true', default=False)
    parser.add_argument('--n_holes', type=int, default=1)
    parser.add_argument('--length', type=int, default=-1)
    parser.add_argument('--cut_fact', type=float, default=0.5)
    parser.add_argument('--orid_norm', action='store_true', default=False)
    # Transformer
    parser.add_argument('--enc_layers', default=1, type=int)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dim_feedforward', default=8192, type=int)
    parser.add_argument('--hidden_dim', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--pre_norm', action='store_true', default=False)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine',))
    parser.add_argument('--duppos_mode', default='zeros', type=str, choices=('zeros', 'repeat'))
    parser.add_argument('--backbone', default='resnet101', type=str)
    parser.add_argument('--keep_other_self_attn_dec', action='store_true', default=False)
    parser.add_argument('--keep_first_self_attn_dec', action='store_true', default=False)
    parser.add_argument('--keep_input_proj', action='store_true', default=False)
    parser.add_argument('--slot_iters', default=3, type=int)

    args = parser.parse_args()
    return args


def seed_everything(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def main():
    args = parser_args()
    seed_everything(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=0, color=False, name="Q2L")
    logger.info("Command: " + ' '.join(sys.argv))
    logger.info("Single-GPU test mode")

    # ========== Build model ==========
    model = build_q2l(args)
    model = model.to(device)

    criterion = models.aslloss.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
        disable_torch_grad_focal_loss=args.dtgfl,
        eps=args.eps,
    )

    # ========== Load checkpoint ==========
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        state_dict = clean_state_dict(checkpoint['state_dict'])
        model.load_state_dict(state_dict, strict=False)  # allow mismatched norm keys
        logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint.get('epoch', '?')})")
        logger.info(f"   best_f1_samples: {checkpoint.get('best_f1_samples', '?')}")
        del checkpoint, state_dict
        torch.cuda.empty_cache()
    else:
        logger.info(f"=> no checkpoint found at '{args.resume}'")
        return

    # ========== Load BERT ==========
    from transformers import BertTokenizer, BertModel
    logger.info("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    bert_model.eval()
    logger.info("BERT loaded.")

    # ========== Load datasets ==========
    train_dataset, val_dataset, test_dataset = get_datasets(args)
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        # ========== Validate ==========
        logger.info('=== Validation ===')
        _, mAP, f1_micros, f1_macros, f1_samples = validate(
            val_loader, model, criterion, args, logger, device, tokenizer, bert_model)
        logger.info(f' * val mAP {mAP:.5f}')
        logger.info(f'Val: f1_micros={f1_micros}, f1_macros={f1_macros}, f1_samples={f1_samples}')

        # ========== Test ==========
        logger.info('=== Test ===')
        _, mAP, f1_micros, f1_macros, f1_samples = validate(
            test_loader, model, criterion, args, logger, device, tokenizer, bert_model)
        logger.info(f' * test mAP {mAP:.5f}')
        logger.info(f'Test: f1_micros={f1_micros}, f1_macros={f1_macros}, f1_samples={f1_samples}')

        return


@torch.no_grad()
def validate(val_loader, model, criterion, args, logger, device, tokenizer, bert_model):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    model.eval()
    saved_data = []
    ww = 0

    with torch.no_grad():
        end = time.time()
        for i, (images, target, caption) in enumerate(val_loader):

            # BERT encode the MLLM psychological analysis
            cap_token = tokenizer(caption, return_tensors='pt', padding=True, truncation=True, max_length=64)
            cap_token = {key: value.to(device) for key, value in cap_token.items()}
            with torch.no_grad():
                cap_feats = bert_model(**cap_token)

            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.amp.autocast('cuda', enabled=args.amp):
                start_time = time.time()
                output = model(images, cap_feats)[0]
                end_time = time.time()
                ww += (end_time - start_time)
                loss = criterion(output, target)
                output_sm = nn.functional.sigmoid(output)

            losses.update(loss.item(), images.size(0))
            mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

            _item = torch.cat((output_sm.detach().cpu(), target.detach().cpu()), 1)
            saved_data.append(_item)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, logger)

        logger.info('=> Done inference, calculating metrics...')

        # Calculate metrics
        saved_data = torch.cat(saved_data, 0).numpy()
        saved_name = 'saved_data_tmp.0.txt'
        np.savetxt(os.path.join(args.output, saved_name), saved_data)

        filenamelist = [saved_name]
        mAP, aps = voc_mAP([os.path.join(args.output, f) for f in filenamelist],
                           args.num_class, return_each=True)
        f1_dict = validate_f1([os.path.join(args.output, f) for f in filenamelist], args.num_class)
        f1_micros = f1_dict['val_micro']
        f1_macros = f1_dict['val_macro']
        f1_samples = f1_dict['val_samples']

        logger.info(f"  mAP: {mAP:.5f}")
        logger.info(f"  aps: {np.array2string(aps, precision=5)}")
        logger.info(f"  f1_micros: {f1_micros:.5f}")
        logger.info(f"  f1_macros: {f1_macros:.5f}")
        logger.info(f"  f1_samples: {f1_samples:.5f}")

        ee = ww / max(len(val_loader), 1)
        logger.info(f"Avg inference time per batch: {ee:.4f}s")

    return losses.avg, mAP, f1_micros, f1_macros, f1_samples


class AverageMeter(object):
    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
