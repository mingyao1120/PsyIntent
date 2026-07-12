"""
MDID single-GPU evaluation for PsyIntent.

MDID uses pre-extracted ResNet-18 features (.npy) and the user-provided image
caption ("Raw"); there is no image backbone and no emotion encoder. Reports
mAP / Micro-F1 / Macro-F1 / Samples-F1 on fold 0 (the "Val_0 set" comparable to
prior work) with a 0.5 sigmoid decision threshold.
"""

import argparse
import os, sys
import random
import time
import json
import numpy as np

import torch
import torch.nn as nn

from utils.logger import setup_logger
import models
import models.aslloss
from models.query2label_mdid import build_q2l
from utils.metric import voc_mAP
from utils.misc import clean_state_dict

from data_utils.get_dataset_new import get_datasets
from data_utils.metrics import validate_f1


def parser_args():
    parser = argparse.ArgumentParser(description='PsyIntent MDID Single GPU Test')
    parser.add_argument('--dataname', default='MDID', choices=['MDID'])
    parser.add_argument('--dataset_dir', default='')
    parser.add_argument('--img_size_hight', default=224, type=int)
    parser.add_argument('--img_size_weight', default=224, type=int)
    parser.add_argument('--output', default='test_output_mdid')
    parser.add_argument('--resume', default='checkpoint/model_best.pth.tar', type=str)
    parser.add_argument('--num_class', default=7, type=int)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--arch', default='R101-448')
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--eps', default=1e-5, type=float)
    parser.add_argument('--dtgfl', action='store_true', default=False)
    parser.add_argument('--gamma_pos', default=0, type=float)
    parser.add_argument('--gamma_neg', default=2, type=float)
    parser.add_argument('--loss_dev', default=-1, type=float)
    parser.add_argument('--loss_clip', default=0.0, type=float)
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--print-freq', default=5, type=int)
    parser.add_argument('--seed', default=666, type=int)
    parser.add_argument('--local-rank', type=int, default=0)
    # dummy args for compatibility
    parser.add_argument('--optim', default='AdamW')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wd', default=1e-2, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--val_interval', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--resume_omit', nargs='*', default=[])
    parser.add_argument('--ema-decay', default=0.9997, type=float)
    parser.add_argument('--ema-epoch', default=0, type=int)
    parser.add_argument('--cutout', action='store_true', default=False)
    parser.add_argument('--n_holes', type=int, default=1)
    parser.add_argument('--length', type=int, default=-1)
    parser.add_argument('--cut_fact', type=float, default=0.5)
    parser.add_argument('--orid_norm', action='store_true', default=False)
    # Transformer
    parser.add_argument('--enc_layers', default=1, type=int)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dim_feedforward', default=8192, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--pre_norm', action='store_true', default=False)
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--duppos_mode', default='zeros', type=str, choices=('zeros', 'repeat'))
    parser.add_argument('--backbone', default='resnet101', type=str)
    parser.add_argument('--keep_other_self_attn_dec', action='store_true', default=False)
    parser.add_argument('--keep_first_self_attn_dec', action='store_true', default=False)
    parser.add_argument('--keep_input_proj', action='store_true', default=False)
    return parser.parse_args()


def seed_everything(seed):
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
    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=0, color=False, name="MDID")
    logger.info("Command: " + ' '.join(sys.argv))

    # Build model
    model = build_q2l(args)
    model = model.to(device)

    criterion = models.aslloss.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
        disable_torch_grad_focal_loss=args.dtgfl, eps=args.eps)

    # Load checkpoint
    if os.path.isfile(args.resume):
        logger.info(f"=> loading checkpoint '{args.resume}'")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        state_dict = clean_state_dict(ckpt['state_dict'])
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"=> loaded (epoch {ckpt.get('epoch','?')}, best_f1_samples={ckpt.get('best_f1_samples','?')})")
        del ckpt, state_dict
        torch.cuda.empty_cache()
    else:
        logger.info(f"=> no checkpoint found at '{args.resume}'")
        return

    # Load BERT
    from transformers import BertTokenizer, BertModel
    logger.info("Loading BERT...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    bert_model.eval()

    # Load datasets
    train_dataset, val_dataset, test_dataset = get_datasets(args)
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Validate
    logger.info('=== Validation ===')
    _, mAP, f1_micros, f1_macros, f1_samples = validate(val_loader, model, criterion, args, logger, device, tokenizer, bert_model)
    logger.info(f' * val mAP {mAP:.5f}')
    logger.info(f'Val: f1_micros={f1_micros:.5f}, f1_macros={f1_macros:.5f}, f1_samples={f1_samples:.5f}')

    # Test
    logger.info('=== Test ===')
    _, mAP, f1_micros, f1_macros, f1_samples = validate(test_loader, model, criterion, args, logger, device, tokenizer, bert_model)
    logger.info(f' * test mAP {mAP:.5f}')
    logger.info(f'Test: f1_micros={f1_micros:.5f}, f1_macros={f1_macros:.5f}, f1_samples={f1_samples:.5f}')


@torch.no_grad()
def validate(val_loader, model, criterion, args, logger, device, tokenizer, bert_model):
    model.eval()
    saved_data = []

    with torch.no_grad():
        end = time.time()
        for i, (features, target, caption) in enumerate(val_loader):
            # features: (B, 512) flat features from pre-extracted ResNet-18
            features = features.float().to(device)

            # BERT encode
            cap_token = tokenizer(caption, return_tensors='pt', padding=True, truncation=True, max_length=64)
            cap_token = {k: v.to(device) for k, v in cap_token.items()}
            cap_feats = bert_model(**cap_token)

            target = target.float().to(device)

            with torch.amp.autocast('cuda', enabled=args.amp):
                output = model(features, cap_feats)[0]
                loss = criterion(output, target)
                output_sm = torch.sigmoid(output)

            saved_data.append(torch.cat((output_sm.cpu(), target.cpu()), 1))

            if i % args.print_freq == 0:
                logger.info(f'  [{i}/{len(val_loader)}] Loss {loss.item():.3f}')

        saved_data = torch.cat(saved_data, 0).numpy()
        np.savetxt(os.path.join(args.output, 'saved_data_tmp.0.txt'), saved_data)

        mAP, aps = voc_mAP([os.path.join(args.output, 'saved_data_tmp.0.txt')], args.num_class, return_each=True)
        f1_dict = validate_f1([os.path.join(args.output, 'saved_data_tmp.0.txt')], args.num_class)

        logger.info(f"  mAP: {mAP:.5f}")
        logger.info(f"  aps: {np.array2string(aps, precision=5)}")
        logger.info(f"  f1_micros: {f1_dict['val_micro']:.5f}")
        logger.info(f"  f1_macros: {f1_dict['val_macro']:.5f}")
        logger.info(f"  f1_samples: {f1_dict['val_samples']:.5f}")

    return loss.item(), mAP, f1_dict['val_micro'], f1_dict['val_macro'], f1_dict['val_samples']


if __name__ == '__main__':
    main()
