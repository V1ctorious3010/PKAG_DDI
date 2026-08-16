import os
import os.path as osp
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

import json
from datasets.builder import build_dataset
from models.builder import build_model
from utils import mkdir_or_exist

os.environ['OPENBLAS_NUM_THREADS'] = '0'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
torch.set_float32_matmul_precision('medium')  # can be medium (bfloat16), high (tensorfloat32), highest (float32)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', help='the config file path')
    parser.add_argument('--devices', default='0,1', help='0,1')
    parser.add_argument('--seednumber', default=42, help='number of seeds')
    parser.add_argument('--save_every_n_epochs', default=10, help='save the model every n_epochs')
    parser.add_argument('--caption_eval_epoch', type=int, default=10)
    parser.add_argument('--opt_model', type=str, default="facebook/galactica-1.3b")
    parser.add_argument('--mode', type=str, default="ft", help='pretrain or ft or breakpoint')
    parser.add_argument('--strategy_name', type=str, default=None)
    parser.add_argument('--work_dir', default=False, help='this is the saved checkpoints for test')
    parser.add_argument('--breakpoint_file', default=False, help='breakpoint retrainin')
    parser.add_argument('--peft_config', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size in config')
    parser.add_argument('--accumulate_grad_batches', type=int, default=None, help='Gradient accumulation steps')

    args = parser.parse_args()

    return args


def main(args):
    pl.seed_everything(args.seednumber)  # set seed
    args = parse_args()  # get args
    cfg = json.load(open(args.config))  # get cfg from file

    if args.batch_size is not None:
        cfg["dataset"]["batch_size"] = args.batch_size
        cfg["model"]["batch_size"] = args.batch_size
        print(f"[Config] Overriding batch_size to {args.batch_size}")

    accumulate_grad_batches = args.accumulate_grad_batches or cfg.get("accumulate_grad_batches", 1)

    # Create output file
    work_dir = os.path.join('work_dirs', osp.splitext(osp.basename(args.config))[0])
    cfg['work_dir'] = work_dir
    cfg["model"]["work_dir"] = work_dir
    cfg["model"]["caption_eval_epoch"] = cfg["caption_eval_epoch"]

    callbacks = []  # callbacks which can save checkpoints auto and continue training
    earlystopping = EarlyStopping('molecule loss', patience=cfg["patience"],
                                  min_delta=cfg["min_delta"], mode="min")

    callbacks.append(plc.ModelCheckpoint(dirpath=work_dir,
                                         filename='{epoch:02d}',
                                         #every_n_epochs=cfg["save_every_n_epochs"],
                                         save_top_k=-1
                                         #save_last=True
                                        ))
    callbacks.append(earlystopping)

    logger = CSVLogger(save_dir=work_dir, name="logs")  

    # Get model
    if args.mode == "ft":
        model = build_model(args, cfg["model"])
        init_ckpt = cfg.get("init_checkpoint", "")
        if init_ckpt and os.path.exists(init_ckpt):
            ckpt = torch.load(init_ckpt, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'], strict=False)
            print(f"Loaded init checkpoint from {init_ckpt}")
        else:
            print(f"init_checkpoint '{init_ckpt}' not found. Training without pre-trained Stage 2 checkpoint.")
        print('total params:', sum(p.numel() for p in model.parameters()))
    elif args.mode == "breakpoint":
        print("breakpoint")
        print("args.breakpoint_file", args.breakpoint_file)
        model = build_model(args, cfg["model"]).load_from_checkpoint(args.breakpoint_file, strict=False, args=args, cfg=cfg["model"])
    elif args.work_dir:
        model = build_model(args, cfg["model"]).load_from_checkpoint(args.work_dir, strict=False, args=args, cfg=cfg["model"])
        print(f"loaded init checkpoint from {args.work_dir}")

    else:
        # few shot other datasets
        model = build_model(args, cfg["model"])

    # Get dataset, becasue the tokenizer is load from model, therefore when construct the dataset, must input the tokenizer
    if args.opt_model.find('galactica') >= 0:
        tokenizer = model.blip2opt.opt_tokenizer
    elif args.opt_model.find('llama') >= 0 or args.opt_model.find('vicuna') >= 0:
        tokenizer = model.blip2opt.llm_tokenizer
    else:
        raise NotImplementedError
    datasets = build_dataset(cfg["dataset"], args, tokenizer)

    # multi device Parallel strategy & hardware detection
    num_available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_list = [int(d) for d in str(args.devices).split(',') if d.strip().isdigit()]
    
    if num_available_gpus > 1 and len(device_list) > 1:
        devices = device_list
        accelerator = "gpu"
        if args.strategy_name == 'fsdp':
            strategy = 'fsdp'
        elif args.strategy_name == 'deepspeed':
            strategy = 'deepspeed_stage_3'
        else:
            strategy = 'ddp_find_unused_parameters_true'
    elif num_available_gpus >= 1:
        print("Single GPU mode")
        devices = [device_list[0]] if device_list else [0]
        accelerator = "gpu"
        strategy = "auto"
    else:
        print("CPU mode")
        devices = 1
        accelerator = "cpu"
        strategy = "auto"

    precision = args.precision
    if accelerator == "cpu" and precision in ["16", "bf16", "16-mixed", "bf16-mixed"]:
        precision = "32"
    elif accelerator == "gpu" and precision in ["bf16", "16"]:
        # Handle PyTorch Lightning precision strings (16-mixed / bf16-mixed)
        try:
            precision = "bf16-mixed" if (precision == "bf16" and torch.cuda.is_bf16_supported()) else "16-mixed"
        except Exception:
            precision = "16-mixed"

    max_epochs = cfg.get("max_epochs", 30)

    # start to train
    if args.mode in {'pretrain', 'ft'}:
        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            callbacks=callbacks,
            strategy=strategy,
            logger=logger,
            precision=precision,
            accumulate_grad_batches=accumulate_grad_batches,
        )
        trainer.fit(model, datamodule=datasets)
    elif args.mode == "breakpoint":
        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            callbacks=callbacks,
            strategy=strategy,
            logger=logger,
            precision=precision,
            accumulate_grad_batches=accumulate_grad_batches,
        )
        trainer.fit(model, datamodule=datasets, ckpt_path=args.breakpoint_file if args.breakpoint_file else None)
    elif args.mode == 'eval':
        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            strategy=strategy,
            logger=logger,
            precision=precision,
        )
        trainer.test(model, datamodule=datasets, ckpt_path=args.work_dir if args.work_dir else None)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main(parse_args())


