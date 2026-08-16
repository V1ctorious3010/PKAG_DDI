import os
import os.path as osp
from typing import Any, Dict
import torch

from .blip2_function_retrieval import Blip2OPT_RETRIEVAL, Blip2OPT_RETRIEVAL_marginalize

import pytorch_lightning as pl
from torch import optim
try:
    from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
except ImportError:
    LinearWarmupCosineLRScheduler = None
    LinearWarmupStepLRScheduler = None

import json
from utils import results_metrics, caption_evaluate, AttrDict, do_compute_metrics
import torch.distributed as dist
from peft import LoraConfig, TaskType
import numpy as np

MODELS = {
    "retrieval": Blip2OPT_RETRIEVAL,
    "marginalize": Blip2OPT_RETRIEVAL_marginalize,
}


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


class MainModel_Function_gen(pl.LightningModule):
    def __init__(self, args, cfg):
        super().__init__()
        self.args = args
        self.cfg = cfg
        self.caption_eval_epoch = cfg.get("caption_eval_epoch", 1)
        self.stage = cfg.get("stage", "marginalize")

        # for generate text
        self.do_sample = cfg.get("do_sample", False)
        self.num_beams = cfg.get("num_beams", 5)
        self.max_len = cfg.get("generate_max_len", 50)
        self.min_len = cfg.get("generate_min_len", 5)
        self.temperature = cfg.get("temperature", 0.1)
        self.top_p = cfg.get("top_p", 0.9)
        self.max_new_tokens = cfg.get("max_new_tokens", 25)
        self.repetition_penalty = cfg.get("repetition_penalty", 1.0)
        self.batch_size = cfg.get("batch_size", 4)

        # for the opt_model
        self.llm_tune = cfg.get("llm_tune", "lora")
        self.peft_dir = cfg.get("peft_dir", "")
        self.opt_model = cfg.get("opt_model", "facebook/galactica-1.3b")

        if self.stage in MODELS:
            self.blip2opt = MODELS[self.stage](cfg)
        else:
            self.blip2opt = Blip2OPT_RETRIEVAL_marginalize(cfg)

        self.tokenizer = self.blip2opt.init_tokenizer()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.scheduler = None

        try:
            self.save_hyperparameters(args)
        except Exception:
            pass

    def training_step(self, batch, batch_idx):
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        loss = self.blip2opt(batch)
        self.log("molecule loss", float(loss['loss']), batch_size=self.batch_size, sync_dist=True)
        try:
            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=self.batch_size, sync_dist=True)
        except Exception:
            pass
        return loss['loss']

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if (self.current_epoch + 1) % self.caption_eval_epoch != 0:
            return None

        loss = self.blip2opt(batch[:-2])
        self.log("val_loss", float(loss['loss']), batch_size=self.batch_size, sync_dist=True)

        CoT_pred, predictions, drug_name1, drug_name2 = self.blip2opt.generate(
            batch,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            min_length=self.min_len,
            max_new_tokens=self.max_new_tokens
        )
        out = (CoT_pred, predictions, batch[-2], batch[-1])
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        if hasattr(self, 'trainer') and getattr(self.trainer, 'sanity_checking', False):
            self.validation_step_outputs = []
            return

        outputs = [o for o in self.validation_step_outputs if o is not None]
        self.validation_step_outputs = []
        if len(outputs) > 0:
            if (self.current_epoch + 1) % self.caption_eval_epoch == 0:
                self._shared_val_epoch_end(outputs)

    def _shared_val_epoch_end(self, outputs):
        CoT_pred_list, list_predictions, CoT_target_list, list_targets = zip(*outputs)
        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]
        CoT_preds = [i for ii in CoT_pred_list for i in ii] if CoT_pred_list[0] is not None else None
        CoT_targets = [i for ii in CoT_target_list for i in ii] if CoT_target_list[0] is not None else None

        if dist.is_available() and dist.is_initialized() and self.trainer.world_size > 1:
            all_predictions = [None for _ in range(self.trainer.world_size)]
            all_targets = [None for _ in range(self.trainer.world_size)]
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)
            predictions = [i for ii in all_predictions for i in ii]
            targets = [i for ii in all_targets for i in ii]

        if self.global_rank == 0:
            self.save_predictions_valid(predictions, targets, CoT_preds, CoT_targets)

    def save_predictions_valid(self, predictions, targets, all_CoT_preds, all_CoT_targets):
        print("****************show result*********************************")
        for j in range(min(5, len(predictions))):
            print("Generated   : %s" % predictions[-j])
            print("Ground truth: %s" % targets[-j])
            print("------------------------------------------------------")
        print("************************************************************")

        perfor = results_metrics(predictions, targets)
        performance = str(perfor)
        work_dir = self.cfg.get("work_dir", "work_dirs/eval")
        file_path = os.path.join(work_dir, "valid_logs")
        mkdir_or_exist(file_path)
        file_name = os.path.join(file_path, "valid_performance.txt")
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(performance)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        CoT_pred, predictions, drug_name1, drug_name2 = self.blip2opt.generate(
            batch,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            min_length=self.min_len,
            max_new_tokens=self.max_new_tokens
        )
        out = (CoT_pred, predictions, batch[-2], batch[-1], drug_name1, drug_name2)
        self.test_step_outputs.append(out)
        return out

    def on_test_epoch_end(self):
        outputs = [o for o in self.test_step_outputs if o is not None]
        self.test_step_outputs = []
        if len(outputs) > 0:
            self._shared_test_epoch_end(outputs)

    def _shared_test_epoch_end(self, outputs):
        CoT_pred_list, list_predictions, CoT_target_list, list_targets, drug_name1, drug_name2 = zip(*outputs)
        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]
        drug1 = [i for ii in drug_name1 for i in ii]
        drug2 = [i for ii in drug_name2 for i in ii]

        if dist.is_available() and dist.is_initialized() and self.trainer.world_size > 1:
            all_predictions = [None for _ in range(self.trainer.world_size)]
            all_targets = [None for _ in range(self.trainer.world_size)]
            all_drug1 = [None for _ in range(self.trainer.world_size)]
            all_drug2 = [None for _ in range(self.trainer.world_size)]
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)
            dist.all_gather_object(all_drug1, drug1)
            dist.all_gather_object(all_drug2, drug2)
            predictions = [i for ii in all_predictions for i in ii]
            targets = [i for ii in all_targets for i in ii]
            drug1 = [i for ii in all_drug1 for i in ii]
            drug2 = [i for ii in all_drug2 for i in ii]

        if self.global_rank == 0:
            self.save_predictions_test(predictions, targets, drug1, drug2)

    def save_predictions_test(self, predictions, targets, drug_name1, drug_name2):
        print("****************show result*********************")
        for j in range(min(5, len(predictions))):
            print("Generated   : %s" % predictions[-j])
            print("Ground truth: %s" % targets[-j])
            print("------------------------------------------------------")
        print("*************************************************")

        work_dir = self.cfg.get("work_dir", "work_dirs/eval")
        file_path = os.path.join(work_dir, "test_logs")
        mkdir_or_exist(file_path)

        json_dict = {}
        for p, t, d1, d2 in zip(predictions, targets, drug_name1, drug_name2):
            json_dict[f"{d1}&{d2}"] = {'prediction': p, 'target': t}
        with open(os.path.join(file_path, 'predictions.txt'), 'w', encoding='utf-8') as json_file:
            json.dump(json_dict, json_file, indent=4)

        perform = results_metrics(predictions, targets)
        with open(os.path.join(file_path, "test_performance.txt"), 'w', encoding='utf-8') as f:
            f.write(str(perform))

    def configure_optimizers(self):
        try:
            if hasattr(self.trainer, "reset_train_dataloader"):
                self.trainer.reset_train_dataloader()
        except Exception:
            pass

        train_loader_len = 1000
        try:
            if hasattr(self.trainer, "train_dataloader") and self.trainer.train_dataloader is not None:
                train_loader_len = len(self.trainer.train_dataloader)
            elif hasattr(self.trainer, "train_dataloaders") and self.trainer.train_dataloaders is not None:
                train_loader_len = len(self.trainer.train_dataloaders)
        except Exception:
            pass

        warmup_steps = min(train_loader_len, self.cfg.get("warmup_steps", 1000))
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.cfg.get("init_lr", 1e-4),
            weight_decay=self.cfg.get("weight_decay", 0.05)
        )
        scheduler_type = self.cfg.get("scheduler", "linear_warmup_cosine_lr")
        if scheduler_type == 'linear_warmup_cosine_lr' and LinearWarmupCosineLRScheduler is not None:
            self.scheduler = LinearWarmupCosineLRScheduler(
                optimizer, self.cfg.get("max_epochs", 30), self.cfg.get("min_lr", 1e-5),
                self.cfg.get("init_lr", 1e-4), warmup_steps, self.cfg.get("warmup_lr", 1e-6)
            )
        elif scheduler_type == 'linear_warmup_step_lr' and LinearWarmupStepLRScheduler is not None:
            self.scheduler = LinearWarmupStepLRScheduler(
                optimizer, self.cfg.get("max_epochs", 30), self.cfg.get("min_lr", 1e-5),
                self.cfg.get("init_lr", 1e-4), self.cfg.get("lr_decay_rate", 0.9),
                self.cfg.get("warmup_lr", 1e-6), warmup_steps
            )
        else:
            self.scheduler = None
        return optimizer