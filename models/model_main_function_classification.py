import os
import os.path as osp
from typing import Any, Dict
import torch

from .blip2_function_retrieval import Blip2OPT_RETRIEVAL,Blip2OPT_RETRIEVAL_marginalize

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

MODOLS = {
    "retrieval":Blip2OPT_RETRIEVAL,
    "marginalize":Blip2OPT_RETRIEVAL_marginalize,
}


# peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
class MainModel_Function_CLS(pl.LightningModule):  #
    def __init__(self, args, cfg):
        super().__init__()
        # if isinstance(args, dict):
        #     args = AttrDict(**args)

        self.args = args
        self.cfg = cfg
        self.caption_eval_epoch = cfg["caption_eval_epoch"]
        self.stage = cfg["stage"]

        # for generate text
        self.do_sample = cfg["do_sample"]
        self.num_beams = cfg["num_beams"]
        self.max_len = cfg["generate_max_len"]
        self.min_len = cfg["generate_min_len"]
        self.temperature = cfg["temperature"]
        self.top_p = cfg["top_p"]
        self.max_new_tokens = cfg["max_new_tokens"]
        self.repetition_penalty = cfg["repetition_penalty"]
        # self.use_rag = cfg["use_rag"]
        self.batch_size = cfg["batch_size"]

        # for the opt_model
        self.llm_tune = cfg["llm_tune"]  # ["lora","freeze","full"]
        self.peft_dir = cfg["peft_dir"]  # whether the peft has init checkpoint
        self.opt_model = cfg["opt_model"]  # "facebook/galactica-1.3b"
        # self.prompt = cfg["prompt"]
        if self.opt_model.find('galactica') >= 0:  
            self.blip2opt = MODOLS[self.stage](cfg) 

        elif self.opt_model.find('llama') >= 0 or self.opt_model.find('vicuna') >= 0:
            self.blip2opt = Blip2Llama(cfg["bert_name"], cfg["gin_num_layers"], cfg["gin_hidden_dim"],
                                       cfg["drop_ratio"],
                                       cfg["tune_gnn"], cfg["num_query_token"], cfg["cross_attention_freq"],
                                       self.llm_tune,
                                       self.peft_dir, self.opt_model, args=cfg)
        else:
            raise NotImplementedError()
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

        ###============== Overall Loss ===================###
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
            # max_length=self.max_len,
            min_length=self.min_len,
            max_new_tokens=self.max_new_tokens
        )
        out = (CoT_pred, predictions, batch[-2], batch[-1])
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = [o for o in self.validation_step_outputs if o is not None]
        self.validation_step_outputs = []
        if len(outputs) > 0 and self.current_epoch != 0:
            if (self.current_epoch + 1) % self.caption_eval_epoch == 0:
                self._shared_val_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        if outputs and len(outputs) > 0:
            self._shared_val_epoch_end(outputs)

    def _shared_val_epoch_end(self, outputs):
        caption_outputs = outputs
        CoT_pred_list, list_predictions, CoT_target_list, list_targets = zip(*caption_outputs)
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
            if CoT_preds is not None:
                all_CoT_preds = [None for _ in range(self.trainer.world_size)]
                all_CoT_targets = [None for _ in range(self.trainer.world_size)]
                dist.all_gather_object(all_CoT_preds, CoT_preds)
                dist.all_gather_object(all_CoT_targets, CoT_targets)
                CoT_preds = [i for ii in all_CoT_preds for i in ii]
                CoT_targets = [i for ii in all_CoT_targets for i in ii]

        if self.global_rank == 0:
            self.save_predictions_valid(predictions, targets, CoT_preds, CoT_targets)

    def save_predictions_valid(self, predictions, targets, all_CoT_preds, all_CoT_targets):
        assert len(predictions) == len(targets), f"predictions:{len(predictions)}, targets:{len(targets)}"
        print("****************show result*********************************")
        for j in range(min(5, len(predictions))):
            print("Generated   : %s" % predictions[-j])
            print("Ground truth: %s" % targets[- j])
            print("------------------------------------------------------")
        print("************************************************************")
        if all_CoT_preds is not None:
            print("****************show CoT results*********************************")
            for j in range(min(5, len(all_CoT_preds))):
                print("Generated   : %s" % all_CoT_preds[-j])
                print("Ground truth: %s" % all_CoT_targets[- j])
                print("------------------------------------------------------")
            print("************************************************************")

        perfor = results_metrics(predictions, targets)
        performance = str(perfor)
        file_path = os.path.join(self.cfg["work_dir"], "valid_logs")
        mkdir_or_exist(file_path)
        file_name = os.path.join(file_path, f"valid_performance.txt")
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(performance)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        CoT_pred, predictions, drug_name1, drug_name2 = self.blip2opt.generate(
            batch,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            # max_length=self.max_len,
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

    def test_epoch_end(self, outputs):
        if outputs and len(outputs) > 0:
            self._shared_test_epoch_end(outputs)

    def _shared_test_epoch_end(self, outputs):
        print("Entering test_epoch_end")
        caption_outputs = outputs
        CoT_pred_list, list_predictions, CoT_target_list, list_targets, drug_name1, drug_name2 = zip(*caption_outputs)
        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]
        drug1 = [i for ii in drug_name1 for i in ii]
        drug2 = [i for ii in drug_name2 for i in ii]
        CoT_preds = [i for ii in CoT_pred_list for i in ii] if CoT_pred_list[0] is not None else None
        CoT_targets = [i for ii in CoT_target_list for i in ii] if CoT_target_list[0] is not None else None

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
            if CoT_preds is not None:
                all_CoT_preds = [None for _ in range(self.trainer.world_size)]
                all_CoT_targets = [None for _ in range(self.trainer.world_size)]
                dist.all_gather_object(all_CoT_preds, CoT_preds)
                dist.all_gather_object(all_CoT_targets, CoT_targets)
                CoT_preds = [i for ii in all_CoT_preds for i in ii]
                CoT_targets = [i for ii in all_CoT_targets for i in ii]

        if self.global_rank == 0:
            if CoT_preds is None:
                self.save_predictions_test(predictions, targets, drug1, drug2)
            else:
                self.save_predictions_test(predictions, targets, CoT_preds, CoT_targets)

    def save_predictions_test(self, predictions, targets, drug_name1, drug_name2):
        assert len(predictions) == len(targets)

        print("****************show result*********************")
        for j in range(min(5, len(predictions))):
            print("Generated   : %s" % predictions[-j])
            print("Ground truth: %s" % targets[- j])
            print("------------------------------------------------------")
        print("*************************************************")

        file_path = os.path.join(self.cfg["work_dir"], "test_logs")
        mkdir_or_exist(file_path)

        json_dict = {}
        for p, t, d1, d2 in zip(predictions, targets, drug_name1, drug_name2):
            json_dict[d1+"&"+d2]={'prediction': p, 'target': t}
        with open(os.path.join(file_path, 'predictions.txt'), 'w', encoding='utf-8') as json_file:
            json.dump(json_dict, json_file, indent=4)
        
        perform = results_metrics(predictions, targets)
        performance = str(perform)

        with open(os.path.join(file_path, f"test_performance.txt"), 'w', encoding='utf-8') as file:
            file.write(performance)

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
        elif scheduler_type == 'None':
            self.scheduler = None
        else:
            self.scheduler = None
        return optimizer

    def load_from_stage1_checkpoint(self, path):
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['state_dict']
        graph_encoder_dict = get_module_state_dict(state_dict, 'blip2qformer.graph_encoder')
        qformer_dict = get_module_state_dict(state_dict, 'blip2qformer.Qformer')
        ln_graph_dict = get_module_state_dict(state_dict, 'blip2qformer.ln_graph')
        qs_weight = get_module_state_dict(state_dict, 'blip2qformer.query_tokens')
        load_ignore_unexpected(self.blip2opt.Qformer, qformer_dict)
        self.blip2opt.graph_encoder.load_state_dict(graph_encoder_dict)
        self.blip2opt.ln_graph.load_state_dict(ln_graph_dict)
        self.blip2opt.query_tokens.data.copy_(qs_weight)
        return self


def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}

    ## try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)


def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def count_subdirectories(folder_path):
    try:
        entries = os.listdir(folder_path)
        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]
        return len(subdirectories)
    except FileNotFoundError:
        print(f"file '{folder_path}' not exist")
        return -1  
    except Exception as e:
        print(f"error: {e}")
        return -2  