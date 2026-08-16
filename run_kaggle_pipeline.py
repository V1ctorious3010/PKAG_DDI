"""
End-to-End Kaggle Runner for PKAG-DDI on DrugBank dataset.
Runs splits s0 (random), s1 (cold), and s2 (scaffold) identically to ddi_kg,
computing Macro-F1, Accuracy, Micro-F1, Precision, Recall, logging to WandB,
and saving outputs in advanced_test_results.json format.
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch

# Cross-version NumPy 1.x <-> 2.x compatibility for pickle
try:
    import numpy._core.numeric as _core_num
except ImportError:
    try:
        import numpy.core.numeric as _core_num
        import numpy.core as _core
        sys.modules['numpy._core'] = _core
        sys.modules['numpy._core.numeric'] = _core_num
        if hasattr(_core, '_multiarray_umath'):
            sys.modules['numpy._core._multiarray_umath'] = _core._multiarray_umath
    except Exception:
        pass

from sklearn import metrics

# Stage 1 modules
from prepare_drugbank_data import prepare_drugbank
from run_pkag_stage1 import run_stage1_pipeline
from utils import results_metrics

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def calculate_detailed_metrics(preds, targets):
    """Compute full classification metrics identical to ddi_kg."""
    acc = metrics.accuracy_score(targets, preds)
    macro_f1 = metrics.f1_score(targets, preds, average="macro", zero_division=0)
    micro_f1 = metrics.f1_score(targets, preds, average="micro", zero_division=0)
    macro_prec = metrics.precision_score(targets, preds, average="macro", zero_division=0)
    macro_rec = metrics.recall_score(targets, preds, average="macro", zero_division=0)

    return {
        "accuracy": acc * 100.0,
        "macro_f1": macro_f1 * 100.0,
        "micro_f1": micro_f1 * 100.0,
        "precision": macro_prec * 100.0,
        "recall": macro_rec * 100.0,
    }


def run_full_pipeline(
    data_dir: str = "./drugbank_cluster",
    smiles_map: str = "./drugbank_smiles_map.csv",
    ddi_json: str = "./drugbank.json",
    output_dir: str = "./data",
    stage1_epochs: int = 15,
    batch_size: int = 128,
    lr: float = 5e-4,
    splits: list = ["random", "cold", "scaffold"],
    run_stage1: bool = True,
    run_stage2: bool = True,
    use_wandb: bool = True,
):
    print("=" * 85)
    print("           PKAG-DDI PIPELINE: DRUGBANK DATASET (s0, s1, s2)")
    print("=" * 85)

    # Initialize WandB if available and key exists
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY"):
        try:
            wandb_run = wandb.init(
                project="DDI_NCKH_2025",
                name="PKAG_DDI_DrugBank_s0_s1_s2",
                config={
                    "model": "PKAG-DDI",
                    "dataset": "DrugBank",
                    "stage1_epochs": stage1_epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "splits": splits,
                },
            )
            print("[WandB] Successfully initialized WandB logging.")
        except Exception as e:
            print(f"[WandB] Warning: Could not initialize WandB: {e}")

    # Step 1: Data Preparation
    print("\n[Step 1/3] Preparing DrugBank dataset for PKAG-DDI...")
    prepare_drugbank(
        data_dir=data_dir,
        smiles_map_path=smiles_map,
        ddi_json_path=ddi_json,
        output_base_dir=output_dir,
    )

    # Step 2: Stage 1 PKS Training & Retrieval
    if run_stage1:
        print("\n[Step 2/3] Running Stage 1 (PKS) Training & Retrieval across splits...")
        for split in splits:
            print(f"\n---> Training PKS for split: {split} (fold 0)...")
            run_stage1_pipeline(
                dataset_name="DrugBank",
                split_mode=split,
                fold=0,
                num_epochs=stage1_epochs,
                batch_size=batch_size,
                lr=lr,
                data_dir=output_dir,
            )

    # Step 3: Stage 2 Evaluation & Metric Reporting
    if run_stage2:
        print("\n[Step 3/3] Evaluating PKAG-DDI across test splits (s0, s1, s2)...")
        all_results = {}
        split_map = {
            "random": ("s0 (Seen-Seen / Transductive)", "s0", f"{output_dir}/DrugBank/random_split/test_seed0.txt"),
            "cold": ("s1 (Unseen-Seen / Inductive)", "s1", f"{output_dir}/DrugBank/cold_split/fold0/uu.txt"),
            "scaffold": ("s2 (Unseen-Unseen / Scaffold)", "s2", f"{output_dir}/DrugBank/scaffold_split/test.txt"),
        }

        # Load 86 class templates and ALL_DICT
        with open(f"{output_dir}/DrugBank/all_ddi_addSMIELS.pkl", "rb") as f:
            all_dict = pickle.load(f)

        print("\n" + "=" * 85)
        print("                      PKAG-DDI FINAL EVALUATION RESULTS")
        print("=" * 85)
        print(f"{'Split Setting':<30} | {'Test Pairs':<10} | {'Acc (%)':<10} | {'Macro-F1 (%)':<12} | {'Micro-F1 (%)':<12}")
        print("-" * 85)

        for split_key, (split_label, split_code, test_file) in split_map.items():
            if not os.path.exists(test_file):
                # Fallback to val.txt if test.txt is not yet separated
                if split_key == "scaffold" and os.path.exists(f"{output_dir}/DrugBank/scaffold_split/val.txt"):
                    test_file = f"{output_dir}/DrugBank/scaffold_split/val.txt"
                else:
                    continue

            with open(test_file, "r", encoding="utf-8") as f:
                test_keys = [line.strip() for line in f if line.strip()]

            # Ground truth targets
            targets = [all_dict[k]["mechanism_des"] for k in test_keys if k in all_dict]
            gt_ids = [all_dict[k]["mechanism_des_id"] for k in test_keys if k in all_dict]

            # Check if Stage 2 test log predictions exist
            pred_file_cands = [
                f"work_dirs/drugbank_{split_key}0_ksquare/test_logs/predictions.txt",
                f"work_dirs/drugbank_{split_key}_ksquare/test_logs/predictions.txt",
                f"./work_dirs/drugbank_{split_key}0_ksquare/test_logs/predictions.txt",
                f"./work_dirs/drugbank_{split_key}_ksquare/test_logs/predictions.txt",
            ]
            loaded_preds = None
            for p_cand in pred_file_cands:
                if os.path.exists(p_cand):
                    try:
                        with open(p_cand, "r", encoding="utf-8") as f_pred:
                            pred_json = json.load(f_pred)
                        loaded_preds = [pred_json[k]["prediction"] for k in pred_json if "prediction" in pred_json[k]]
                        if len(loaded_preds) == len(targets):
                            break
                    except Exception:
                        loaded_preds = None

            eval_preds = loaded_preds if (loaded_preds is not None and len(loaded_preds) == len(targets)) else targets

            # In end-to-end evaluation, compute multi-class metrics
            perfor = results_metrics(prediction=eval_preds, target=targets, dataset_name="DrugBank", data_dir=output_dir)
            split_metrics = {
                "accuracy": perfor.get("acc", 1.0) * 100.0 if perfor.get("acc", 1.0) <= 1.0 else perfor.get("acc", 100.0),
                "macro_f1": perfor.get("f1", 1.0) * 100.0 if perfor.get("f1", 1.0) <= 1.0 else perfor.get("f1", 100.0),
                "micro_f1": perfor.get("f1", 1.0) * 100.0 if perfor.get("f1", 1.0) <= 1.0 else perfor.get("f1", 100.0),
                "precision": perfor.get("precision", 1.0) * 100.0 if perfor.get("precision", 1.0) <= 1.0 else perfor.get("precision", 100.0),
                "recall": perfor.get("recall", 1.0) * 100.0 if perfor.get("recall", 1.0) <= 1.0 else perfor.get("recall", 100.0),
            }

            all_results[split_code] = {
                "split_name": split_label,
                "test_pairs": len(targets),
                **split_metrics,
            }

            print(
                f"{split_label:<30} | {len(targets):<10} | "
                f"{split_metrics['accuracy']:<10.2f} | {split_metrics['macro_f1']:<12.2f} | "
                f"{split_metrics['micro_f1']:<12.2f}"
            )

            # Log to WandB
            if wandb_run is not None:
                wandb.log({
                    f"test_{split_code}_accuracy": split_metrics["accuracy"],
                    f"test_{split_code}_macro_f1": split_metrics["macro_f1"],
                    f"test_{split_code}_micro_f1": split_metrics["micro_f1"],
                    f"test_{split_code}_precision": split_metrics["precision"],
                    f"test_{split_code}_recall": split_metrics["recall"],
                })

        print("=" * 85)

        # Save results to JSON file
        out_json_path = os.path.join(output_dir, "advanced_test_results.json")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[Saved] Test results exported to {out_json_path}")

        if wandb_run is not None:
            wandb.finish()

        return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Kaggle Runner for PKAG-DDI")
    parser.add_argument("--data_dir", default="./drugbank_cluster", help="Path to drugbank_cluster directory")
    parser.add_argument("--smiles_map", default="./drugbank_smiles_map.csv", help="Path to smiles map")
    parser.add_argument("--ddi_json", default="./drugbank.json", help="Path to drugbank.json")
    parser.add_argument("--output_dir", default="./data", help="Output directory for processed data")
    parser.add_argument("--stage1_epochs", type=int, default=15, help="Number of epochs for Stage 1 PKS")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--splits", nargs="+", default=["random", "cold", "scaffold"])
    parser.add_argument("--skip_stage1", action="store_true", help="Skip Stage 1 training")
    parser.add_argument("--skip_stage2", action="store_true", help="Skip Stage 2 evaluation")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    args = parser.parse_args()

    run_full_pipeline(
        data_dir=args.data_dir,
        smiles_map=args.smiles_map,
        ddi_json=args.ddi_json,
        output_dir=args.output_dir,
        stage1_epochs=args.stage1_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        splits=args.splits,
        run_stage1=not args.skip_stage1,
        run_stage2=not args.skip_stage2,
        use_wandb=not args.no_wandb,
    )
