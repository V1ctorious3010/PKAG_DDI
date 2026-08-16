"""
Optimized Data Builder for PKAG-DDI on DrugBank dataset.
Generates compact, high-speed dataset caches (data_cache.pt and data_cache.pkl)
to prevent filesystem inode/disk exhaustion on Kaggle and cloud environments.
"""

import os
import json
import pickle
import argparse
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm
from ogb.utils import smiles2graph


def mol_to_graph_data_obj_simple(smiles: str) -> Data:
    try:
        g_dict = smiles2graph(smiles)
        x = torch.tensor(g_dict["node_feat"], dtype=torch.long)
        edge_index = torch.tensor(g_dict["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(g_dict["edge_feat"], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except Exception:
        x = torch.zeros((1, 9), dtype=torch.long)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_input_data(
    dataset_name: str = "DrugBank",
    split_mode: str = "random",
    mode: str = "train",
    fold: int = 0,
    max_samples: int = None,
    data_base_dir: str = "./data",
):
    """Build input cache for a given split."""
    # Determine input text list file
    if split_mode == "random":
        input_file = f"{data_base_dir}/{dataset_name}/random_split/{mode}_seed{fold}.txt"
        output_dir = f"{data_base_dir}/{dataset_name}_inputdata/{mode}_{split_mode}_split{fold}"
    elif split_mode == "cold":
        if mode == "train":
            input_file = f"{data_base_dir}/{dataset_name}/cold_split/fold{fold}/train.txt"
        elif mode == "val":
            input_file = f"{data_base_dir}/{dataset_name}/cold_split/fold{fold}/su.txt"
        elif mode == "test":
            input_file = f"{data_base_dir}/{dataset_name}/cold_split/fold{fold}/uu.txt"
        output_dir = f"{data_base_dir}/{dataset_name}_inputdata/{mode}_{split_mode}_split{fold}"
    elif split_mode == "scaffold":
        if mode == "train":
            input_file = f"{data_base_dir}/{dataset_name}/scaffold_split/train.txt"
            output_dir = f"{data_base_dir}/{dataset_name}_inputdata/train_{split_mode}_split{fold}"
        elif mode == "val":
            val_file = f"{data_base_dir}/{dataset_name}/scaffold_split/val.txt"
            input_file = val_file if os.path.exists(val_file) else f"{data_base_dir}/{dataset_name}/scaffold_split/test.txt"
            output_dir = f"{data_base_dir}/{dataset_name}_inputdata/val_{split_mode}_split{fold}"
        elif mode == "test":
            test_file = f"{data_base_dir}/{dataset_name}/scaffold_split/test.txt"
            input_file = test_file if os.path.exists(test_file) else f"{data_base_dir}/{dataset_name}/scaffold_split/val.txt"
            output_dir = f"{data_base_dir}/{dataset_name}_inputdata/test_{split_mode}_split{fold}"

    if not os.path.exists(input_file):
        print(f"File {input_file} not found. Skipping.")
        return

    # Load ALL_DICT
    all_dict_path = f"{data_base_dir}/{dataset_name}/all_ddi_addSMIELS.pkl"
    with open(all_dict_path, "rb") as f:
        all_dict = pickle.load(f)

    # Read items
    with open(input_file, "r", encoding="utf-8") as f:
        ddi_keys = [line.strip() for line in f if line.strip()]

    if max_samples is not None:
        ddi_keys = ddi_keys[:max_samples]

    print(f"Building {output_dir} ({len(ddi_keys)} items)...")

    # Cache graph objects for unique SMILES
    graph_cache = {}
    items = []

    for key in tqdm(ddi_keys, desc=f"{split_mode}_{mode}{fold}"):
        item = all_dict[key]
        d1, d2 = item["drug1_name"], item["drug2_name"]
        s1, s2 = item["SMILES1"], item["SMILES2"]
        mech_des = item["mechanism_des"]
        f1 = item["function1"][0] if isinstance(item["function1"], (list, tuple)) else item["function1"]
        f2 = item["function2"][0] if isinstance(item["function2"], (list, tuple)) else item["function2"]

        if s1 not in graph_cache:
            graph_cache[s1] = mol_to_graph_data_obj_simple(s1)
        if s2 not in graph_cache:
            graph_cache[s2] = mol_to_graph_data_obj_simple(s2)

        items.append({
            "drug1_name": d1,
            "drug2_name": d2,
            "smiles1": s1,
            "smiles2": s2,
            "mechanism_des": mech_des,
            "function1": f1,
            "function2": f2,
        })

    os.makedirs(output_dir, exist_ok=True)
    cache_data = {
        "items": items,
        "graphs": graph_cache,
    }
    torch.save(cache_data, os.path.join(output_dir, "data_cache.pt"))
    with open(os.path.join(output_dir, "data_cache.pkl"), "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Finished building {output_dir} (Saved {len(items)} items & {len(graph_cache)} graphs).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build input data for PKAG-DDI")
    parser.add_argument("--dataset", default="DrugBank", help="Dataset name")
    parser.add_argument("--split_mode", default="random", help="random, cold, scaffold")
    parser.add_argument("--mode", default="train", help="train, val, test")
    parser.add_argument("--fold", type=int, default=0, help="0, 1, 2")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples for testing")
    parser.add_argument("--data_dir", default="./data", help="Base data directory")
    args = parser.parse_args()

    build_input_data(
        dataset_name=args.dataset,
        split_mode=args.split_mode,
        mode=args.mode,
        fold=args.fold,
        max_samples=args.max_samples,
        data_base_dir=args.data_dir,
    )
