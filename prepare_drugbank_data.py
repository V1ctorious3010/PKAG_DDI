"""
Data Preparation Script for DrugBank dataset in PKAG-DDI format.
Prepares datasets for s0 (random), s1 (cold-start / unseen-seen), and s2 (scaffold / unseen-unseen) splits.
"""

import os
import sys
import json
import pickle
import argparse
import pandas as pd
import numpy as np
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

from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from collections import defaultdict


def get_morgan_fingerprint(smiles: str, n_bits: int = 1024):
    """Compute 1024-dim Morgan circular fingerprint from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((1, n_bits), dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((1, n_bits), dtype=np.float32)
    for bit in fp.GetOnBits():
        arr[0, bit] = 1.0
    return arr


def mol_to_graph_data_obj_simple(smiles: str) -> Data:
    """Convert SMILES to PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Fallback dummy single-atom graph
        x = torch.zeros((1, 10), dtype=torch.long)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 5), dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [
            atom.GetAtomicNum(),
            int(atom.GetChiralTag()),
            atom.GetTotalDegree(),
            atom.GetFormalCharge() + 5,
            atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            int(atom.GetMass()),
        ]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    num_bond_features = 5
    if len(mol.GetBonds()) > 0:
        edge_index_list = []
        edge_attr_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_feature = [
                int(bond.GetBondType()),
                int(bond.GetBondDir()),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
                int(bond.GetStereo()),
            ]
            edge_index_list.append([i, j])
            edge_attr_list.append(bond_feature)
            edge_index_list.append([j, i])
            edge_attr_list.append(bond_feature)
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr_list), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def prepare_drugbank(
    data_dir: str = "./drugbank_cluster",
    smiles_map_path: str = "./drugbank_smiles_map.csv",
    ddi_json_path: str = "./drugbank.json",
    output_base_dir: str = "./data",
    force_recompute: bool = False,
):
    """
    Build all PKAG-DDI data files from DrugBank data.
    """
    if not force_recompute and os.path.exists(os.path.join(output_base_dir, "DrugBank", "all_ddi_addSMIELS.pkl")):
        print(f"DrugBank data already prepared in {output_base_dir}. Skipping re-preparation.")
        return

    os.makedirs(os.path.join(output_base_dir, "DrugBank"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "retrieval"), exist_ok=True)

    # 1. Load SMILES map (from CSV and from id2graph.pt)
    drug2smiles = {}
    if os.path.exists(smiles_map_path):
        smiles_df = pd.read_csv(smiles_map_path)
        for _, row in smiles_df.iterrows():
            drug_id = str(row["Drugbank_ID"]).strip()
            smiles = str(row["SMILES"]).strip()
            drug2smiles[drug_id] = smiles

    for id2graph_cand in ["./modality/drugbank/id2graph.pt", "./mecddi/id2graph.pt"]:
        if os.path.exists(id2graph_cand):
            g_dict = torch.load(id2graph_cand, map_location="cpu", weights_only=False)
            for k, data in g_dict.items():
                if hasattr(data, "smiles") and data.smiles and k not in drug2smiles:
                    drug2smiles[k] = str(data.smiles).strip()

    # Save drug_smiles.csv
    drug_smiles_list = [{"drug_id": k, "smiles": v} for k, v in drug2smiles.items()]
    pd.DataFrame(drug_smiles_list).to_csv(
        os.path.join(output_base_dir, "DrugBank", "drug_smiles.csv"), index=False
    )
    print(f"Saved drug_smiles.csv with {len(drug2smiles)} drugs.")

    # 2. Load 86 DDIE interaction templates
    with open(ddi_json_path, "r", encoding="utf-8") as f:
        id2template = json.load(f)

    # Build id2mechanism and mechanism_des2id
    id2ddie = {int(k): v for k, v in id2template.items()}
    ddie2id = {v: int(k) for k, v in id2template.items()}
    with open(os.path.join(output_base_dir, "DrugBank", "id2mechanism.pkl"), "wb") as f:
        pickle.dump(id2ddie, f)
    with open(os.path.join(output_base_dir, "DrugBank", "mechanism_des2id.pkl"), "wb") as f:
        pickle.dump(ddie2id, f)
    print(f"Saved id2mechanism and mechanism_des2id with {len(id2ddie)} classes.")

    # 3. Compute Fingerprints
    drug2fingerprint = {}
    for drug_id, smiles in drug2smiles.items():
        fp = get_morgan_fingerprint(smiles, n_bits=1024)
        drug2fingerprint[drug_id] = fp
    with open(os.path.join(output_base_dir, "drug2fingerprint.pkl"), "wb") as f:
        pickle.dump(drug2fingerprint, f)
    print(f"Saved drug2fingerprint.pkl with {len(drug2fingerprint)} fingerprints.")

    # 4. Process split files (train, val_s0/1/2, test_s0/1/2)
    splits = {
        "train": "train.csv",
        "val_s0": "val_s0.csv",
        "val_s1": "val_s1.csv",
        "val_s2": "val_s2.csv",
        "test_s0": "test_s0.csv",
        "test_s1": "test_s1.csv",
        "test_s2": "test_s2.csv",
    }

    all_ddi_dict = {}
    split_indices = defaultdict(list)

    # Keywords for mechanism categorization
    pk_keywords = {
        "metabolism": "Metabolism",
        "metabolized": "Metabolism",
        "excretion": "Excretion",
        "absorption": "Absorption",
        "serum concentration": "Distribution",
        "adverse effects": "Adverse Reaction",
        "activities": "Pharmacodynamics",
        "risk": "Adverse Reaction",
        "decrease": "Inhibition",
        "increase": "Induction / Synergism",
    }

    def infer_functions(desc: str):
        desc_lower = desc.lower()
        f1, f2 = "General Interaction", "General Interaction"
        for kw, cat in pk_keywords.items():
            if kw in desc_lower:
                f1 = cat
                f2 = cat
                break
        return [f1], [f2]

    # Create split text files
    for split_name, fname in splits.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"Warning: {fpath} not found.")
            continue

        df = pd.read_csv(fpath)
        print(f"Processing {split_name} ({fname}): {len(df)} pairs...")

        for idx, row in df.iterrows():
            d1 = str(row["Drug1"]).strip()
            d2 = str(row["Drug2"]).strip()
            interaction_id = int(row["Interaction"])
            des = id2template.get(str(interaction_id), f"Interaction type {interaction_id}")

            ddi_key = f"{d1}_{d2}_{interaction_id}"
            if ddi_key not in all_ddi_dict:
                s1 = drug2smiles.get(d1, "")
                s2 = drug2smiles.get(d2, "")
                f1, f2 = infer_functions(des)

                all_ddi_dict[ddi_key] = {
                    "drug1_name": d1,
                    "drug2_name": d2,
                    "SMILES1": s1,
                    "SMILES2": s2,
                    "mechanism_des": des,
                    "mechanism_des_id": interaction_id,
                    "function1": f1,
                    "function2": f2,
                }

            split_indices[split_name].append(ddi_key)

    # Save all_ddi_addSMIELS.pkl
    with open(os.path.join(output_base_dir, "DrugBank", "all_ddi_addSMIELS.pkl"), "wb") as f:
        pickle.dump(all_ddi_dict, f)
    print(f"Saved all_ddi_addSMIELS.pkl with {len(all_ddi_dict)} unique DDI pairs.")

    # Save split text lists for s0, s1, s2
    split_configs = [
        ("random_split", "train_seed0.txt", split_indices["train"]),
        ("random_split", "val_seed0.txt", split_indices["val_s0"]),
        ("random_split", "test_seed0.txt", split_indices["test_s0"]),
        ("cold_split/fold0", "train.txt", split_indices["train"]),
        ("cold_split/fold0", "su.txt", split_indices["val_s1"]),
        ("cold_split/fold0", "uu.txt", split_indices["test_s1"]),
        ("scaffold_split", "train.txt", split_indices["train"]),
        ("scaffold_split", "val.txt", split_indices["val_s2"]),
        ("scaffold_split", "test.txt", split_indices["test_s2"]),
    ]

    for subdir, txt_name, keys in split_configs:
        target_dir = os.path.join(output_base_dir, "DrugBank", subdir)
        os.makedirs(target_dir, exist_ok=True)
        target_file = os.path.join(target_dir, txt_name)
        with open(target_file, "w", encoding="utf-8") as f:
            for k in keys:
                f.write(f"{k}\n")
        print(f"Saved {target_file} ({len(keys)} items).")

    # Build unique function list
    uni_functions = sorted(
        list(
            set(
                [item["function1"][0] for item in all_ddi_dict.values()]
                + [item["function2"][0] for item in all_ddi_dict.values()]
            )
        )
    )
    with open(os.path.join(output_base_dir, "retrieval", "uni_function_list.pkl"), "wb") as f:
        pickle.dump(uni_functions, f)
    print(f"Saved uni_function_list.pkl with {len(uni_functions)} functions: {uni_functions}")

    print("DrugBank preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DrugBank dataset for PKAG-DDI")
    parser.add_argument("--data_dir", default="./drugbank_cluster", help="Path to drugbank_cluster")
    parser.add_argument("--smiles_map", default="./drugbank_smiles_map.csv", help="Path to smiles map")
    parser.add_argument("--ddi_json", default="./drugbank.json", help="Path to drugbank.json")
    parser.add_argument("--output_dir", default="./data", help="Output base directory")
    args = parser.parse_args()

    prepare_drugbank(
        data_dir=args.data_dir,
        smiles_map_path=args.smiles_map,
        ddi_json_path=args.ddi_json,
        output_base_dir=args.output_dir,
    )
