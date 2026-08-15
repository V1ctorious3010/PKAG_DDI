"""
Stage 1: Pairwise Knowledge Selector (PKS) Training and Retrieval for PKAG-DDI.
Supports splits:
- s0: random split (seed0)
- s1: cold-start split (fold0)
- s2: scaffold split
"""

import os
import sys
import math
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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

from torch.nn import Parameter
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim import AdamW
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import metrics
from models.mol.gin_model import GNN_STM, GNN_graphpred_STM


def get_morgan_fingerprint(smiles: str, n_bits: int = 1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((1, n_bits), dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((1, n_bits), dtype=np.float32)
    for bit in fp.GetOnBits():
        arr[0, bit] = 1.0
    return arr


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


def do_compute_metrics(preds, targets):
    pred_labels = np.argmax(preds, axis=1)
    acc = metrics.accuracy_score(targets, pred_labels)
    macro_f1 = metrics.f1_score(targets, pred_labels, average="macro", zero_division=0)
    print(f"Accuracy: {acc * 100:.2f}%, Macro-F1: {macro_f1 * 100:.2f}%")
    return {"accuracy": acc, "macro_f1": macro_f1}


class Mol_Adapter(nn.Module):
    def __init__(self, hidden_dim: int = 300, num_clusters: int = 80, residual: bool = False):
        super().__init__()
        self.Q = nn.Parameter(torch.Tensor(1, num_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.Q)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
        self.residual = residual

    def forward(self, x, mask):
        K = self.W_K(x)
        V = self.W_V(x)
        attn_mask = (~mask).float().unsqueeze(1) * (-1e9)
        Q = self.Q.tile(K.size(0), 1, 1)
        Q = self.W_Q(Q)
        A = Q @ K.transpose(-1, -2) / (Q.size(-1) ** 0.5)
        A = A + attn_mask
        A = A.softmax(dim=-2)
        out = Q + A @ V
        if self.residual:
            out = out + self.W_O(out).relu()
        else:
            out = self.W_O(out).relu()
        return out, K, V


class MLP(nn.Module):
    def __init__(self, hidden_dims, input_dim, output_dim, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Model_stage1(nn.Module):
    def __init__(self, num_classes: int, node_clusters: int = 80, dropout: float = 0.3, ratio: float = 0.9):
        super().__init__()
        self.ratio = ratio
        molecule_node_model = GNN_STM(num_layer=3, emb_dim=300, JK="last", drop_ratio=dropout, gnn_type="gin")
        self.molecule_model = GNN_graphpred_STM(
            num_layer=3, emb_dim=300, JK="last", graph_pooling="mean", num_tasks=1, molecule_node_model=molecule_node_model
        )
        self.mol_adapter = Mol_Adapter(hidden_dim=300, num_clusters=node_clusters)

        self.W_q = Parameter(torch.Tensor(300, 300))
        self.W_k = Parameter(torch.Tensor(300, 300))
        self.W_v_1 = Parameter(torch.Tensor(300, 300))
        self.W_v_2 = Parameter(torch.Tensor(300, 300))
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v_1)
        init.xavier_normal_(self.W_v_2)

        self.mlp_finger = MLP([1024, 300], 1024, 300, dropout=dropout)
        self.mlp_mol = MLP([600], 600, num_classes, dropout=dropout)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        graphs1, graphs2, finger1, finger2, fun_label1, fun_label2, _, _ = batch

        mol1, mask1 = self.molecule_model(graphs1)[0]
        mol1, _, _ = self.mol_adapter(mol1, mask1)
        mol1_q = torch.matmul(mol1, self.W_q)

        mol2, mask2 = self.molecule_model(graphs2)[0]
        mol2, _, _ = self.mol_adapter(mol2, mask2)
        mol2_k = torch.matmul(mol2, self.W_k)

        A = mol1_q @ mol2_k.transpose(-1, -2) / (mol2.size(-1) ** 0.5)

        # Drug 1 Perspective
        A1 = A.softmax(dim=-1)
        mol2_v = torch.matmul(mol2, self.W_v_1)
        out1 = mol1 + self.ratio * (A1 @ mol2_v)
        out1 = torch.mean(out1, dim=1)
        finger1 = self.mlp_finger(finger1)
        out1 = torch.cat((finger1, out1), -1)
        logits1 = self.mlp_mol(out1).squeeze()
        loss1 = self.loss(logits1, fun_label1)

        # Drug 2 Perspective (Reusing A^T)
        A2 = A.transpose(-1, -2).softmax(dim=-1)
        mol1_v = torch.matmul(mol1, self.W_v_2)
        out2 = mol2 + self.ratio * (A2 @ mol1_v)
        out2 = torch.mean(out2, dim=1)
        finger2 = self.mlp_finger(finger2)
        out2 = torch.cat((finger2, out2), -1)
        logits2 = self.mlp_mol(out2).squeeze()
        loss2 = self.loss(logits2, fun_label2)

        loss = loss1 + loss2
        return loss, logits1, logits2, fun_label1, fun_label2

    def forward_retrieval(self, batch):
        graphs1, graphs2, finger1, finger2, fun_label1, fun_label2, d1_names, d2_names = batch
        with torch.no_grad():
            _, logits1, logits2, _, _ = self.forward(batch)
            prob1 = F.softmax(logits1, dim=-1)
            prob2 = F.softmax(logits2, dim=-1)
        return d1_names, d2_names, prob1, prob2, fun_label1, fun_label2


class PKSDataset(Dataset):
    def __init__(self, data_file: str, all_dict: dict, drug2smiles: dict, drug2fp: dict, uni_functions: list, graph_cache: dict = None):
        self.all_dict = all_dict
        self.drug2smiles = drug2smiles
        self.drug2fp = drug2fp
        self.uni_functions = uni_functions
        self.graph_cache = graph_cache if graph_cache is not None else {}

        self.items = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                key = line.strip()
                if key and key in self.all_dict:
                    item = self.all_dict[key]
                    d1, d2 = item["drug1_name"], item["drug2_name"]
                    f1 = item["function1"][0]
                    f2 = item["function2"][0]
                    self.items.append((d1, d2, f1, f2))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def collate_fn(self, batch):
        d1_list, d2_list, f1_list, f2_list = zip(*batch)

        graphs1 = [self.graph_cache[d] for d in d1_list]
        graphs2 = [self.graph_cache[d] for d in d2_list]

        batch_g1 = Batch.from_data_list(graphs1)
        batch_g2 = Batch.from_data_list(graphs2)

        fp1 = torch.tensor(np.array([self.drug2fp[d].squeeze() for d in d1_list]), dtype=torch.float32)
        fp2 = torch.tensor(np.array([self.drug2fp[d].squeeze() for d in d2_list]), dtype=torch.float32)

        lbl1 = torch.tensor([self.uni_functions.index(f) for f in f1_list], dtype=torch.long)
        lbl2 = torch.tensor([self.uni_functions.index(f) for f in f2_list], dtype=torch.long)

        return batch_g1, batch_g2, fp1, fp2, lbl1, lbl2, d1_list, d2_list


def run_stage1_pipeline(
    dataset_name: str = "DrugBank",
    split_mode: str = "random",
    fold: int = 0,
    num_epochs: int = 15,
    batch_size: int = 128,
    lr: float = 5e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    data_dir: str = "./data",
):
    print("=" * 60)
    print(f"Running Stage 1 (PKS) for {dataset_name} | Split: {split_mode} | Fold: {fold}")
    print("=" * 60)

    # 1. Load mappings
    with open(f"{data_dir}/{dataset_name}/all_ddi_addSMIELS.pkl", "rb") as f:
        all_dict = pickle.load(f)
    with open(f"{data_dir}/retrieval/uni_function_list.pkl", "rb") as f:
        uni_functions = pickle.load(f)

    drug_smiles_df = pd.read_csv(f"{data_dir}/{dataset_name}/drug_smiles.csv")
    drug2smiles = {r["drug_id"]: r["smiles"] for _, r in drug_smiles_df.iterrows()}

    # Compute or load fingerprints safely
    drug2fp = None
    if os.path.exists(f"{data_dir}/drug2fingerprint.pkl"):
        try:
            with open(f"{data_dir}/drug2fingerprint.pkl", "rb") as f:
                drug2fp = pickle.load(f)
        except Exception:
            drug2fp = None

    if drug2fp is None:
        print(f"Generating Morgan fingerprints for {len(drug2smiles)} drugs...")
        drug2fp = {d: get_morgan_fingerprint(smiles) for d, smiles in drug2smiles.items()}

    # 2. Get file paths
    if split_mode == "random":
        train_f = f"{data_dir}/{dataset_name}/random_split/train_seed{fold}.txt"
        val_f = f"{data_dir}/{dataset_name}/random_split/val_seed{fold}.txt"
        test_f = f"{data_dir}/{dataset_name}/random_split/test_seed{fold}.txt"
    elif split_mode == "cold":
        train_f = f"{data_dir}/{dataset_name}/cold_split/fold{fold}/train.txt"
        val_f = f"{data_dir}/{dataset_name}/cold_split/fold{fold}/su.txt"
        test_f = f"{data_dir}/{dataset_name}/cold_split/fold{fold}/uu.txt"
    elif split_mode == "scaffold":
        train_f = f"{data_dir}/{dataset_name}/scaffold_split/train.txt"
        val_f = f"{data_dir}/{dataset_name}/scaffold_split/val.txt"
        test_f = f"{data_dir}/{dataset_name}/scaffold_split/test.txt"
    # Precompute molecular graphs once for all drugs
    print(f"Precomputing molecular graphs for {len(drug2smiles)} drugs...")
    graph_cache = {d: mol_to_graph_data_obj_simple(smiles) for d, smiles in drug2smiles.items()}

    train_ds = PKSDataset(train_f, all_dict, drug2smiles, drug2fp, uni_functions, graph_cache=graph_cache)
    val_ds = PKSDataset(val_f, all_dict, drug2smiles, drug2fp, uni_functions, graph_cache=graph_cache)
    test_ds = PKSDataset(test_f, all_dict, drug2smiles, drug2fp, uni_functions, graph_cache=graph_cache)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_ds.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=val_ds.collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=test_ds.collate_fn)

    print(f"Train instances: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # 3. Model
    model = Model_stage1(num_classes=len(uni_functions)).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=6e-5)

    os.makedirs("./all_checkpoints", exist_ok=True)
    os.makedirs(f"{data_dir}/retrieval", exist_ok=True)

    # 4. Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            batch = [x.to(device) if hasattr(x, "to") else x for x in batch]
            optimizer.zero_grad()
            loss, _, _, _, _ = model(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(train_loader):.4f}")

    ckpt_path = f"./all_checkpoints/pks_{dataset_name}_{split_mode}_fold{fold}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # 5. Retrieval step: compute Top-K biological functions for train, val, test
    print("Running Top-K Retrieval...")
    model.eval()

    def do_retrieval(loader, split_name):
        twodrug2topk = {}
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Retrieval {split_name}"):
                batch = [x.to(device) if hasattr(x, "to") else x for x in batch]
                d1_names, d2_names, prob1, prob2, _, _ = model.forward_retrieval(batch)
                scores1, indices1 = torch.topk(prob1, k=2, dim=-1)
                scores2, indices2 = torch.topk(prob2, k=2, dim=-1)

                indices1 = indices1.cpu().numpy()
                indices2 = indices2.cpu().numpy()
                scores1 = scores1.cpu().numpy()
                scores2 = scores2.cpu().numpy()

                for i in range(len(d1_names)):
                    key = f"{d1_names[i]}&{d2_names[i]}"
                    twodrug2topk[key] = [indices1[i], indices2[i], scores1[i], scores2[i]]

        out_file = f"{data_dir}/retrieval/{split_mode}_{split_name}{fold}_retrieval.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(twodrug2topk, f)
        print(f"Saved retrieval file: {out_file} ({len(twodrug2topk)} pairs)")

    do_retrieval(train_loader, "train")
    do_retrieval(val_loader, "val")
    do_retrieval(test_loader, "test")

    print(f"Stage 1 completed successfully for {split_mode}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 PKS runner for PKAG-DDI")
    parser.add_argument("--dataset", default="DrugBank")
    parser.add_argument("--split_mode", default="random", help="random (s0), cold (s1), scaffold (s2)")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--data_dir", default="./data", help="Data directory")
    args = parser.parse_args()

    run_stage1_pipeline(
        dataset_name=args.dataset,
        split_mode=args.split_mode,
        fold=args.fold,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir,
    )
