"""
Generates the complete, self-contained Kaggle-ready Jupyter Notebook pkag-ddi-drugbank.ipynb.
Includes all helper scripts and configs directly so that it runs out-of-the-box on any Kaggle environment!
"""

import json

# Read local script contents to embed
with open("PKAG_DDI_repo/prepare_drugbank_data.py", "r", encoding="utf-8") as f:
    prepare_script_content = f.read()

with open("PKAG_DDI_repo/run_pkag_stage1.py", "r", encoding="utf-8") as f:
    stage1_script_content = f.read()

with open("PKAG_DDI_repo/run_kaggle_pipeline.py", "r", encoding="utf-8") as f:
    pipeline_script_content = f.read()

with open("PKAG_DDI_repo/configs/final/drugbank_random0_ksquare.json", "r", encoding="utf-8") as f:
    cfg_s0_content = f.read()

with open("PKAG_DDI_repo/configs/final/drugbank_cold0_ksquare.json", "r", encoding="utf-8") as f:
    cfg_s1_content = f.read()

with open("PKAG_DDI_repo/configs/final/drugbank_scaffold_ksquare.json", "r", encoding="utf-8") as f:
    cfg_s2_content = f.read()


cells = []

# Cell 0: Secrets & Clone
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from kaggle_secrets import UserSecretsClient\n",
        "user_secrets = UserSecretsClient()\n",
        "secret_value_0 = user_secrets.get_secret(\"git_token\")\n",
        "secret_value_1 = user_secrets.get_secret(\"wandb_api_key\")\n",
        "\n",
        "# Clone PKAG-DDI repository\n",
        "!git clone https://\"$secret_value_0\"@github.com/wzy-Sarah/PKAG-DDI pkag_ddi || git clone https://github.com/wzy-Sarah/PKAG-DDI pkag_ddi\n",
        "%cd pkag_ddi\n"
    ]
})

# Cell 1: Environment / WandB
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "os.environ[\"WANDB_API_KEY\"] = secret_value_1\n",
        "os.environ[\"TOKENIZERS_PARALLELISM\"] = \"false\"\n"
    ]
})

# Cell 2: Dependencies Installation
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Install required dependencies directly (no missing requirements.txt errors)\n",
        "!pip install torch-geometric ogb rdkit transformers pytorch-lightning peft deepspeed wandb rouge_score salesforce-lavis --quiet\n"
    ]
})

# Cell 3: Local Package Setup
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os, sys\n",
        "if not os.path.exists(\"setup.py\"):\n",
        "    with open(\"setup.py\", \"w\", encoding=\"utf-8\") as f:\n",
        "        f.write(\"from setuptools import setup, find_packages\\nsetup(name='pkag_ddi', version='0.1.0', packages=find_packages())\\n\")\n",
        "!pip install -e . --quiet\n",
        "if os.getcwd() not in sys.path:\n",
        "    sys.path.append(os.getcwd())\n"
    ]
})

# Cell 4: Markdown Documentation
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# **PKAG-DDI: Pharmacokinetic-Aware Graph-Language Model for Drug-Drug Interactions**\n",
        "\n",
        "## **1. Architecture Overview**\n",
        "- **Stage 1 (Pairwise Knowledge Selector - PKS)**:\n",
        "  - Extracts graph node prototypes ($M=80$) via GIN encoder.\n",
        "  - Calculates mutual prototype cross-attention matrix $A = \\frac{(H_a W_Q)(H_b W_K)^T}{\\sqrt{d}}$.\n",
        "  - **Transpose Reuse Optimization**: Directly reuses $A^T$ for Drug B without recomputing attention, halving computational overhead.\n",
        "  - Combines graph prototypes with 1024-dim Morgan Circular Fingerprints ($fp_a, fp_b$) to compute selection probabilities $p_\\theta(c_a \\mid a, b)$ and $p_\\theta(c_b \\mid a, b)$.\n",
        "\n",
        "- **Stage 2 (PKA-LM & Marginalization Training)**:\n",
        "  - Augments drug prompts with top-$K$ biological mechanisms retrieved in Stage 1.\n",
        "  - Marginalizes loss over $K \\times K$ candidate mechanism pairs:\n",
        "    $$\\mathcal{L} = -\\sum_{i} \\log \\sum_{c_a \\in C_a, c_b \\in C_b} p_\\theta(c_a, c_b \\mid a, b) p_\\eta(y_i \\mid x(a,b), c_a, c_b, y_{<i})$$\n",
        "  - Employs LoRA fine-tuning with Galactica-1.3B / MolTC language backbone.\n",
        "\n",
        "---\n",
        "\n",
        "## **2. Evaluation Settings**\n",
        "- **`s0` (Random Split / Transductive)**: Both interacting drugs appear in training set (`Seen-Seen`).\n",
        "- **`s1` (Cold-Start Split / Inductive)**: At least one drug is unseen during training (`Unseen-Seen` / `Unseen-Unseen`).\n",
        "- **`s2` (Scaffold Split / Double Inductive)**: Drug pairs are partitioned according to Bemis-Murcko scaffold clusters.\n",
        "\n",
        "---\n",
        "\n",
        "## **3. Dataset & Modalities**\n",
        "- **Dataset**: DrugBank (86 multi-class DDIE interaction types).\n",
        "- **Modalities**:\n",
        "  - PyG 2D Molecular Graph (`/kaggle/input/sample-data/modality/drugbank/id2graph.pt`)\n",
        "  - SMILES Map (`/kaggle/input/sample-data/drugbank_smiles_map.csv`)\n",
        "  - 86 Class Interaction Descriptions (`/kaggle/input/sample-data/drugbank.json`)\n",
        "  - DrugBank Cluster Splits (`/kaggle/input/datasets/dattotien/pharmacokinetics-dataset/drugbank_cluster`)\n"
    ]
})

# Cell 5: Setup Scripts and Configurations
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Ensure all DrugBank configs and helper scripts exist\n",
        "import os\n",
        "os.makedirs(\"configs/final\", exist_ok=True)\n",
        "\n",
        "# 1. Config s0\n",
        "with open(\"configs/final/drugbank_random0_ksquare.json\", \"w\", encoding=\"utf-8\") as f:\n",
        f"    f.write({repr(cfg_s0_content)})\n",
        "\n",
        "# 2. Config s1\n",
        "with open(\"configs/final/drugbank_cold0_ksquare.json\", \"w\", encoding=\"utf-8\") as f:\n",
        f"    f.write({repr(cfg_s1_content)})\n",
        "\n",
        "# 3. Config s2\n",
        "with open(\"configs/final/drugbank_scaffold_ksquare.json\", \"w\", encoding=\"utf-8\") as f:\n",
        f"    f.write({repr(cfg_s2_content)})\n",
        "\n",
        "# 4. Prepare DrugBank script\n",
        "with open(\"prepare_drugbank_data.py\", \"w\", encoding=\"utf-8\") as f:\n",
        f"    f.write({repr(prepare_script_content)})\n",
        "\n",
        "# 5. Stage 1 Runner script\n",
        "with open(\"run_pkag_stage1.py\", \"w\", encoding=\"utf-8\") as f:\n",
        f"    f.write({repr(stage1_script_content)})\n",
        "\n",
        "# 6. Full Pipeline Runner script\n",
        "with open(\"run_kaggle_pipeline.py\", \"w\", encoding=\"utf-8\") as f:\n",
        f"    f.write({repr(pipeline_script_content)})\n",
        "\n",
        "print(\"All DrugBank configs and runner scripts are ready!\")\n"
    ]
})

# Cell 6: Data Preparation
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Data Preparation: Preprocesses DrugBank into PKAG-DDI format (s0, s1, s2 splits)\n",
        "import os\n",
        "\n",
        "data_dir = \"/kaggle/input/datasets/dattotien/pharmacokinetics-dataset/drugbank_cluster\"\n",
        "if not os.path.exists(data_dir):\n",
        "    data_dir = \"./drugbank_cluster\"\n",
        "\n",
        "smiles_map = \"/kaggle/input/sample-data/drugbank_smiles_map.csv\"\n",
        "if not os.path.exists(smiles_map):\n",
        "    smiles_map = \"./drugbank_smiles_map.csv\"\n",
        "\n",
        "ddi_json = \"/kaggle/input/sample-data/drugbank.json\"\n",
        "if not os.path.exists(ddi_json):\n",
        "    ddi_json = \"./drugbank.json\"\n",
        "\n",
        "!python prepare_drugbank_data.py --data_dir \"$data_dir\" --smiles_map \"$smiles_map\" --ddi_json \"$ddi_json\" --output_dir ./data\n"
    ]
})

# Cell 7: Stage 1 PKS Training & Retrieval
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Stage 1 (PKS): Train GNN prototype cross-attention selector & run Top-K retrieval across splits s0, s1, s2\n",
        "for split in [\"random\", \"cold\", \"scaffold\"]:\n",
        "    print(f\"\\n==================== Training PKS Stage 1: {split} ====================\")\n",
        "    !python run_pkag_stage1.py --dataset DrugBank --split_mode {split} --fold 0 --epochs 15 --batch_size 128 --data_dir ./data\n"
    ]
})

# Cell 8: Stage 2 PKA-LM Training & Evaluation
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Stage 2 (PKA-LM): Fine-tune and evaluate language model across test_s0, test_s1, test_s2\n",
        "# 1. Random Split (s0: Seen-Seen)\n",
        "print(\"\\n==================== Running Stage 2 on Split s0 (Random) ====================\")\n",
        "!python main_stage2.py --config configs/final/drugbank_random0_ksquare.json --mode ft --devices 0,1\n",
        "\n",
        "# 2. Cold-Start Split (s1: Unseen-Seen)\n",
        "print(\"\\n==================== Running Stage 2 on Split s1 (Cold-Start) ====================\")\n",
        "!python main_stage2.py --config configs/final/drugbank_cold0_ksquare.json --mode ft --devices 0,1\n",
        "\n",
        "# 3. Scaffold Split (s2: Unseen-Unseen)\n",
        "print(\"\\n==================== Running Stage 2 on Split s2 (Scaffold) ====================\")\n",
        "!python main_stage2.py --config configs/final/drugbank_scaffold_ksquare.json --mode ft --devices 0,1\n"
    ]
})

# Cell 9: Metric Aggregation & Summary
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# End-to-End Evaluation & Metric Summary across splits s0, s1, s2\n",
        "!python run_kaggle_pipeline.py --data_dir \"$data_dir\" --smiles_map \"$smiles_map\" --ddi_json \"$ddi_json\" --output_dir ./data --skip_stage1\n"
    ]
})

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": []
        },
        "gpuClass": "standard",
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("e:/PKAG DDI/pkag-ddi-drugbank.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Successfully generated self-contained e:/PKAG DDI/pkag-ddi-drugbank.ipynb with", len(cells), "cells.")
