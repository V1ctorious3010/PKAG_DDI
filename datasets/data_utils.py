import os
import pickle
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

def remove_element(lst, target):
    aaa = [x for x in lst if x != target]
    new = ""
    for id, i in enumerate(aaa):
        if id == 0:
            new = new + i
        else:
            new = new + " " + i
    return new

if __name__ == '__main__':
    all_dict_path = "./data/MecDDI/all_ddi_addSMIELS.pkl"
    if os.path.exists(all_dict_path):
        with open(all_dict_path, 'rb') as f:
            ALL_DICT = pickle.load(f)

        # get the biological function pool
        uni_function = []
        drug2functio = defaultdict(list)
        drug2smiles = {}
        for k, v in ALL_DICT.items():
            drug1, drug2 = v["drug1_name"], v["drug2_name"]
            smiles1, smiles2 = v["SMILES1"], v["SMILES2"]
            drug2smiles[drug1] = smiles1
            drug2smiles[drug2] = smiles2
            count = 0
            for s in v.keys():
                if "Mechanism" in s:
                    count = count + 1
            for i in range(1, count + 1):
                Mech = v[f"Mechanism_{i}"]
                function_name1, function_name2 = Mech["drug1"].strip().split("  "), Mech["drug2"].strip().split("  ")
                function_name1_new = remove_element(function_name1, '')
                function_name2_new = remove_element(function_name2, '')
                if function_name1_new not in uni_function:
                    uni_function.append(function_name1_new)
                if function_name2_new not in uni_function:
                    uni_function.append(function_name2_new)
                drug2functio[drug1].append(function_name1_new)
                drug2functio[drug2].append(function_name1_new)
        drug2function = defaultdict(list)
        for k, v in drug2functio.items():
            new_factos = []
            for i in v:
                if i not in new_factos:
                    new_factos.append(i)
            drug2function[k] = new_factos

        biobert_path = "all_checkpoints/biobert-base-cased-v1.2"
        if not os.path.exists(biobert_path):
            biobert_path = "dmis-lab/biobert-base-cased-v1.2"
        biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_path)
        model = AutoModel.from_pretrained(biobert_path)

        def get_init_function_emb(i):
            factor_inputs = biobert_tokenizer(i, return_tensors="pt", padding='max_length', max_length=57)
            factor_outputs = model(**factor_inputs)
            all_functions_emb = factor_outputs[0].detach().cpu().numpy()
            return all_functions_emb

        function2embedding = {}
        for i in uni_function:
            emb = get_init_function_emb(i)
            function2embedding[i] = emb