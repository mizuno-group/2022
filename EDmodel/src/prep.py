# 220609

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import Descriptors
import re
from tqdm import tqdm

def is_organic(smiles):
    organic_atom_set = set([1,5,6,7,8,9,15,16,17,35,53])
    sm = []
    for s in smiles:
        try:
            m = Chem.MolFromSmiles(s)
            atom_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
            judge = set(atom_list) <= organic_atom_set
            sm.append(judge)
        except:
            sm.append(False)
    return sm

def heavy_atom(smiles,lower=3,upper=50):
    sm = []
    for s in smiles:
        try:
            m = Chem.MolFromSmiles(s)
            heavy = Descriptors.HeavyAtomCount(m)
            if (heavy >= lower) & (heavy <= upper):
                sm.append(True)
            else:
                sm.append(False)
        except:
            sm.append(False)
    return sm

def salt_remove(smiles):
    remover = SaltRemover()
    sm = []
    for s in smiles:
        s2 = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(s),dontRemoveEverything=True),isomericSmiles=True)
        if "." in s2:
            mol_frags = Chem.GetMolFrags(Chem.MolFromSmiles(s2),asMols=True)
            largest = None
            largest_size = 0
            for mol in mol_frags:
                size = mol.GetNumAtoms()
                if size > largest_size:
                    largest = mol
                    largest_size = size
            s2 = Chem.MolToSmiles(largest)
        sm.append(s2)
    return sm

def randomize_smiles(smiles):
    sm = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        sm.append(Chem.MolToSmiles(nm,canonical=False))
    return sm

def smiles_vocab_dict():
    regex_sml = CHARSET["smiles"]
    a = r"Cl|Br|#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps\/\\"
    temp = re.findall(regex_sml,a)
    vocab_smi = {}
    for i,v in enumerate(temp):
        vocab_smi[v] = i+3
    vocab_smi.update({"<pad>":0,"<s>":1,"</s>":2}) 
    return vocab_smi

CHARSET = {"smiles":r"Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps\/\\]"}

def char_to_idx(seq_list,vocab,charset="smiles"):
    regex_sml = CHARSET[charset]
    length = 0
    idx_list = []
    for v in seq_list:
        char = re.findall(regex_sml,v)
        seq = [vocab[w] for w in char]
        seq = np.array(seq).astype(np.int32)
        seq = np.concatenate([np.array([vocab["<s>"]]),seq,np.array([vocab["</s>"]])]).astype(np.int32)
        length = len(seq) if len(seq) > length else length
        idx_list.append(seq)
    return idx_list, length

def padding(seq_list,maxlen):
    """
    zero padding
    """
    pad_seq = []
    for v in seq_list:
        pad_len = maxlen - len(v)
        temp = np.concatenate([v,np.zeros(pad_len)])
        pad_seq.append(temp)
    return np.stack(pad_seq) 