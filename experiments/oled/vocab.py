from rdkit import Chem
import torch

def count_inters(s):
    if s == '<pad>': return -1
    mol = Chem.MolFromSmiles(s)
    inters = [a for a in mol.GetAtoms() if a.GetAtomMapNum() > 0]
    return max(1, len(inters))

class Vocab:
    def __init__(self, smiles_list):
        self.vocab = [x for x in smiles_list] #copy
        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        
    def __getitem__(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def size(self):
        return len(self.vocab)
    
class PairVocab(object):
    def __init__(self, smiles_pairs):
        cls = list(zip(*smiles_pairs))[0]
        self.hvocab = sorted( list(set(cls)) )
        # one-hot mapping of motifs
        self.hmap = {x:i for i,x in enumerate(self.hvocab)}

        self.vocab = [tuple(x) for x in smiles_pairs] #copy
        self.inter_size = [count_inters(x[1]) for x in self.vocab]
        # one-hot mapping of attachment
        self.vmap = {x:i for i,x in enumerate(self.vocab)}

        self.mask = torch.zeros(len(self.hvocab), len(self.vocab))
        for h,s in smiles_pairs:
            hid = self.hmap[h]
            idx = self.vmap[(h,s)]
            self.mask[hid, idx] = 1000.0

        self.mask = self.mask - 1000.0
            
    def __getitem__(self, x):
        assert type(x) is tuple
        return self.hmap[x[0]], self.vmap[x]

    def get_smiles(self, idx):
        return self.hvocab[idx]

    def get_ismiles(self, idx):
        return self.vocab[idx][1] 

    def size(self):
        return len(self.hvocab), len(self.vocab)

    def get_mask(self, cls_idx):
        if type(cls_idx) is torch.Tensor:
            self.mask = self.mask.to(cls_idx.device)
        return self.mask.index_select(index=cls_idx, dim=0)

    def get_inter_size(self, icls_idx):
        return self.inter_size[icls_idx]

    
COMMON_ATOMS = [('B', 0), ('B', -1), ('Br', 0), ('Br', -1), ('Br', 2), ('C', 0), ('C', 1), ('C', -1), ('Cl', 0), ('Cl', 1), ('Cl', -1), ('Cl', 2), ('Cl', 3), ('F', 0), ('F', 1), ('F', -1), ('I', -1), ('I', 0), ('I', 1), ('I', 2), ('I', 3), ('N', 0), ('N', 1), ('N', -1), ('O', 0), ('O', 1), ('O', -1), ('P', 0), ('P', 1), ('P', -1), ('S', 0), ('S', 1), ('S', -1), ('Se', 0), ('Se', 1), ('Se', -1), ('Si', 0), ('Si', -1)]
COMMON_ATOM_VOCAB = Vocab(COMMON_ATOMS)
BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC] 

