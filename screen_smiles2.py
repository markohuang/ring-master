import torch
import multiprocess as mp
import pandas as pd
from tqdm import tqdm
from rdkit import RDLogger
from rdkit import Chem
from functools import partial
from experiments.zinc250k import cfg, setup_experiment
from ringmaster.lumberjack import MolParser
from ringmaster.timberwright import decode
from ringmaster.chem_utils import get_mol
from ringmaster.nn_utils import NetworkPrediction
from torch.nn import functional as F

lg = RDLogger.logger() 
lg.setLevel(RDLogger.CRITICAL)


def check_smiles(smiles):
    try:
        smiles = smiles.strip()
        if get_mol(smiles) is None:
            print('invalid smiles:', smiles, flush=True)
            return None
        mol_tree, _, traversal_order = MolParser(smiles).tensors
        # edge_index = (mol_tree.fmess[:,:2]).T
        if mol_tree.fnode.shape[0] > 50: 
            print('too many nodes:', smiles, flush=True)
            return None
        max_cls_size = cfg['setupparams']['max_cand_size']
        tfnode, _, _, assm_cands, _ = mol_tree
        seq_len = tfnode.shape[0]
        cls, icls = tfnode.T
        cls_pred = torch.nn.functional.one_hot(cls.long()).float() * 1000
        icls_pred = torch.nn.functional.one_hot(icls.long(), num_classes=NetworkPrediction.vocab.size()[1]).float() * 1000
        tree_vec = torch.rand(seq_len, 10)
        class fake_cand_nn:
            __index = -1
            __assm_cands = assm_cands[1:].long()
            @staticmethod
            def fetch():
                fake_cand_nn.__index += 1
                return fake_cand_nn.__assm_cands[fake_cand_nn.__index]
            def __call__(self, _):
                ans = self.fetch()
                if ans == -100:
                    return torch.zeros(max_cls_size)
                else:
                    return torch.nn.functional.one_hot(ans, max_cls_size).float()*1000
        cand_nn = fake_cand_nn()

        networkprediction = NetworkPrediction(
            tree_vec=tree_vec,
            cls_pred=cls_pred,
            icls_pred=icls_pred,
            traversal_predictions=traversal_order,
            cand_nn=cand_nn,
        )
        predicted_smiles = decode(networkprediction).smiles
        if Chem.CanonSmiles(smiles) == predicted_smiles:
            return predicted_smiles
    except:
        print('error', smiles, flush=True)


def main(smiles_list):
    data = list(set(smiles_list))
    batch_size = len(data) // mp.cpu_count()
    batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
    for b in batches:
        with mp.Pool(mp.cpu_count()) as pool:
            valid_smiles = list(pool.map( check_smiles, b, chunksize=100 ))
        with open('zinc_smiles.txt', 'a') as f:
            f.write('\n'.join(list(s for s in valid_smiles if (s is not None)))+'\n')




if __name__ == '__main__':
    vocab = setup_experiment(cfg)
    smiles_list = pd.read_csv('zinc250k.csv')['smiles'].to_list()
    main(smiles_list)