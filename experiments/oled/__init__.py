import os
import multiprocessing as mp
from functools import partial
from itertools import chain
from .vocab import PairVocab, COMMON_ATOM_VOCAB, BOND_LIST
from tqdm import tqdm
from ringmaster import RingMaster
from ringmaster.timber import MotifNode
from ringmaster.nn_utils import NetworkPrediction
from ringmaster.timberwright import MolParser
from argparse import Namespace


params = Namespace(**dict(
    vocab_path='vocab.txt',
    smiles_path='smiles.txt',
    max_cand_size=20,
    cands_hidden_size=64,
))

def process_vocab(batch, data):
    vocab = set()
    for smiles in data[batch]:
        try:
            smiles = smiles.strip()
            mol_tree = MolParser(smiles).tree
            for _, attr in mol_tree.nodes(data=True):
                csmiles = attr['smiles']
                vocab.add(attr['label'])
                for _, s in attr['inter_label']:
                    vocab.add((csmiles, s))
        except Exception as e:
            print(f"Error processing line {smiles}: {e}")
            continue
    return vocab

def setup_vocab(params):
    vocab_path = params.vocab_path
    if os.path.exists(vocab_path):
        return
    with open(params.smiles_path, 'r') as f:
        smiles_list = f.readlines()
    data = list(set(smiles_list))
    ncpu = mp.cpu_count()
    batch_size = len(data) // ncpu + 1
    batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
    process_with_batch = partial(process_vocab, data=batches)
    with mp.Pool(mp.cpu_count()) as pool:
        set_vocab = set().union(
            chain.from_iterable(
                tqdm(pool.imap( process_with_batch, range(len(batches)) ), total=len(batches))
            )
        )
    lst_vocab = sorted(list(set_vocab))
    with open(vocab_path, 'w') as f:
        f.write('<pad> <pad>\n')
        f.write('\n'.join(list(' '.join(x) for x in lst_vocab)))
    
def setup_experiment(params):
    MolParser.bond_list = BOND_LIST
    setup_vocab(params)
    with open(params.vocab_path, 'r') as f:
        MOTIF_VOCAB = PairVocab([x.strip("\r\n ").split() for x in f])
    RingMaster.vocab = MOTIF_VOCAB
    RingMaster.atom_vocab = COMMON_ATOM_VOCAB
    MotifNode.vocab = MOTIF_VOCAB
    NetworkPrediction.vocab = MOTIF_VOCAB
    NetworkPrediction.max_cand_size = params.max_cand_size
    NetworkPrediction.cands_hidden_size = params.cands_hidden_size


__all__ = ['params', 'setup_experiment']