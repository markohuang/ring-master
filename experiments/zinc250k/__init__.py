import os
import toml
import multiprocessing as mp
import pandas as pd
from pathlib import Path
from functools import partial
from itertools import chain
from .vocab import PairVocab, COMMON_ATOM_VOCAB, BOND_LIST
from tqdm import tqdm
from ringmaster.timber import MotifNode
from ringmaster.nn_utils import NetworkPrediction
from ringmaster.lumberjack import MolParser
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datetime import datetime
from datasets import Dataset

config_path = Path(__file__).parent.parent.parent.absolute() / 'configs' / 'zinc_config.toml'
cfg = toml.load(str(config_path))

run_name = datetime.now().strftime(f"{cfg['setupparams']['name']}_run_%m%d_%H_%M")
cfg['setupparams'] |= dict(
    run_name=run_name,
    atom_vocab=COMMON_ATOM_VOCAB,
    bond_list=BOND_LIST,
)

def process_vocab(batch, data):
    vocab = set()
    smiles_list = set()
    num_errors = 0
    for smiles in data[batch]:
        try:
            smiles = smiles.strip()
            mol_tree = MolParser(smiles).tree
            for _, attr in mol_tree.nodes(data=True):
                csmiles = attr['smiles']
                vocab.add(attr['label'])
                for _, s in attr['inter_label']:
                    vocab.add((csmiles, s))
            smiles_list.add(smiles)
        except Exception as e:
            # print(f"Error processing line {smiles}: {e}")
            num_errors += 1
            continue
    return vocab, smiles_list, num_errors


def setup_vocab(setupparams):
    vocab_path = setupparams['vocab_path']
    smiles_path = setupparams['smiles_path']
    csv_path = setupparams['csv_path']
    if os.path.exists(vocab_path):
        return
    data = pd.read_csv(csv_path).smiles.tolist()
    ncpu = mp.cpu_count()
    batch_size = len(data) // ncpu + 1
    batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
    process_with_batch = partial(process_vocab, data=batches)
    with mp.Pool(mp.cpu_count()) as pool:
        a,b,c = zip(*chain(
            tqdm(pool.imap( process_with_batch, range(len(batches)) ), total=len(batches))
        ))
    set_vocab = set().union(chain.from_iterable(a))
    lst_vocab = sorted(list(set_vocab))
    with open(vocab_path, 'w') as f:
        f.write('<pad> <pad>\n')
        f.write('\n'.join(list(' '.join(x) for x in lst_vocab)))
    set_smiles = set().union(chain.from_iterable(b))
    with open(smiles_path,'w') as f:
        f.write('\n'.join(list(set_smiles)))
    print(f"Number of errors: {sum(c)}")


def setup_experiment(cfg):
    setupparams = cfg['setupparams']
    trainingparams = cfg['trainingparams']
    MolParser.bond_list = setupparams['bond_list']
    setup_vocab(setupparams)
    with open(setupparams['vocab_path'], 'r') as f:
        motif_vocab = PairVocab([x.strip("\r\n ").split() for x in f])
    MolParser.vocab = motif_vocab
    MolParser.atom_vocab = setupparams['atom_vocab']
    MotifNode.vocab = motif_vocab
    NetworkPrediction.vocab = motif_vocab
    NetworkPrediction.max_cand_size = setupparams['max_cand_size']
    NetworkPrediction.cands_hidden_size = trainingparams['cands_hidden_size']
    dataset = Dataset.from_text(cfg['setupparams']['smiles_path']).train_test_split(test_size=0.1)
    trainset = dataset['train']
    valset = dataset['test']
    return motif_vocab, trainset, valset

from ringmaster.custom_callbacks import GenerateOnValidationCallback
callbacks = [
    GenerateOnValidationCallback(),
    EarlyStopping(monitor="val/loss", mode="min", patience=3),
    LearningRateMonitor(logging_interval='step'),
    ModelCheckpoint(
        dirpath=f'checkpoints/{run_name}',
        monitor='val/loss',
        every_n_train_steps=5000,
        save_top_k=3,
    ),
]


__all__ = ['cfg', 'setup_experiment', 'callbacks']