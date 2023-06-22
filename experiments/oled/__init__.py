import os
import toml
import multiprocessing as mp
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
from argparse import Namespace

config_path = Path(__file__).parent.parent.parent.absolute() / 'configs' / 'oled_config.toml'
cfg = toml.load(str(config_path))

run_name = datetime.now().strftime("run_%m%d_%H_%M")
cfg['setupparams'] |= dict(
    run_name=run_name,
    atom_vocab=COMMON_ATOM_VOCAB,
    bond_list=BOND_LIST,
)


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


def setup_vocab(setupparams):
    vocab_path = setupparams['vocab_path']
    if os.path.exists(vocab_path):
        return
    with open(setupparams['smiles_path'], 'r') as f:
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
    return motif_vocab

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
    # GeneratePlots(),
    # ScreenValidSmiles(1000),
]
# tloader, vloader = setup_dataloader(params.hyperparameters, dataset, use_multiprocessing=True)


__all__ = ['cfg', 'setup_experiment', 'callbacks']