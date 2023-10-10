import os
import toml
import multiprocessing as mp
from pathlib import Path
from functools import partial
from itertools import chain
from ringmaster.vocab import PairVocab, COMMON_ATOM_VOCAB, BOND_LIST
from tqdm import tqdm
from ringmaster.timber import MotifNode
from ringmaster.nn_utils import NetworkPrediction
from ringmaster.lumberjack import MolParser
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datetime import datetime
from datasets import Dataset, DatasetDict
from argparse import Namespace

config_path = Path(__file__).parent.parent.parent.absolute() / 'configs' / 'oled_llm_config.toml'
cfg = toml.load(str(config_path))

run_name = datetime.now().strftime(f"{cfg['setupparams']['name']}_run_%m%d_%H_%M")
cfg['setupparams'] |= dict(
    run_name=run_name,
    atom_vocab=COMMON_ATOM_VOCAB,
    bond_list=BOND_LIST,
)


def process_vocab(batch, data):
    vocab = set()
    bad_smiles = set()
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
            bad_smiles.add(smiles)
            # print(f"Error processing line {smiles}: {e}")
            continue
    return vocab, bad_smiles


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
        vocab_patches, bad_smile_patches = zip(*tqdm(
            pool.imap( process_with_batch, range(len(batches)) ), total=len(batches)
        ))
    set_vocab = set().union(
        chain.from_iterable(
            vocab_patches
        )
    )
    bad_smiles = set().union(
        chain.from_iterable(
            bad_smile_patches
        )
    )
    lst_vocab = sorted(list(set_vocab))
    with open(vocab_path, 'w') as f:
        f.write('<pad> <pad>\n')
        f.write('\n'.join(list(' '.join(x) for x in lst_vocab)))
    
    with open('/home/marko/Projects/data/oled/bad_smiles.txt', 'w') as f:
        f.write('\n'.join(list(bad_smiles)))
        


def setup_experiment(cfg):
    setupparams = cfg['setupparams']
    trainingparams = cfg['trainingparams']
    MolParser.bond_list = setupparams['bond_list']
    # setup_vocab(setupparams)
    with open(setupparams['vocab_path'], 'r') as f:
        motif_vocab = PairVocab([x.strip("\r\n ").split() for x in f])
    MolParser.vocab = motif_vocab
    MolParser.atom_vocab = setupparams['atom_vocab']
    MotifNode.vocab = motif_vocab
    NetworkPrediction.vocab = motif_vocab
    NetworkPrediction.max_cand_size = setupparams['max_cand_size']
    NetworkPrediction.cands_hidden_size = trainingparams['cands_hidden_size']
    # dataset = Dataset.from_text(cfg['setupparams']['smiles_path']).train_test_split(test_size=0.1)
    # dataset.save_to_disk('/home/marko/Projects/ring-master/dataset')
    dataset = DatasetDict.load_from_disk('/home/marko/Projects/ring-master/dataset')
    # trainset = dataset['train'].select(range(14000*128, 14000*128*2))
    # valset = dataset['test'].select(range(1400*128))
    trainset = dataset['train']
    valset = dataset['test']
    tokenizer = AutoTokenizer.from_pretrained('/home/marko/Projects/ring-master/tokenizer')
    tokenizer.model_max_length = trainingparams['smiles_max_length']
    return motif_vocab, tokenizer, trainset, valset

from ringmaster.custom_callbacks import GenerateOnValidationCallback
callbacks = [
    # GenerateOnValidationCallback(),
    EarlyStopping(monitor="val/loss", mode="min", patience=20),
    LearningRateMonitor(logging_interval='step'),
    ModelCheckpoint(
        dirpath=f'checkpoints/{run_name}',
        monitor='val/loss',
        # every_n_train_steps=5000,
        every_n_epochs=1,
        save_top_k=1,
    ),
]

# from datasets import Dataset
# dataset = Dataset.from_text(cfg['setupparams']['smiles_path']).train_test_split(test_size=0.1)
# trainset = dataset['train']
# valset = dataset['test']

__all__ = ['cfg', 'setup_experiment', 'callbacks']