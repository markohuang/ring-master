import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
from torch_geometric.data import HeteroData
from torch_geometric.data.collate import collate
from multiprocessing import cpu_count

import pytorch_lightning as pl
# from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger


from importlib import import_module
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--experiment', type=str, default='mof')
# args = parser.parse_args()
# module = import_module('experiments.' + args.experiment)
module = import_module('experiments.' + 'zinc250k')
cfg, setup_experiment, callbacks = module.cfg, module.setup_experiment, module.callbacks

from ringmaster.lumberjack import MolParser
from ringmaster.model import DiffusionTransformer
vocab, trainset, valset = setup_experiment(cfg)
pl.seed_everything(cfg['setupparams'].get('seed', 42))

def pad_tensor(tensor, pad_length):
    return F.pad(tensor, (0,pad_length-tensor.shape[1]), value=-1)

def my_collator(batch, max_atom_neighbor, max_motif_neighbor, key):
    graph_list = []
    smiles_list = [ b[key] for b in batch ]
    for smiles in smiles_list:
        parser = MolParser(smiles)
        mol_tree, mol_graph, traversal_order = parser.tensors

        hierGraph = HeteroData()
        hierGraph["atoms"].atom_type = mol_graph.fnode
        hierGraph["atoms"].agraph = pad_tensor(mol_graph.agraph, max_atom_neighbor)
        hierGraph["atoms", "connect", "atoms"].edge_index = (mol_graph.fmess[:,:2]).T
        hierGraph["atoms"].bond_type = mol_graph.fmess[:,2]
        hierGraph["atoms"].n_child = mol_graph.fmess[:,3]
        hierGraph["atoms"].num_nodes = mol_graph.fnode.shape[0]

        cgraph = mol_tree.cgraph
        # Find indices of non-negative elements
        rows, cols = torch.where(cgraph >= 0)
        values = cgraph[rows, cols]
        motif_atom_edge_index = torch.vstack((rows, values))
        hierGraph["motif"].cls = mol_tree.fnode[:,0]
        hierGraph["motif"].icls = mol_tree.fnode[:,1]
        hierGraph["motif"].agraph = pad_tensor(mol_tree.agraph, max_motif_neighbor)
        hierGraph["motif", "connect", "motif"].edge_index = (mol_tree.fmess[:,:2]).T if len(mol_tree.fmess) > 0 else mol_tree.fmess
        hierGraph["motif"].edge_attr = mol_tree.fmess[:,2] if len(mol_tree.fmess) > 0 else mol_tree.fmess
        hierGraph["motif"].num_nodes = mol_tree.fnode.shape[0]
        hierGraph["motif"].assm_cands = mol_tree.assm_cands
        hierGraph["motif"].order = traversal_order
        hierGraph["motif", "contains", "atoms"].edge_index = motif_atom_edge_index
        graph_list.append(hierGraph)

    graph_data, slices, _ = collate(
        graph_list[0].__class__,
        data_list=graph_list,
        increment=True,
        add_batch=True,
    )
    return graph_data, slices


collate_fn = partial(
    my_collator,
    max_atom_neighbor=cfg['setupparams']['max_atom_neighbors'],
    max_motif_neighbor=cfg['setupparams']['max_motif_neighbors'],
    key=cfg['setupparams']['dataset_key'],
)


if __name__ == "__main__":

    tloader = DataLoader(
        trainset,
        num_workers=cpu_count(),
        batch_size=cfg['trainingparams']['batch_size'],
        collate_fn=collate_fn,
        shuffle=True,
    )
    vloader = DataLoader(
        valset,
        num_workers=cpu_count(),
        batch_size=cfg['trainingparams']['batch_size'],
        collate_fn=collate_fn,
    )

    torch.set_float32_matmul_precision('medium') # as per warning
    # logger = CSVLogger(params.setup_parameters.log_dir, name=params.setup_parameters.experiment_name)
    # logger = TensorBoardLogger("tb_logs", name="my_model")
    logger = WandbLogger(
        project="ringmaster",
        name=cfg['setupparams']['run_name'],
        log_model=True,
    )
    if hasattr(cfg['setupparams'], 'load_model'):
        # check if path is a directory
        model_path = cfg['setup_parameters']['load_model']
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, os.listdir(model_path)[0])
        model = DiffusionTransformer.load_from_checkpoint(model_path)
    else:
        model = DiffusionTransformer(cfg, vocab)

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        **{k: cfg['trainingparams'][k] for k in (
            'accelerator',
            'max_epochs',
            'check_val_every_n_epoch',
            'devices',
            # 'val_check_interval',
        )}
    )
    trainer.fit(model, tloader, vloader)
