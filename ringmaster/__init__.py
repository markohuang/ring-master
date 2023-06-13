"""RingMaster: requires vocab, atom_vocab"""

import torch
from dataclasses import dataclass
from typing import *
from ringmaster.nn_utils import NetworkPrediction, create_pad_tensor
from ringmaster.timberwright import MolParser
from ringmaster.timber import EditableMol, MotifNode

@dataclass
class MolGraphTensors:
    tfnode: torch.Tensor
    tfmess: torch.Tensor
    tagraph: torch.Tensor
    assm_cands: Optional[torch.Tensor] = None
    cgraph: Optional[torch.Tensor] = None
    def __iter__(self):
        return iter((self.tfnode, self.tfmess, self.tagraph, self.assm_cands, self.cgraph))
    

@dataclass
class MolData:
    mol_tree: MolGraphTensors
    mol_graph: MolGraphTensors
    order: torch.Tensor
    def __iter__(self):
        return iter((self.mol_tree, self.mol_graph, self.order))


def sort_candidates(
        candidates: List[int],
        used_indices: List[int],
        candidate_scores: torch.Tensor
):
    candidate_scores[used_indices] = -torch.inf
    sorted_cands = sorted( list(zip(candidates, candidate_scores.tolist())), key = lambda x:x[1], reverse=True )
    return sorted_cands


def tensorize(G, vocab, is_motif=False):
    fnode, fmess = [], []
    agraph = []
    if is_motif:
        cgraph, assm_cands = [], []
    edge_dict = {}
    fnode.extend( [None for _ in G.nodes] )
    for v, attr in sorted(G.nodes(data=True)):
        fnode[v] = vocab[attr['label']]
        agraph.append([])
        if is_motif:
            cgraph.append(list(attr['cluster']))	
            assm_cands.append(attr['assm_cands'])
    for u, v, attr in sorted(G.edges(data='label')):
        if type(attr) is tuple:
            fmess.append( (u, v, attr[0], attr[1]) )
        else:
            fmess.append( (u, v, attr, 0) )
        edge_dict[(u, v)] = eid = len(edge_dict) + 1 # edge index starts at 1
        G[u][v]['mess_idx'] = eid
        agraph[v].append(eid)
    fnode = torch.IntTensor(fnode)
    fmess = torch.IntTensor(fmess)
    agraph = create_pad_tensor(agraph)
    if is_motif:
        assm_cands = torch.IntTensor(assm_cands)
        cgraph = create_pad_tensor(cgraph)
        return MolGraphTensors(fnode, fmess, agraph, assm_cands, cgraph)
    return MolGraphTensors(fnode, fmess, agraph)


class RingMaster:
    """encoder and decoder"""
    vocab = None
    atom_vocab = None
    def __init__(self, smiles, topk):
        self.smiles = smiles
        self.parsed_mol = MolParser(smiles)
        self.topk = topk

    def encoded_mol(self):
        assert RingMaster.vocab is not None and RingMaster.atom_vocab is not None
        return MolData(
            mol_tree=tensorize(self.parsed_mol.tree, vocab=self.vocab, is_motif=True),
            mol_graph=tensorize(self.parsed_mol.graph, vocab=self.atom_vocab, is_motif=False),
            order=torch.IntTensor(self.parsed_mol.order)
        )

    def decode(self, treenet: NetworkPrediction) -> EditableMol:
        molecule = EditableMol()
        root = MotifNode(treenet.root_info)
        molecule.add_motif(root)
        stack = [root]
        curr_idx = 0
        for idx, do_traversal in enumerate(treenet.traversal_predictions):
            if not do_traversal:
                stack.pop()
                continue
            curr_idx += 1
            father_motif = stack[-1]
            add_motif_success = False
            for motif_prediction in treenet.get_topk_motifs(curr_idx, self.topk):
                if add_motif_success: break
                curr_motif = MotifNode(motif_prediction)
                candidates, used_indices = father_motif.get_candidates(curr_motif)
                candidate_scores = treenet.get_candidate_scores(curr_idx)
                sorted_candidates = sort_candidates(candidates, used_indices, candidate_scores) # [ [(43,2),(44,3)], [...] ]
                curr_atom_idx = curr_motif.attachment_info.attach_point_indices
                for fa_atom_idx, _ in sorted_candidates:
                    atom_pairs = list(zip(fa_atom_idx, curr_atom_idx)) # e.g., [(43, 2), (44, 3)]
                    # if (42,1) in atom_pairs:
                    #     pass
                    # print('trying:', idx, curr_idx, atom_pairs)
                    if molecule.add_motif(curr_motif, father_motif, atom_pairs): # updates both motifs with used information
                        stack.append(curr_motif)
                        add_motif_success = True
                        # print('success!')
                        break
            if not add_motif_success:
                stack.pop()
        return molecule
    