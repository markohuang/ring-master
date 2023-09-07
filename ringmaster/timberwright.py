import torch
import math
from rdkit import Chem
from ringmaster.chem_utils import ParseAtomInfo, sanitize, clear_global_atom_info, get_all_candidates
from ringmaster.nn_utils import NetworkPrediction
from ringmaster.timber import MotifNode
from itertools import chain
from typing import *


def get_candidates(father_motif, child_motif):
    father = father_motif.as_father
    father_global_num = ParseAtomInfo(father.mol).global_idx
    if father.mol.GetNumAtoms() == 1:
        candidates = [[father_global_num(0)]]
        return candidates, []
    child = child_motif.attachment_info
    all_candidates = get_all_candidates(
        father.order,
        child.num_attach_points,
        child.is_symmetric
    )
    
    def not_the_same_atom(x):
        child_atom = child.attach_point_atoms[0]
        father_atom = father.mol.GetAtomWithIdx(x)
        not_same_symbol = child_atom.GetSymbol() != father_atom.GetSymbol()
        not_same_charge = child_atom.GetFormalCharge() != father_atom.GetFormalCharge()
        return not_same_symbol or not_same_charge

    def not_the_same_atoms(atoms):
        # this does not check if they are in the correct order
        curr_set = set( (atom.GetSymbol(), atom.GetFormalCharge()) for atom in child.attach_point_atoms )
        fa_set = set( (father.mol.GetAtomWithIdx(x).GetSymbol(), father.mol.GetAtomWithIdx(x).GetFormalCharge()) for x in atoms )
        return curr_set != fa_set
    # 1. get candidates
    # 2. get indices of candidates that are used
    # 3. get indices of candidates that do not match with attachment points (i.e., different atom / formal charge)
    # bond_match(mol, c[0], c[-1], emol, attach_points[0], attach_points[-1])
    used_indices = []
    if child.num_attach_points == 1:
        assert len(child.attach_point_atoms) == 1
        for cand_idx, x in enumerate(chain.from_iterable(all_candidates)):
            if (father_global_num(x) in father.used) or not_the_same_atom(x):
                used_indices.append(cand_idx)
    else:
        for cand_idx, cand in enumerate(all_candidates):
            # cand can be 2, 3, 4 ( list(pairwise(cand)) )
            bonds = [cand[i : i + 2] for i in range(len(cand)-1)]
            if not_the_same_atoms(cand) or any(tuple(father_global_num(b) for b in bond) in father.used for bond in bonds):
                used_indices.append(cand_idx)
    candidates = [ list(map(father_global_num, x)) for x in all_candidates ]
    return candidates, used_indices


def sort_candidates(
        candidates: List[int],
        used_indices: List[int],
        candidate_scores: torch.Tensor
):
    candidate_scores[used_indices] = -torch.inf
    # sorted_cands = sorted( list(zip(candidates, candidate_scores.tolist())), key = lambda x:x[1], reverse=True )
    sorted_cands = sorted([(x,y) for x,y in zip(candidates, candidate_scores.tolist()) if not math.isinf(y)], key = lambda x:x[1], reverse=True)
    return sorted_cands

class EditableMol:
    """the editable molecule built by decoding network predictions"""
    def __init__(self):
        self.molecule = Chem.RWMol()

    def add_motif(self, curr_motif, father_motif=False, atom_pairs=[]):
        """
        if succeeds, updates 
            the curr_motif and father_motif with new used atoms
            the curr_motif with its global atom indices
        if fails, undos the trial and returns False
        """
        if not atom_pairs and father_motif:
            return False
        global_index_of_intersecting_atoms = [x[0] for x in atom_pairs]
        atom_map = {y: x for x, y in atom_pairs}
        molecule = Chem.RWMol(self.molecule)
        new_atoms, used_bonds = [], []
        curr_motif_mol, curr_motif_atom_order = curr_motif.mol
        for atom_idx in curr_motif_atom_order:
            atom = curr_motif_mol.GetAtomWithIdx(atom_idx)
            if atom_idx in atom_map: # shared with fa cls
                idx = atom_map[atom_idx]
                new_atoms.append(idx)
            else: # only in current cls
                new_atom = Chem.Atom(atom.GetSymbol())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                idx = molecule.AddAtom(new_atom) # if not in atom map, add new node_idx = len(num nodes) + 1
                atom_map[atom_idx] = idx
                new_atoms.append(idx)
                # if atommapnum = 1, atom is in intersection with fa_cls

        for bond in curr_motif_mol.GetBonds():
            a1 = atom_map[bond.GetBeginAtom().GetIdx()]
            a2 = atom_map[bond.GetEndAtom().GetIdx()]
            if a1 == a2:
                return False
            bond_type = bond.GetBondType()
            existing_bond = molecule.GetBondBetweenAtoms(a1, a2)
            if existing_bond is None:
                molecule.AddBond(a1, a2, bond_type)
            else:
                if bond_type != existing_bond.GetBondType():
                    return False
                used_bonds.extend( [(a1,a2),(a2,a1)] ) # sharing bonds only possible for ring2ring
        
        tmp_mol = Chem.Mol(molecule)
        if sanitize(tmp_mol, kekulize=False) is None:
            return False
        
        # if past sanitize it means motif is valid for adding
        # set self.molecule and update curr_motif and father_motif
        if father_motif:
            used_atoms = []
            for atom_idx in father_motif.as_father.order:
                global_idx = ParseAtomInfo(father_motif.as_father.mol).global_idx(atom_idx)
                if global_idx in global_index_of_intersecting_atoms:
                    used_atoms.append(global_idx)
            father_motif.used.atoms.extend(used_atoms)
            father_motif.used.bonds.extend(used_bonds)
            curr_motif.used.atoms.extend(used_atoms)
            curr_motif.used.bonds.extend(used_bonds)
            father_motif.molecule = molecule
        curr_motif.global_atom_indices = new_atoms
        curr_motif.molecule = molecule
        self.molecule = molecule
        return True

    @property
    def smiles(self):
        copy_mol = Chem.Mol(self.molecule)
        clear_global_atom_info(copy_mol)
        return Chem.CanonSmiles(Chem.MolToSmiles(copy_mol))


def decode(treenet: NetworkPrediction, topk=5) -> EditableMol:
    molecule = EditableMol()
    root_prediction = treenet.root_info
    if not root_prediction: return molecule
    root = MotifNode(root_prediction)
    molecule.add_motif(root)
    stack = [root]
    curr_idx = 0
    for idx, do_traversal in enumerate(treenet.traversal_predictions):
        if not stack: break
        if not do_traversal:
            stack.pop()
            continue
        curr_idx += 1
        if curr_idx >= treenet.max_seq_length:
            break
        father_motif = stack[-1]
        add_motif_success = False
        father_is_ring = father_motif.num_atoms > 2
        for motif_prediction in treenet.get_topk_motifs(father_is_ring, curr_idx, topk):
            if add_motif_success: break
            if not motif_prediction: break # <pad> token
            father_motif.decorate_father()
            curr_motif = MotifNode(motif_prediction)
            candidates, used_indices = get_candidates(father_motif, curr_motif)
            candidate_scores = treenet.get_candidate_scores(curr_idx)
            # masked_indices = torch.arange(len(candidates),candidate_scores.shape[0]).long() # zip takes care of this
            # candidate_scores[masked_indices] = -torch.inf
            # # used_indices = torch.tensor(used)
            # # masked_indices = torch.cat((used_indices, masked_indices)).long()
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