"""MotifNode: requires vocab"""

from rdkit import Chem
from dataclasses import dataclass
from functools import cached_property, partial
from itertools import chain
from ringmaster.chem_utils import *
from collections import abc
from typing import *

## ------------------ dataclasses ------------------ ##
@dataclass
class MotifAttachmenInfo:
    """attachment info for current motif"""
    is_symmetric: bool
    attach_point_indices: list[int]
    attach_point_atoms: list[Chem.rdchem.Atom]
    num_attach_points: int

@dataclass
class FatherMotifInfo:
    """responsibilities as a father: canonicalized mol with global+used atoms information"""
    mol: Chem.rdchem.Mol
    order: List[int]
    used: List[int]
        
@dataclass
class UsedCandidates:
    atoms: List[int]
    bonds: List[Tuple[int, int]]
    @property
    def used(self):
        return self.atoms + self.bonds

@dataclass
class CanonicalizedMol:
    mol: Chem.rdchem.Mol
    order: List[int]
    def __iter__(self):
        return iter((self.mol, self.order))


## ------------------ motif suite ------------------ ##
class MotifNode:
    """contains all information for a motif"""
    vocab = None
    def __init__(self, key) -> None:
        self.ismiles = key if type(key) == str else self.vocab.get_ismiles(key)
        self.molecule = None
        self.target_atoms = [] # used in encoding
        self.global_atom_indices = set()
        self.used = UsedCandidates(atoms=[], bonds=[])

    @cached_property
    def mol(self) -> Chem.rdchem.Mol:
        """return canonicalized mol for current motif"""
        return CanonicalizedMol(*canonicalize(get_mol(self.ismiles)))

    @property
    def attachment_info(self) -> MotifAttachmenInfo:
        """return attachment info for current motif"""
        if mol.GetNumAtoms() == 1:
            mol = get_mol(self.ismiles)
            attach_points = [0]
        else:
            mol, atom_order = canonicalize(mol)
            attach_points = [a for a in atom_order if mol.GetAtomWithIdx(a).GetAtomMapNum() > 0]
        num_attach_points = len(attach_points)
        anchors = attach_points
        def idxfunc(x):
            return x.GetIdx()
        is_symmetric = False
        if num_attach_points == 1:
            anchor_smiles = [self.ismiles]
        elif num_attach_points == 2:
            anchor_smiles = [get_anchor_smiles(mol, a, idxfunc) for a in anchors]
            is_symmetric = anchor_smiles[0] == anchor_smiles[1]
        elif num_attach_points > 0:
            anchors = [a for a in attach_points if is_anchor(mol.GetAtomWithIdx(a), [0])] #all attach points are labeled with 1
            attach_points = [a for a in attach_points if a not in anchors]
            attach_points = [anchors[0]] + attach_points + [anchors[1]] #force the attach_points to be a chain like anchor ... anchor
            anchor_smiles = [get_anchor_smiles(mol, a, idxfunc) for a in anchors]
            is_symmetric = anchor_smiles[0] == anchor_smiles[1]
        assert len(anchors) <= 2
        attach_point_atoms = [mol.GetAtomWithIdx(a) for a in attach_points]
        return MotifAttachmenInfo(
            is_symmetric=is_symmetric,
            attach_point_indices=attach_points,
            num_attach_points=num_attach_points,
            attach_point_atoms=attach_point_atoms
        )
    
    @property
    def global_atom_indices(self):
        return self.__dict__['global_atom_indices']

    @global_atom_indices.setter
    def global_atom_indices(self, global_atom_indices):
        assert isinstance(global_atom_indices, abc.Iterable)
        self.__dict__['global_atom_indices'] = global_atom_indices

    @property
    def as_father(self):
        """
        get canonicalized motif with:
        1. global atom index information
        2. used atom information
        """
        assert self.molecule is not None
        assert self.global_atom_indices
        molecule = Chem.Mol(self.molecule)
        set_global_atom_info(molecule)
        for ans in self.target_atoms:
            set_atom_label(molecule, ans)
        label_all_used_atoms(molecule, self.used.used)
        mol = get_clique_mol(molecule, self.global_atom_indices)
        mol, atom_order = canonicalize(mol)
        atom_order = check_rotation_order(mol, atom_order)
        return FatherMotifInfo(mol=mol, order=atom_order, used=self.used.used)
    
    def get_candidates(self, child_motif):
        father = self.as_father
        father_global_num = ParseAtomInfo(father.mol).global_idx
        if father.mol.GetNumAtoms() == 1:
            candidates = [[father_global_num(0)]]
            return candidates
        child = child_motif.attachment_info
        all_candidates = get_all_candidates(
            father.order,
            child.num_attach_points,
            child.is_symmetric
        )
        def not_the_same_atom(x):
            return child.attach_point_atoms[0].GetSymbol() !=\
                    father.mol.GetAtomWithIdx(x).GetSymbol()
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


class EditableMol:
    """the editable molecule built by decoding network predictions"""
    def __init__(self):
        self.molecule = Chem.RWMol()

    def add_motif(self, curr_motif, father_motif=None, atom_pairs=[]):
        """
        if succeeds, updates 
            the curr_motif and father_motif with new used atoms
            the curr_motif with its global atom indices
        if fails, undos the trial and returns False
        """
        global_index_of_intersecting_atoms = [x[0] for x in atom_pairs]
        atom_map = {y: x for x, y in atom_pairs}
        molecule = Chem.Mol(self.molecule)
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
        if father_motif is not None:
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