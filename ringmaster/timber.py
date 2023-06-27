"""MotifNode: requires vocab"""

from rdkit import Chem
from dataclasses import dataclass
from functools import cached_property, partial, cache
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
        self.target_atoms = [] # used in encoding
        self.molecule = None
        self.global_atom_indices = set()
        self.used = UsedCandidates(atoms=[], bonds=[])
    
    def __repr__(self) -> str:
        return f"""MotifNode(\n\tsmiles: {self.ismiles},\n\tindices: {self.global_atom_indices},\n\tused: {self.used}\n)"""

    @cached_property
    def num_atoms(self) -> int:
        return get_mol(self.ismiles).GetNumAtoms()

    @cached_property
    def mol(self) -> Chem.rdchem.Mol:
        """return canonicalized mol for current motif"""
        return CanonicalizedMol(*canonicalize(get_mol(self.ismiles)))

    @cached_property
    def attachment_info(self) -> MotifAttachmenInfo:
        """return attachment info for current motif"""
        mol = get_mol(self.ismiles)
        if mol.GetNumAtoms() == 1:
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
            inter_atoms = [idxfunc(a) for a in mol.GetAtoms() if a.GetAtomMapNum() > 0]
            anchors = [a for a in attach_points if is_anchor(mol.GetAtomWithIdx(a), inter_atoms, idxfunc=idxfunc)] #all attach points are labeled with 1
            # assert len(anchors) == 2, f"attach points: {attach_points}, anchors: {anchors}, smiles: {self.ismiles}"
            attach_points = [a for a in attach_points if a not in anchors]
            attach_points = [anchors[0]] + attach_points + [anchors[1]] #force the attach_points to be a chain like anchor ... anchor
            anchor_smiles = [get_anchor_smiles(mol, a, idxfunc) for a in anchors]
            is_symmetric = anchor_smiles[0] == anchor_smiles[1]
        assert len(anchors) <= 2, f"attach points: {attach_points}, anchors: {anchors}, smiles: {self.ismiles}"
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

    def decorate_father(self):
        """
        call to set the .as_father attribute
        -> get canonicalized motif with:
        1. global atom index information
        2. used atom information
        """
        assert self.molecule is not None
        assert self.global_atom_indices
        molecule = Chem.Mol(self.molecule)
        set_global_atom_info(molecule)
        for ans in self.target_atoms:
            set_atom_label(molecule, ans)
        if self.num_atoms != 1: # one atom is special case
            label_all_used_atoms(molecule, self.used.atoms)
        mol = get_clique_mol(molecule, self.global_atom_indices)
        mol, atom_order = canonicalize(mol)
        atom_order = check_rotation_order(mol, atom_order)
        self.as_father = FatherMotifInfo(mol=mol, order=atom_order, used=self.used.used)

