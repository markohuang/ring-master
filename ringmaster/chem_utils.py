import rdkit.Chem as Chem
from typing import *
from dataclasses import dataclass

#----------- Visualization ----------
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from IPython.display import SVG

def draw_graph(mol: Union[Chem.rdchem.Mol, str], size=(450,400)):
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    d2d = rdMolDraw2D.MolDraw2DSVG(*size)
    d2d.drawOptions().addAtomIndices=True
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return SVG(d2d.GetDrawingText())


## ------------------ atom idx suite ------------------ ##
ATOM_NUM_DIVISOR = 100
ATOM_USED_DIVISOR = 10
ATOM_LABEL = 1
ATOM_USED = 9

@dataclass
class ParseAtomInfo:
    """atom info for current motif"""
    mol: Chem.rdchem.Mol

    def num(self, idx):
        return self.mol.GetAtomWithIdx(idx).GetAtomMapNum()

    def label(self, idx):
        return (self.num(idx)%ATOM_NUM_DIVISOR)//ATOM_USED_DIVISOR

    def is_label(self, idx):
        return self.label(idx) == ATOM_LABEL

    def is_used(self, idx):
        return self.label(idx) == ATOM_USED

    def global_idx(self, idx):
        """get global atom index given atom index within cluster"""
        return self.num(idx)//ATOM_NUM_DIVISOR

    def used_idx(self, idx):
        return self.num(idx)%ATOM_USED_DIVISOR
    

def idxfunc(atom):
    """maps atom map number back to atom index"""
    return atom.GetAtomMapNum()//ATOM_NUM_DIVISOR

def set_global_atom_info(mol):
    for a in mol.GetAtoms():
        a.SetAtomMapNum( a.GetIdx() * ATOM_NUM_DIVISOR )

def label_all_used_atoms(mol, used_atoms):
    for offset, atom in enumerate(used_atoms):
        set_used_atoms(mol, atom, offset+1)

def set_used_atoms(mol, idx, offset=1):
    atom = mol.GetAtomWithIdx(idx)
    atom.SetAtomMapNum(atom.GetAtomMapNum()+ATOM_USED*ATOM_USED_DIVISOR+offset)

def set_atom_label(mol, idx):
    atom = mol.GetAtomWithIdx(idx)
    atom.SetAtomMapNum(atom.GetAtomMapNum()+ATOM_LABEL*ATOM_USED_DIVISOR)

def clear_global_atom_info(mol):
    parser = ParseAtomInfo(mol)
    for atom in mol.GetAtoms():
        if parser.is_used(atom.GetIdx()): 
            atom.SetAtomMapNum(atom.GetAtomMapNum()%ATOM_NUM_DIVISOR)
            continue
        atom.SetAtomMapNum(0)
    return mol


## ------------------ mol helpers ------------------ ##
def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # clearAromaticFlag as per https://github.com/wengong-jin/hgraph2graph/issues/24
    if mol is not None: Chem.Kekulize(mol, clearAromaticFlags=True) 
    return mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def copy_atom(atom, atommap=True):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap: 
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def sanitize(mol, kekulize=True):
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
        #if bt == Chem.rdchem.BondType.AROMATIC and not aromatic:
        #    bt = Chem.rdchem.BondType.SINGLE
    return new_mol

def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol) 
    #if tmp_mol is not None: new_mol = tmp_mol
    return new_mol


## ------------------ encoding helpers ------------------ ##
def get_inter_label(mol, atoms, inter_atoms, idxfunc=idxfunc): # complete mol, atoms in cls in mol, inter_atom (set of atoms) in intersection of cls and parent cls
    new_mol = get_clique_mol(mol, atoms)
    if new_mol.GetNumBonds() == 0: # if single atom then smiles and ismiles are the same
        inter_atom = list(inter_atoms)[0]
        for a in new_mol.GetAtoms():
            a.SetAtomMapNum(0) # brings smarts back to smiles
        return new_mol, [ (inter_atom, Chem.MolToSmiles(new_mol)) ]

    inter_label = []
    for a in new_mol.GetAtoms():
        idx = idxfunc(a) # atommapnum coded to be atom.getidx()+1, this reverses the process
        if idx in inter_atoms and is_anchor(a, inter_atoms, idxfunc): # if a has neighbors outside of intersection
            inter_label.append( (idx, get_anchor_smiles(new_mol, idx, idxfunc)) ) # if anchor atommapnum = 1, else 0

    for a in new_mol.GetAtoms():
        a.SetAtomMapNum( 1 if idxfunc(a) in inter_atoms else 0 )
    return new_mol, inter_label


## ------------------ canonicalization ------------------ ##
def rotate_list(lst):
    # rotate so index 0 is the first element
    k = lst.index(0)
    rotated_lst = lst[k:] + lst[:k]
    return rotated_lst

def check_rotation_order(mol, order):
    curr_offset = -1
    parser = ParseAtomInfo(mol)
    for a in order:
        offset = parser.used_idx(a)
        if offset > 0:
            if offset <= curr_offset: 
                return order[::-1]
            curr_offset = offset
    return order

def canonicalize(m: Chem.Mol):
    atom_maps = [atom.GetAtomMapNum() for atom in m.GetAtoms()]
    clear_global_atom_info(m)
    neworder = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(m))])))[1]
    for i, atom_map in enumerate(atom_maps): # ∵this might change the traversal order 
        m.GetAtomWithIdx(i).SetAtomMapNum(atom_map)
    traversal_order = list(neworder.index(x) for x in range(len(neworder)))
    new_order = rotate_list(traversal_order)
    if len(new_order) > 2:
        # ∴bond traversal order may not be consistent: order could be [0, 2, 4, 3, 1] or [0, 1, 3, 4, 2]
        # if the second atom is closer to the first atom than the last atom is to the first atom, reverse the order
        if new_order[1] - new_order[0] > new_order[-1] - new_order[0]:
            new_order = new_order[::-1]
            new_order = rotate_list(new_order)
    return Chem.RenumberAtoms(m, neworder), new_order


## ------------------ attachment info helpers ------------------ ##
def is_anchor(atom, inter_atoms, idxfunc=idxfunc):
    for a in atom.GetNeighbors():
        if idxfunc(a) not in inter_atoms:
            return True
    return False
            
def get_anchor_smiles(mol, anchor, idxfunc=idxfunc):
    copy_mol = Chem.Mol(mol)
    for a in copy_mol.GetAtoms():
        idx = idxfunc(a)
        if idx == anchor: a.SetAtomMapNum(1)
        else: a.SetAtomMapNum(0)

    return get_smiles(copy_mol)


## ------------------ candidate helpers ------------------ ##
def get_all_candidates(fa_cluster_order, inter_size, is_symmetric):
    # find father candidates
    if inter_size == 1:
        cands = [ [x] for x in fa_cluster_order ]
    elif is_symmetric:
        fa_cluster2 = fa_cluster_order + fa_cluster_order # [0,1,2,3,4,0,1,2,3,4] next line is for wrap-around only
        cands = [fa_cluster2[i : i + inter_size] for i in range(len(fa_cluster_order))] #not pairs if inter_size >= 3
    else: 
        fa_cluster2 = fa_cluster_order + fa_cluster_order
        cands = [fa_cluster2[i : i + inter_size] for i in range(len(fa_cluster_order))]
        fa_cluster2 = fa_cluster2[::-1]
        cands += [fa_cluster2[i : i + inter_size] for i in range(len(fa_cluster_order))]
    return cands

