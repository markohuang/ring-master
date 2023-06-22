"""MolParser: requires bond_list"""

import torch
from rdkit import Chem
import networkx as nx
from collections import defaultdict
from functools import cached_property
from ringmaster.chem_utils import *
from ringmaster.timber import MotifNode

@dataclass
class MolGraphTensors:
    fnode: torch.Tensor
    fmess: torch.Tensor
    agraph: torch.Tensor
    assm_cands: Optional[torch.Tensor] = None
    cgraph: Optional[torch.Tensor] = None
    def __iter__(self):
        return iter((self.fnode, self.fmess, self.agraph, self.assm_cands, self.cgraph))

@dataclass
class MolData:
    mol_tree: MolGraphTensors
    mol_graph: MolGraphTensors
    order: torch.Tensor
    def __iter__(self):
        return iter((self.mol_tree, self.mol_graph, self.order))


def create_pad_tensor(alist):
    max_len = max([len(a) for a in alist])
    for a in alist:
        pad_len = max_len - len(a)
        a.extend([-1] * pad_len)
    return torch.IntTensor(alist)


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
        # marko/jun14: edge index now starts at 0
        edge_dict[(u, v)] = eid = len(edge_dict) # edge index starts at 1
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


class MolParser:
    vocab = None
    atom_vocab = None
    bond_list = None
    def __init__(self, smiles: str) -> None:
        self.smiles = smiles
        self.mol = get_mol(smiles)
        assert self.mol is not None, print(f'Invalid SMILES: {smiles}')
        self.n_atoms = self.mol.GetNumAtoms()
        self.father_of, self.order = self.update_tree_order()
        self.label_tree()
    
    @property
    def tensors(self):
        return MolData(
            mol_tree=tensorize(self.tree, vocab=self.vocab, is_motif=True),
            mol_graph=tensorize(self.graph, vocab=self.atom_vocab, is_motif=False),
            order=torch.IntTensor(self.order)
        )
    
    @cached_property
    def clusters(self):
        if self.n_atoms == 1: #special case
            return [(0,)], [[0]]
        clusters = []
        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                clusters.append( (a1,a2) )
        ssr = [tuple(x) for x in Chem.GetSymmSSSR(self.mol)]
        clusters.extend(ssr)
        if 0 not in clusters[0]: #root is not node[0]
            for i,cls in enumerate(clusters):
                if 0 in cls:
                    clusters = [clusters[i]] + clusters[:i] + clusters[i+1:]
                    #clusters[i], clusters[0] = clusters[0], clusters[i]
                    break
        return clusters

    @property
    def n_clusters(self):
        return len(self.clusters)
    
    @property
    def atom_cls_indices(self):
        atom_cls = [[] for _ in range(self.n_atoms)]
        for i in range(len(self.clusters)):
            for atom in self.clusters[i]:
                atom_cls[atom].append(i)
        return atom_cls

    @cached_property
    def graph(self):
        assert self.bond_list is not None
        graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        for atom in self.mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())
        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = self.bond_list.index( bond.GetBondType() )
            graph[a1][a2]['label'] = btype
            graph[a2][a1]['label'] = btype
        return graph
    
    @cached_property
    def tree(self):
        # tree representation of molecule
        graph = nx.empty_graph( self.n_clusters )
        for atom, nei_cls in enumerate(self.atom_cls_indices):
            if len(nei_cls) <= 1: continue # if atom is only in one aba or ring group
            bonds = [c for c in nei_cls if len(self.clusters[c]) == 2]
            rings = [c for c in nei_cls if len(self.clusters[c]) > 4] # TODO: need to change to 2
            # if atom belongs to more than 2 (e.g., atom-bond-atom) clusters
            if len(nei_cls) > 2 and len(bonds) >= 2:
                # add edge between [atom] cluster (appended to clusters) and nei_cls
                self.clusters.append([atom])
                c2 = self.n_clusters - 1
                graph.add_node(c2)
                for c1 in nei_cls:
                    graph.add_edge(c1, c2, weight = 100) # TODO: remove weights
            # if atom belongs to more than 2 rings
            elif len(rings) > 2: #Bee Hives, len(nei_cls) > 2 
                self.clusters.append([atom]) #temporary value, need to change
                c2 = self.n_clusters - 1
                graph.add_node(c2)
                for c1 in nei_cls:
                    graph.add_edge(c1, c2, weight = 100)
            # pairwise connect clusters that the atom belongs to and set edge weight to be intersection of clusters
            else:
                for i,c1 in enumerate(nei_cls):
                    for c2 in nei_cls[i + 1:]:
                        inter = set(self.clusters[c1]) & set(self.clusters[c2])
                        graph.add_edge(c1, c2, weight = len(inter))
        n, m = len(graph.nodes), len(graph.edges)
        assert n - m <= 1 #must be connected
        tree = graph if n - m == 1 else nx.maximum_spanning_tree(graph)
        return nx.DiGraph(tree)


    def update_tree_order(self):
        """updates moltree/clusters to match the order of dfs traversal"""
        order, pa = [], {}
        def dfs(order, pa, x, fa):
            pa[x] = fa 
            # fix traversal order
            sorted_child = sorted([ y for y in self.tree[x] if y != fa ]) #better performance with fixed order
            for idx,y in enumerate(sorted_child):
                # x -> curr, y -> child
                # each child of node x will have label ++ in visiting order
                self.tree[x][y]['label'] = 0 
                self.tree[y][x]['label'] = idx + 1 #position encoding
                # prev_sib contains all previously visited sibling nodes + parent + grandfather node
                order.append( (x,y,1) )
                dfs(order, pa, y, x)
                order.append( (y,x,0) )
        dfs(order, pa, 0, -1)
        order.append( (0, None, 0) )
        # update motif labeling to match tree traversal order
        order_list = [0] 
        order_labels = []
        for _, curr, topo in order:
            order_labels.append(topo)
            if topo == 1:
                order_list.append(curr)
        order_map = {value: index for index, value in enumerate(order_list)}
        self.tree = nx.relabel_nodes(self.tree, order_map)
        # update clusters
        self.clusters = [self.clusters[i] for i in order_list]
        # update parents
        father_of = {} 
        for k,v in pa.items():
            if v == -1:
                father_of[k] = v
                continue
            father_of[order_map[k]] = order_map[v]
        return father_of, order_labels


    def label_tree(self):
        mol = Chem.Mol(self.mol)
        set_global_atom_info(mol)
        used = defaultdict(list)
        first_time = True
        for i,cls in enumerate(self.clusters):
            fa_cls_idx = self.father_of[i]
            fa_cls = self.clusters[fa_cls_idx]
            inter_atoms = set(cls) & set(fa_cls) if fa_cls_idx >= 0 else set()
            # cmol is the mol for the cluster of atoms within the original smiles
            cmol, inter_label = get_inter_label(mol, cls, inter_atoms)
            ismiles = get_smiles(cmol)
            clear_global_atom_info(cmol)
            smiles = get_smiles(cmol)
            self.tree.nodes[i].update(dict(
                smiles=smiles,
                ismiles=ismiles,
                inter_label=inter_label,
                label=(smiles, ismiles) if len(cls) > 1 else (smiles, smiles),
                cluster=cls,
                assm_cands=-100,
            ))
            if (fa_cls_idx == 0 and first_time) or (fa_cls_idx >= 0 and len(self.clusters[ fa_cls_idx ]) > 2): #uncertainty occurs in assembly
                first_time = False
                father_motif = MotifNode(self.tree.nodes[fa_cls_idx]['ismiles'])
                father_motif.global_atom_indices = fa_cls
                father_motif.used.atoms.extend(used[fa_cls_idx])
                father_motif.target_atoms = inter_atoms
                father_motif.molecule = mol
                father_motif.decorate_father()
                father = father_motif.as_father
                label, global_label = [], []
                fa_parser = ParseAtomInfo(father.mol)
                for atom_idx in father.order:
                    if fa_parser.is_label(atom_idx):
                        label.append(atom_idx)
                        global_label.append(fa_parser.global_idx(atom_idx))
                used[i].extend(global_label)
                used[fa_cls_idx].extend(global_label)
                is_symmetric = True
                # construct cands
                inter_size = len(inter_atoms)
                if inter_size > 1:
                    anchor_smiles = [a[1] for a in inter_label]
                    assert len(anchor_smiles) == 2
                    is_symmetric = anchor_smiles[0] == anchor_smiles[1]
                candidates = get_all_candidates(father.order, inter_size, is_symmetric)
                
                if not is_symmetric:
                    copy_mol = Chem.Mol(mol)
                    clear_global_atom_info(copy_mol)
                    for j,l in enumerate(global_label): # label atoms' global idx
                        copy_mol.GetAtomWithIdx(l).SetAtomMapNum(j+1)
                    curr_mol = get_clique_mol(copy_mol, cls)
                    curr_mol_reordered, cluster_order = canonicalize(curr_mol)
                    reverse_assm = False
                    curr_idx = -1
                    for a in cluster_order:
                        new_idx = curr_mol_reordered.GetAtomWithIdx(a).GetAtomMapNum()
                        if new_idx:
                            if new_idx < curr_idx:
                                reverse_assm = True
                                break
                            curr_idx = new_idx
                    if reverse_assm:
                        self.tree.nodes[i]['assm_cands'] = candidates.index(label[::-1])
                    else:
                        self.tree.nodes[i]['assm_cands'] = candidates.index(label)
                else:
                    if label in candidates:
                        self.tree.nodes[i]['assm_cands'] = candidates.index(label)
                    else:
                        self.tree.nodes[i]['assm_cands'] = candidates.index(label[::-1]) # reverse order

                child_order = self.tree[i][fa_cls_idx]['label']
                diff = set(cls) - set(fa_cls)
                for fa_atom in inter_atoms:
                    for ch_atom in self.graph[fa_atom]:
                        if ch_atom in diff:
                            label = self.graph[ch_atom][fa_atom]['label']
                            if type(label) is int: #in case one bond is assigned multiple times
                                self.graph[ch_atom][fa_atom]['label'] = (label, child_order)
            else:
                used[i].extend(inter_atoms)


