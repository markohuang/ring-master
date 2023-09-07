"""NetworkPrediction: requires max_cand_size, cands_hidden_size, vocab """
import torch, math
import torch.nn.functional as F
from torch_geometric.utils import unbatch
from torch.nn.utils.rnn import pad_sequence

def pad_graph_data(tensor, batch, sequence_length, value=0):
    unbatched = list(unbatch(tensor, batch))
    unbatched[0] = F.pad(unbatched[0], (0,0,0,sequence_length-unbatched[0].shape[0]), value=value)
    return pad_sequence(unbatched, batch_first=True, padding_value=value)


def inc_agraph(agraph, slice1, slice2):
    """slice1 is number of nodes slice, slice2 is number of edges slice"""
    device = agraph.device
    num_nodes, num_neighbors = agraph.size()
    msk1 = (agraph != -1).to(torch.int).to(device)
    index_list = torch.arange(num_nodes).unsqueeze(1).to(device)
    msk2_1d =  ((index_list >= slice1[:-1]) & (index_list < slice1[1:])).float() @ slice2[:-1, None].float()
    msk2 = msk2_1d.repeat(1, num_neighbors)
    msk = msk1 * msk2
    return (agraph + msk).long()


def agg_agraph_info(agraph, edge_emb):
    device = agraph.device
    num_neighbors = agraph.shape[-1]
    hidden_size = edge_emb.shape[-1]
    bond_lookup_table = torch.cat((edge_emb, torch.zeros(1,hidden_size, device=device)))
    agraph = agraph.clone().detach()
    agraph[agraph == -1] = bond_lookup_table.shape[0]-1
    return bond_lookup_table.index_select(0, agraph.view(-1)).view(-1,num_neighbors,hidden_size).sum(dim=1)


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

# from functools import cache
# @cache
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class NetworkPrediction:
    max_cand_size = None
    cands_hidden_size = None
    vocab = None
    def __init__(self,
        tree_vec: torch.Tensor, # [max_seq_length, hidden_size]
        cls_pred: torch.Tensor, # [max_seq_length, num_cls]
        icls_pred: torch.Tensor, # [max_seq_length, num_icls]
        traversal_predictions: torch.Tensor, # [max_seq_length*2-1]
        cand_nn: torch.nn.modules,
    ):
        self.tree_vec = tree_vec
        self.cls_pred = cls_pred
        self.icls_pred = icls_pred
        self.traversal_predictions = traversal_predictions
        self.cand_nn = cand_nn

    @property
    def max_seq_length(self):
        return self.tree_vec.shape[0]

    @property
    def root_info(self):
        return self.icls_pred[0].max(dim=-1)[1].item()

    def get_topk_motifs(self, father_is_ring, curr_idx, topk):
        # TODO: if fake score used for testing is implemented with one hot,
        #       it fails, anything greater than 1.25, however, works
        cls_scores = self.cls_pred[curr_idx]
        icls_scores = self.icls_pred[curr_idx]
        cls_scores = F.log_softmax(cls_scores, dim=-1)
        cls_scores_topk, cls_topk = cls_scores.topk(topk, dim=-1)
        final_topk = []
        for i in range(topk):
            clab = cls_topk[i]
            # len(re.findall(r':1',str)) >= 2
            # regex to count :1 in string
            mask = NetworkPrediction.vocab.get_mask(clab, father_is_ring).to(cls_scores.device)
            masked_icls_scores = F.log_softmax(icls_scores + mask, dim=-1)
            icls_scores_topk, icls_topk = masked_icls_scores.topk(topk, dim=-1)
            topk_scores = cls_scores_topk[i].unsqueeze(-1) + icls_scores_topk
            final_topk.append( (topk_scores, clab.unsqueeze(-1).expand(topk), icls_topk) )

        topk_scores, cls_topk, icls_topk = zip(*final_topk)
        topk_scores = torch.cat(topk_scores, dim=-1)
        cls_topk = torch.cat(cls_topk, dim=-1)
        icls_topk = torch.cat(icls_topk, dim=-1)

        topk_scores, topk_index = topk_scores.topk(topk, dim=-1)
        cls_topk = cls_topk.squeeze()[topk_index]
        icls_topk = icls_topk.squeeze()[topk_index]
        # return topk_scores, cls_topk.squeeze().tolist(), icls_topk.squeeze().tolist()
        return icls_topk.squeeze().tolist()
        
    def get_candidate_scores(self, curr_idx):
        assert curr_idx > 0 and curr_idx < len(self.cls_pred)
        curr_cls_emb = self.tree_vec[curr_idx]
        fa_cls_emb = self.tree_vec[curr_idx-1]
        cands_input = fa_cls_emb + curr_cls_emb
        return self.cand_nn(cands_input)
