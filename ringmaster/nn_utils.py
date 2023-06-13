"""NetworkPrediction: requires max_cand_size, cands_hidden_size, vocab """
import torch
import torch.nn.functional as F

class NetworkPrediction:
    max_cand_size = None
    cands_hidden_size = None
    vocab = None
    def __init__(self,
        tree_vec: torch.Tensor, # [max_seq_length, hidden_size]
        cls_pred: torch.Tensor, # [max_seq_length, num_cls]
        icls_pred: torch.Tensor, # [max_seq_length, num_icls]
        traversal_predictions: torch.Tensor, # [max_seq_length*2-1]
        candidate_vector_nn: torch.nn.modules, # [hidden_size] -> [max_cand_size, cands_hidden_size]
        candidate_nn: torch.nn.modules, # [max_cand_size, cands_hidden_size] -> [max_cand_size]
    ):
        self.tree_vec = tree_vec
        self.cls_pred = cls_pred
        self.icls_pred = icls_pred
        self.traversal_predictions = traversal_predictions
        self.candidate_vector_nn = candidate_vector_nn
        self.candidate_nn = candidate_nn

    @property
    def max_seq_length(self):
        return self.tree_vec.shape[0]

    @property
    def root_info(self):
        return self.icls_pred[0].max(dim=-1)[1].item()

    def get_topk_motifs(self, curr_idx, topk):
        # TODO: if fake score used for testing is implemented with one hot,
        #       it fails, anything greater than 1.25, however, works
        cls_scores = self.cls_pred[curr_idx]
        icls_scores = self.icls_pred[curr_idx]
        cls_scores = F.log_softmax(cls_scores, dim=-1)
        cls_scores_topk, cls_topk = cls_scores.topk(topk, dim=-1)
        final_topk = []
        for i in range(topk):
            clab = cls_topk[i]
            mask = NetworkPrediction.vocab.get_mask(clab)
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
        cand_vecs = self.candidate_vector_nn(cands_input).\
            reshape(NetworkPrediction.max_cand_size, NetworkPrediction.cands_hidden_size)
        return self.candidate_nn(cand_vecs).squeeze()
