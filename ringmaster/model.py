from argparse import Namespace
from transformers import get_cosine_schedule_with_warmup
import time
import math
import random
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from diffusers import *
from typing import *

from ringmaster.nn_utils import (
    NetworkPrediction,
    pad_graph_data,
    inc_agraph,
    agg_agraph_info,
    mean_flat,
    timestep_embedding,
)
from ringmaster.timberwright import decode
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool

# model imports
from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertEmbeddings
)

from rdkit import RDLogger
lg = RDLogger.logger() 
lg.setLevel(RDLogger.CRITICAL)


class TopoNN(nn.Module):
    def __init__(self, hs, ts, num_heads=1) -> None:
        super().__init__()
        self.topo_query = nn.Parameter(torch.randn(ts, hs))
        self.topo_attn = nn.MultiheadAttention(
            embed_dim=hs,
            num_heads=num_heads,
            batch_first=True
        )
        self.out_proj = nn.Sequential(nn.Linear(hs,1), nn.Sigmoid())

    def forward(self, key):
        bs = key.shape[0]
        query = self.topo_query.expand(bs,-1,-1)
        attn_output, _ = self.topo_attn(query, key, key)
        return self.out_proj(attn_output)


class RingsNet(nn.Module):
    def __init__(self, params, vocab) -> None:
        super().__init__()
        self.hyperparams = Namespace(**params['trainingparams'])
        self.setupparams = Namespace(**params['setupparams'])
        self.vocab = vocab
        self.hidden_size = self.hyperparams.hidden_size
        self.bertconfig = BertConfig(
            max_position_embeddings=self.setupparams.max_length,
            num_attention_heads=self.hyperparams.num_heads,
            hidden_size=self.hyperparams.hidden_hidden_size,
            intermediate_size=self.hyperparams.intermediate_size,
            num_hidden_layers=self.hyperparams.num_hidden_layers,
            position_embedding_type=self.hyperparams.position_embedding_type,
            hidden_dropout_prob=self.hyperparams.dropout_p,
            attention_probs_dropout_prob=self.hyperparams.dropout_p,
            use_cache=False,
        )

        # marko/jun19: stuff for graph tensor message passing
        atom_vocab_size = params['setupparams']['atom_vocab'].size()
        motif_vocab_size, imotif_vocab_size = vocab.size()
        bond_list_size = len(params['setupparams']['bond_list'])
        max_motif_neighbors = params['setupparams']['max_motif_neighbors'] # arbitrary
        seq_nn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        ).requires_grad_(False)
        self.atom_gat = GPSConv(self.hidden_size, GINEConv(seq_nn), heads=4).requires_grad_(False)
        self.motif_gat = GPSConv(self.hidden_size, GINEConv(seq_nn), heads=4).requires_grad_(False)
        self.atom_type_embedder = nn.Embedding(atom_vocab_size, self.hidden_size).requires_grad_(False)
        self.bond_type_embedder = nn.Embedding(bond_list_size, self.hidden_size).requires_grad_(False)
        self.child_num_embedder = nn.Embedding(max_motif_neighbors, self.hidden_size).requires_grad_(False)
        self.max_cand_size = params['setupparams']['max_cand_size']
        self.sequence_length = params['setupparams']['max_length']
        self.topo_size = self.sequence_length * 2 - 1
        dropout = params['trainingparams']['dropout_p']
        self.motif_type_embedder = nn.Embedding(motif_vocab_size, self.hidden_size//2).requires_grad_(False)
        self.imotif_type_embedder = nn.Embedding(imotif_vocab_size, self.hidden_size//2).requires_grad_(False)
        self.motif_edge_attr_embedder = nn.Embedding(max_motif_neighbors, self.hidden_size).requires_grad_(False)
        # self.candvec_nn = nn.Linear(self.hidden_size, max_cand_size*cands_hidden_size)
        # self.cand_nn = nn.Linear(cands_hidden_size, 1) # output shape (max_cand_size, 1)
        self.cand_nn = nn.Linear(self.hidden_size, self.max_cand_size)
        self.topoNN = TopoNN(self.hidden_size, self.topo_size, num_heads=2)
        # self.topoNN = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.topo_size),
        #     nn.Sigmoid()
        # )
        self.clsNN = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, motif_vocab_size)
        )
        self.iclsNN = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, imotif_vocab_size)
        )


        # stuff for denoising transformer
        self.encoder = BertEncoder(self.bertconfig)

        self.in_channels = self.hidden_size * 2
        self.out_channels = self.hidden_size
        self.model_channels = self.hyperparams.time_channels
        
        time_embed_dim = self.model_channels * 2
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, self.hyperparams.hidden_hidden_size),
        )
        self.input_up_proj = nn.Sequential(
            nn.Linear(self.in_channels, self.hyperparams.hidden_hidden_size), # *2 for self-conditioning
            nn.Tanh(), 
            nn.Linear(self.hyperparams.hidden_hidden_size, self.hyperparams.hidden_hidden_size)
        )
        self.output_down_proj = nn.Sequential(
            nn.Linear(self.hyperparams.hidden_hidden_size, self.hyperparams.hidden_hidden_size),
            nn.Tanh(),
            nn.Linear(self.hyperparams.hidden_hidden_size, self.out_channels)
        )
        self.register_buffer("position_ids", torch.arange(self.bertconfig.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(self.bertconfig.max_position_embeddings, self.bertconfig.hidden_size).requires_grad_(False)
        self.LayerNorm = nn.LayerNorm(self.hyperparams.hidden_hidden_size, eps=self.bertconfig.layer_norm_eps)
        self.dropout = nn.Dropout(self.bertconfig.hidden_dropout_prob)
        # self.conditional_nn = nn.Linear(1,2) #TODO: change dimensions to match actual conditional data
        self.init_weights()
    

    def init_weights(self):
        self.apply(self._initialize_weights)
        # self._tie_or_clone_weights(self.get_logits, self.embedder.word_embeddings)


    def _initialize_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.bertconfig.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.bertconfig.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    def encode_molecule(self, graph_data, slices):
        # atom message passing
        device = graph_data['atoms'].atom_type.device
        gbatch, tbatch = graph_data['atoms'].batch, graph_data['motif'].batch
        atom_emb = self.atom_type_embedder(graph_data['atoms'].atom_type)
        bond_emb = self.bond_type_embedder(graph_data['atoms'].bond_type)
        n_child_emb = self.child_num_embedder(graph_data['atoms'].n_child)
        graph_agraph = inc_agraph(
            graph_data['atoms'].agraph,
            slices['atoms']['agraph'],
            slices['atoms']['bond_type'],
        )
        atom_pos_emb = agg_agraph_info(graph_agraph, bond_emb)
        atom_node_emb = self.atom_gat(
            x          = (atom_emb+atom_pos_emb),
            edge_index = graph_data['atoms', 'connect', 'atoms'].edge_index.long(),
            batch      = gbatch,
            edge_attr  = (bond_emb+n_child_emb)
        )

        # sum over atom embeddings to get motif embeddings
        motif_idx, atom_idx = graph_data["motif", "contains", "atoms"].edge_index
        motif_emb = torch.zeros(graph_data['motif'].num_nodes, self.hidden_size, device=device)
        motif_emb.scatter_add_(0, motif_idx.unsqueeze(1).expand(-1,self.hidden_size), atom_node_emb[atom_idx])
        # for loop version of the above code
        # x = graph_data["motif", "contains", "atoms"].edge_index.T
        # motif_emb = torch.zeros(graph_data['motif'].num_nodes, self.hidden_size, device=device)
        # for atom_idx, motif_idx in enumerate(x[:,0]):
        #     motif_emb[motif_idx] += atom_node_emb[x[atom_idx,1]]
        motif_emb += torch.cat((
            self.motif_type_embedder(graph_data['motif'].cls),
            self.imotif_type_embedder(graph_data['motif'].icls)
        ), dim=-1)
        motif_edge_emb = self.motif_edge_attr_embedder(graph_data['motif'].edge_attr)
        tree_agraph = inc_agraph(
            graph_data['motif'].agraph,
            slices['motif']['agraph'],
            slices['motif']['edge_attr'],
        )
        motif_pos_emb = agg_agraph_info(tree_agraph, motif_edge_emb)
        tree_vec = self.motif_gat(
            x          = (motif_emb + motif_pos_emb),
            edge_index = graph_data['motif', 'connect', 'motif'].edge_index.long(),
            batch      = tbatch,
            edge_attr  = motif_edge_emb
        ) # (bs*sq_len, hidden_size)
        tree_vec = nn.functional.normalize(tree_vec)
        # pad first to be max_seq_length
        return pad_graph_data(tree_vec, tbatch, self.sequence_length)


    def decode_molecule(self, tree_vec):
        cls_pred = self.clsNN(tree_vec) # (bs, seq_len, cls_vocab_size)
        icls_pred = self.iclsNN(tree_vec) # (bs, seq_len, icls_vocab_size)
        topo_pred = self.topoNN(tree_vec) # (bs, hidden_size)
        cands_input = tree_vec[:,:-1,:] + tree_vec[:,1:,:]
        assm_pred = self.cand_nn(cands_input)
        return cls_pred, icls_pred, assm_pred, topo_pred


    def forward(self, x_t, t, conditional=None):
        x_t = self.input_up_proj(x_t)
        seq_length = x_t.shape[1]
        temb = self.time_embed(timestep_embedding(t, self.model_channels))
        position_ids = self.position_ids[:, : seq_length ]
        emb_inputs = self.position_embeddings(position_ids) + x_t + temb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        # marko/may8: self-conditional, conditional
        # if conditional is not None:
        #     encoder_hidden_states = self.conditional_nn(conditional)[:,None,:]
        # else:
        #     encoder_hidden_states = None
        #     # encoder_extended_attention_mask = None
        encoder_hidden_states = None
        out = self.encoder(
            emb_inputs,
            encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_extended_attention_mask,
        )[0]
        out = self.output_down_proj(out)
        return out




class DiffusionTransformer(pl.LightningModule):
    def __init__(self, params, vocab):
        super().__init__()
        self._starttime = None
        self.hyperparams = Namespace(**params['trainingparams'])
        self.setupparams = Namespace(**params['setupparams'])
        self.vocab = vocab
        self.timesteps = self.hyperparams.timesteps
        self.save_hyperparameters()
        self.model = RingsNet(params, vocab)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        prediction_type = "sample"
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.timesteps,
            prediction_type=prediction_type
        )

    def denoise_sample(self, bs, num_inference_steps=100, skip_special_tokens=False):
        latent_size = self.hyperparams.hidden_size
        latents_shape = (bs, self.setupparams.max_length, latent_size)
        latents = torch.randn(latents_shape, device=self.device)
        latents = latents * self.noise_scheduler.init_noise_sigma
        self.noise_scheduler.set_timesteps(num_inference_steps)

        timesteps_tensor = self.noise_scheduler.timesteps.to(self.device)
        noise_pred = torch.zeros_like(latents).to(self.device)
        for t in timesteps_tensor:
            # t = t.expand(bs)
            latent_model_input = self.noise_scheduler.scale_model_input(latents, t)
            # predict the text embedding
            noise_pred = self.model(torch.cat((latent_model_input, noise_pred), dim=-1), t.expand(bs))
            # noise_pred = self.model(latent_model_input, t.expand(bs))
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        predicted_smiles = []
        payload = self.model.decode_molecule(latents)
        for tree_vec, cls_pred, icls_pred, _, topo_pred in zip(latents,*payload):
            networkprediction = NetworkPrediction(
                tree_vec=tree_vec,
                cls_pred=cls_pred,
                icls_pred=icls_pred,
                traversal_predictions=topo_pred.squeeze(),
                cand_nn=self.model.cand_nn,
            )
            predicted_smiles.append(decode(networkprediction).smiles)
        return predicted_smiles

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hyperparams.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, self.hyperparams.warmup_steps, self.hyperparams.warmup_steps*2)
        return [optimizer], {"scheduler": scheduler, "interval": "step"}
        # return optimizer

    def get_loss(self, batch, batch_idx):
        graph_data, slices = batch
        # if 'conditionals' in batch:
        #     conditionals = batch['conditionals']
        # else:
        #     conditionals = None
        x_embeds = self.model.encode_molecule(graph_data, slices) # TODO: morph back into shape (bs, seq_len, hidden_size)
        self.bs = bs = x_embeds.shape[0]

        noise = torch.randn(x_embeds.shape).to(x_embeds.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=x_embeds.device).long()
        # x_t for lsimple
        x_t = self.noise_scheduler.add_noise(x_embeds, noise, timesteps)
        # marko/may8: self-conditioning, conditioning
        prev_output = torch.zeros_like(x_t).to(x_embeds.device)
        # if random.random() > 0.5:
        #     conditionals = None
        conditionals = None
        if random.random() > 0.5:
            with torch.no_grad():
                prev_output = self.model(torch.cat((x_t, prev_output), dim=-1), timesteps, conditionals).detach()
        # model_output = self.model(x_t, timesteps)
        model_output = self.model(torch.cat((x_t, prev_output), dim=-1), timesteps, conditionals)
        decoder_nll = self.token_discrete_loss(model_output, graph_data, slices)
        lsimple = mean_flat((x_embeds - model_output) ** 2)
        self.log("train/lsimple", lsimple.mean(), batch_size=self.bs, on_step=True, sync_dist=True)
        # marko/may16: try tweeking the weight
        loss = 0.5 * lsimple + 2 * decoder_nll
        return loss.mean()
    
    # decoder_nll
    def token_discrete_loss(self, tree_vec, graph_data, slices):
        cls_pred, icls_pred, assm_pred, topo_pred = self.model.decode_molecule(tree_vec)
        batch = graph_data['motif'].batch
        def prep_labels(labels, batch=batch, seq_length=self.model.sequence_length):
            return pad_graph_data(labels[:,None], batch, seq_length, -100).squeeze().long()
        cls_label = prep_labels(graph_data['motif'].cls)
        icls_label = prep_labels(graph_data['motif'].icls)
        assm_label = prep_labels(graph_data['motif'].assm_cands)
        ptr = graph_data['motif'].ptr
        num_nodes = ptr[1:]-ptr[:-1]
        batch_num = torch.arange(num_nodes.shape[0]).to(ptr.device)
        topo_batch = torch.repeat_interleave(batch_num, (num_nodes*2-1), dim=0)
        topo_label = prep_labels(graph_data['motif'].order, topo_batch, self.model.topo_size)
        def calculate_loss(logits, labels):
            return self.ce_loss(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            ).reshape(labels.shape).mean(dim=-1)
        def calculate_bce_loss(logits, labels):
            msk = (labels != -100).long()
            loss = self.bce_loss(
                logits.view(-1),
                (labels*msk).view(-1).float()
            ).reshape(labels.shape)
            return (loss*msk).mean(dim=-1)
        cls_loss = calculate_loss(cls_pred, cls_label)
        icls_loss = calculate_loss(icls_pred, icls_label)
        assm_loss = calculate_loss(assm_pred, assm_label[:,1:])
        topo_loss = calculate_bce_loss(topo_pred, topo_label)
        self.log("train/cls_loss", cls_loss.mean(), batch_size=self.bs, on_step=True, sync_dist=True)
        self.log("train/icls_loss", icls_loss.mean(), batch_size=self.bs, on_step=True, sync_dist=True)
        self.log("train/assm_loss", assm_loss.mean(), batch_size=self.bs, on_step=True, sync_dist=True)
        self.log("train/topo_loss", topo_loss.mean(), batch_size=self.bs, on_step=True, sync_dist=True)
        return cls_loss + icls_loss + 2*assm_loss + 2*topo_loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)    
        self.log("train/loss", loss, batch_size=self.bs, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # print("embedder weight", self.model.embedder.word_embeddings.weight.sum().item())
        loss = self.get_loss(batch, batch_idx)
        self.log("val/loss", loss, batch_size=self.bs, on_step=True, sync_dist=True)

    def on_train_start(self):
        self._starttime = time.monotonic()

    def on_train_epoch_end(self):
        # marko/may26: for comparing training time
        if self.current_epoch == 0:
            time_used =  time.monotonic() - self._starttime
            self.log("train/time", time_used, sync_dist=True)