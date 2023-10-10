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
from ringmaster.clip_model import MyCLIPModel
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool
from torch.nn.utils.parametrizations import orthogonal

# model imports
from transformers import GPT2Config, GPT2Model


from rdkit import RDLogger
# lg = RDLogger.logger() 
# lg.setLevel(RDLogger.CRITICAL)


class RingsNet(nn.Module):
    def __init__(self, params, vocab) -> None:
        super().__init__()
        self.hyperparams = Namespace(**params['trainingparams'])
        self.setupparams = Namespace(**params['setupparams'])
        self.vocab = vocab
        self.hidden_size = self.hyperparams.hidden_size
        
        cfg = params['trainingparams']
        self.gpt2cfg = gpt2_scaffold_cfg = gpt2_smiles_cfg = GPT2Config(**{ k: cfg[k] for k in GPT2Config().to_dict().keys() & cfg.keys() })
        self.scaffold_transformer = MyGPT2Model(gpt2_scaffold_cfg)
        gpt2_smiles_cfg.vocab_size = cfg['smiles_vocab_size']
        gpt2_smiles_cfg.n_positions = cfg['smiles_max_length']
        gpt2_smiles_cfg.add_cross_attention = True
        self.smiles_transformer = MyGPT2Model(gpt2_smiles_cfg)
        
        self.clip_loss = MyCLIPModel(self.hyperparams.n_embd, self.hyperparams.n_embd, self.hyperparams.clip_dim)

        # marko/jun19: stuff for graph tensor message passing
        atom_vocab_size = params['setupparams']['atom_vocab'].size()
        motif_vocab_size, imotif_vocab_size = vocab.size()
        bond_list_size = len(params['setupparams']['bond_list'])
        max_motif_neighbors = params['setupparams']['max_motif_neighbors'] # arbitrary
        seq_nn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.atom_gat = GPSConv(self.hidden_size, GINEConv(seq_nn), heads=4)
        self.motif_gat = GPSConv(self.hidden_size, GINEConv(seq_nn), heads=4)
        self.atom_type_embedder = orthogonal(nn.Embedding(atom_vocab_size, self.hidden_size))
        self.bond_type_embedder = orthogonal(nn.Embedding(bond_list_size, self.hidden_size))
        self.child_num_embedder = orthogonal(nn.Embedding(max_motif_neighbors, self.hidden_size))
        self.max_cand_size = params['setupparams']['max_cand_size']
        self.sequence_length = params['setupparams']['max_length']
        # self.topo_size = self.sequence_length * 2 - 1
        # dropout = params['trainingparams']['dropout_p']
        self.motif_type_embedder = nn.Embedding(motif_vocab_size, self.hidden_size//2)
        self.imotif_type_embedder = nn.Embedding(imotif_vocab_size, self.hidden_size//2)
        self.motif_edge_attr_embedder = orthogonal(nn.Embedding(max_motif_neighbors, self.hidden_size))
        # self.candvec_nn = nn.Linear(self.hidden_size, max_cand_size*cands_hidden_size)
        # self.cand_nn = nn.Linear(cands_hidden_size, 1) # output shape (max_cand_size, 1)
        # self.cand_nn = nn.Linear(self.hidden_size, self.max_cand_size)
        # self.topoNN = TopoNN(self.hidden_size, self.topo_size, num_heads=2)
        # self.topoNN = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.topo_size),
        #     nn.Sigmoid()
        # )
        self.clsNN = nn.Sequential(
                # nn.Dropout(dropout),
                nn.Linear(self.hyperparams.n_embd, motif_vocab_size)
        )
        # self.iclsNN = nn.Sequential(
        #         # nn.Dropout(dropout),
        #         nn.Linear(self.hyperparams.n_embd, imotif_vocab_size)
        # )
        self.bosNN = nn.Linear(self.hyperparams.latent_size, self.hyperparams.n_embd)
        self.smilesNN = nn.Linear(self.hyperparams.n_embd, cfg['smiles_vocab_size'])

        # mu, logvar for VAE
        self.mu = nn.Linear(self.hidden_size, self.hyperparams.latent_size)
        self.logvar = nn.Linear(self.hidden_size, self.hyperparams.latent_size)
        
        self.init_weights()
    

    def init_weights(self):
        self.apply(self._initialize_weights)
        # self._tie_or_clone_weights(self.get_logits, self.embedder.word_embeddings)


    def _initialize_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.gpt2cfg.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.gpt2cfg.initializer_range)
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
        # tree_vec = pad_graph_data(tree_vec, tbatch, self.sequence_length)
        x = global_add_pool(tree_vec, tbatch)
        mu = self.mu(x)
        logvar = self.logvar(x)
        # TODO:
        # 3. add <eos> token to end of tree_vec
        # 4. add <bos> token to beginning of tree_vec
        return mu, logvar



class ScaffoldVAE(pl.LightningModule):
    def __init__(self, params, vocab):
        super().__init__()
        self._starttime = None
        self.hyperparams = Namespace(**params['trainingparams'])
        self.setupparams = Namespace(**params['setupparams'])
        self.vocab = vocab
        # self.timesteps = self.hyperparams.timesteps
        self.save_hyperparameters()
        self.model = RingsNet(params, vocab)
        # self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.loss_fct = nn.CrossEntropyLoss()
        # self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hyperparams.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, self.hyperparams.warmup_steps, self.hyperparams.warmup_steps*2)
        return [optimizer], {"scheduler": scheduler, "interval": "step"}
        # return optimizer

    def get_loss(self, batch, batch_idx):
        graph_data, slices, smiles_input_ids, smiles_attention_mask = batch
        mu, logvar = self.model.encode_molecule(graph_data, slices) # TODO: morph back into shape (bs, seq_len, hidden_size)
        self.bs = mu.shape[0]
        z_sample = self.latent_sample(mu, logvar)
        
        # cls_pred, icls_pred = self.model.decode_molecule(z_sample)
        recon_loss, scaffold_hs, smiles_hs = self.token_discrete_loss(
            z_sample,
            graph_data,
            slices,
            smiles_input_ids,
            smiles_attention_mask
        )
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        self.log("train/kl_loss", kl_loss.mean(), batch_size=self.bs, on_step=True, sync_dist=True)
        clip_loss = self.model.clip_loss(smiles_hs, scaffold_hs)
        self.log("train/clip_loss", clip_loss, batch_size=self.bs, on_step=True, sync_dist=True)
        return (recon_loss + kl_loss).mean(dim=0) + clip_loss
        # return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)
        
    
    def latent_sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    
    # decoder_nll
    def token_discrete_loss(
        self,
        z_sample,
        graph_data,
        slices,
        smiles_input_ids,
        smiles_attention_mask
    ):
        batch = graph_data['motif'].batch
        def prep_labels(labels, batch=batch, seq_length=self.model.sequence_length):
            return pad_graph_data(labels[:,None], batch, seq_length, -100).squeeze().long()
        # icls_label = prep_labels(graph_data['motif'].icls)
        
        cls_label = prep_labels(graph_data['motif'].cls)
        # add <bos> token id to beginning of cls_label
        cls_label = torch.concat((torch.zeros(cls_label.shape[0], 1).to(cls_label.device).long(), cls_label), dim=1)
        # add <eos> token id to end of cls_label
        temp = cls_label * torch.arange(cls_label.shape[1], 0, -1).to(cls_label.device)
        cls_label[torch.arange(cls_label.shape[0]), temp.argmin(dim=1)] = 1
        
        scaffold_input_ids = torch.clone(cls_label).to(cls_label.device)
        scaffold_input_ids[scaffold_input_ids==-100] = 1
        
        scaffold_mask = torch.clone(cls_label).to(cls_label.device)
        scaffold_mask[cls_label!=-100] = 1
        scaffold_mask[cls_label==-100] = 0
        scaffold_mask = torch.concat((torch.ones(scaffold_mask.shape[0], 1).to(cls_label.device), scaffold_mask), dim=1)

        bos_embeds = self.model.bosNN(z_sample)
        scaffold_hidden_states = self.model.scaffold_transformer(
            bos_embeds=bos_embeds,
            input_ids=scaffold_input_ids,
            attention_mask=scaffold_mask
        )[0]
        
        smiles_attention_mask = torch.concat(
            (
                torch.ones(smiles_attention_mask.shape[0], 1).to(cls_label.device),
                smiles_attention_mask
            ),
            dim=1
        )
        smiles_hidden_states = self.model.smiles_transformer(
            bos_embeds=bos_embeds,
            input_ids=smiles_input_ids,
            attention_mask=smiles_attention_mask,
            encoder_hidden_states=scaffold_hidden_states
        )[0]
        
        scaffold_logits = self.model.clsNN(scaffold_hidden_states)
        shifted_scaffold_logits = scaffold_logits[..., :-1, :].contiguous()
        scaffold_loss = self.loss_fct(
            shifted_scaffold_logits.view(-1, shifted_scaffold_logits.size(-1)).contiguous(),
            cls_label.view(-1).contiguous()
        )
        # predict smiles loss 
        smiles_label = smiles_input_ids.clone()
        pad_indices = torch.where(smiles_input_ids == 2) # tokenizer.pad_token_id
        smiles_label[pad_indices] = -100
        smiles_logits = self.model.smilesNN(smiles_hidden_states)
        shifted_smiles_logits = smiles_logits[..., :-1, :].contiguous()
        smiles_loss = self.loss_fct(
            shifted_smiles_logits.view(-1, shifted_smiles_logits.size(-1)).contiguous(),
            smiles_label.view(-1).contiguous()
        )
        
        # loss = scaffold_loss
        loss = scaffold_loss + smiles_loss
        self.log("train/recon_loss", loss, batch_size=self.bs, on_step=True, sync_dist=True)
        return loss, scaffold_hidden_states[:,0,:], smiles_hidden_states[:,0,:]

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


from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
    
class MyGPT2Model(GPT2Model):
    def __init__(self, config):
        self.cfg = config
        super().__init__(config)

    def generate_next(self, bos_embeds, input_ids, encoder_hidden_states=None, encoder_attention_mask=None):
        device = bos_embeds.device
        inputs_embeds = self.wte(input_ids[:,1:])
        inputs_embeds = torch.concat((bos_embeds[:,None,:], inputs_embeds), dim=1)
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
        
        past_length = 0
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        if encoder_hidden_states is not None:
            assert encoder_attention_mask is not None
            # If a 2D or 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            # encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            # encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            # if encoder_attention_mask is None:
            #     encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        # if inputs_embeds is None:
        #     inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)
        past_key_values = tuple([None] * len(self.h))
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            hidden_states = outputs[0]

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        return hidden_states
        
    
    def forward(
        self,
        bos_embeds: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        
        inputs_embeds = self.wte(input_ids)
        inputs_embeds = torch.concat((bos_embeds[:,None,:], inputs_embeds), dim=1)
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        # if inputs_embeds is None:
        #     inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
