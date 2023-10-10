import torch
from torch import nn

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())

    return (caption_loss + image_loss) / 2.0


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    

class MyCLIPModel(nn.Module):
    def __init__(
            self,
            mol_emb_dim,
            ppt_emb_dim,
            projection_dim,
        ):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) # self.config.logit_scale_init_value
        self.mol_projection = ProjectionHead(embedding_dim=mol_emb_dim, projection_dim=projection_dim)
        self.ppt_projection = ProjectionHead(embedding_dim=ppt_emb_dim, projection_dim=projection_dim)

    def forward(self, mol_embeds, ppt_embeds):
        mol_embeds = self.mol_projection(mol_embeds)
        ppt_embeds = self.ppt_projection(ppt_embeds)

        # normalized features
        mol_embeds = mol_embeds / mol_embeds.norm(p=2, dim=-1, keepdim=True)
        ppt_embeds = ppt_embeds / ppt_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(ppt_embeds, mol_embeds.t()) * logit_scale
        # logits_per_image = logits_per_text.t()

        return clip_loss(logits_per_text)
