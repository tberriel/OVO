from typing import Any, Tuple
from torchvision.transforms import Resize, Normalize, CenterCrop, Compose
import open_clip
import torch

def siglip_cosine_similarity(txt_embeds:torch.Tensor, img_embed: torch.Tensor, logit_scale:torch.Tensor, logit_bias:float) -> torch.Tensor:
    p = txt_embeds.to(img_embed.dtype) 
    logits = torch.mm(img_embed, p.T) * logit_scale.exp()+logit_bias # obj x phrases
    output = torch.sigmoid(logits)
    return output

def clip_cosine_similarity(txt_embeds:torch.Tensor, img_embed: torch.Tensor) -> torch.Tensor:
    p = txt_embeds.to(img_embed.dtype)
    output = torch.mm(img_embed, p.T)   # obj x phrases
    return output

def fuse_clips(clip_g:torch.Tensor, clip_seg:torch.Tensor, clip_bbox:torch.Tensor, embed_type: str, w_masked: float, w_global: float)-> torch.Tensor:
    cos = torch.nn.CosineSimilarity(dim=-1, eps = 1e-6)
    if embed_type == "hovsg" or embed_type == "fixed_weights" :
        w_local = w_masked
        clip_l = torch.nn.functional.normalize(clip_seg*w_local+clip_bbox*(1-w_local), p=2, dim=-1)

        if embed_type == "fixed_weights":
            w_global = w_global
        else:
            w_global = cos(clip_g, clip_l)
            w_global = torch.softmax(w_global, dim=0).unsqueeze(1)
        clip_embed = torch.nn.functional.normalize(clip_g*w_global + clip_l*(1-w_global), p=2, dim=-1)

    elif embed_type == "adaptive_weights":
        w_local = (cos(clip_seg, clip_bbox)*w_masked).unsqueeze(-1) #
        clip_l = torch.nn.functional.normalize(clip_seg*w_local+clip_bbox*(1-w_local), p=2, dim=-1)

        w_global =  (cos(clip_g, clip_l)*w_global).unsqueeze(-1)# start with 0.18
        clip_embed = torch.nn.functional.normalize(clip_g*w_global + clip_l*(1-w_global), p=2, dim=-1)
    else:
        # vanilla
        clip_embed = clip_seg
    return clip_embed


def load_clip_model(model_card: str, use_half: bool) -> Tuple[Any, Any, Compose, str]:

    cards = {
        "SigLIP": 'hf-hub:timm/ViT-SO400M-14-SigLIP',#224x224
        "SigLIP-384": 'hf-hub:timm/ViT-SO400M-14-SigLIP-384',#384x384
        "SigLIP2-384": 'hf-hub:timm/ViT-SO400M-16-SigLIP2-384',#384x384
        "ViT-H-14": 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K',#224x224
        "ViT-B-16-qg": 'hf-hub:apple/DFN2B-CLIP-ViT-B-16',#224x224
        "ViT-L-14-qg": 'hf-hub:apple/DFN2B-CLIP-ViT-L-14-39B',#224x224
        "ViT-H-14-qg": 'hf-hub:apple/DFN5B-CLIP-ViT-H-14', #224x224
        "ViT-H-14-378qg": 'hf-hub:apple/DFN5B-CLIP-ViT-H-14-378'#384x384
    }
    clip_dim_cards = {
        "SigLIP": 1152,
        "SigLIP-384": 1152,
        "SigLIP2-384": 1152,
        "ViT-H-14": 1024,
        "ViT-B-16-qg": 512,
        "ViT-L-14-qg": 768,
        "ViT-H-14-qg": 1024, 
        "ViT-H-14-378qg": 124
    }
    assert model_card in list(cards.keys()), f"Select one of {cards.keys()} model cards"
    model, preprocess = open_clip.create_model_from_pretrained(
        cards[model_card],
        precision="fp32" if not use_half else "fp16")
    
    tokenizer = open_clip.get_tokenizer(cards[model_card])

    tf_to_keep = [tf for tf in preprocess.transforms if isinstance(tf, Resize) or isinstance(tf,CenterCrop) or isinstance(tf, Normalize)]
    preprocess = Compose(tf_to_keep)

    return model.eval(), tokenizer, preprocess, clip_dim_cards[model_card]
