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
