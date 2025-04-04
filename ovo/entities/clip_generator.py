from typing import Dict, List
import torch
import yaml
import os

from ..utils import clip_utils
from .clips_merging import WeightsPredictorMerger

class CLIPGenerator:
    def __init__(self, config: Dict, device: str = "cuda"):
        self.config = config
        self.device = device
        self.embed_type = config.get("embed_type", "vanilla")
        if self.embed_type == "learned":

            with open(os.path.join(config["weights_predictor_path"], "hparams.yaml"), "r") as f:
                model_config = yaml.full_load(f)
            pt_path = os.path.join(config["weights_predictor_path"], "model.pt")

            model_weights = torch.load(pt_path, weights_only=False)
            self.clips_fusion_model = WeightsPredictorMerger(model_config["model"]).eval()
            self.clips_fusion_model.load_state_dict(model_weights)

            self.clips_fusion = lambda clip_g, clip_seg, clip_bbox: self.clips_fusion_model(torch.cat([clip_g[:,None], clip_seg[:,None],clip_bbox[:,None]], dim=1))
            if config.get("use_half", False):
                self.clips_fusion_model.half()
        else:
            w_masked = config.get("w_masked", 0.4418)
            w_global = config.get("w_global", 0.1)
            self.clips_fusion = lambda clip_g, clip_seg, clip_bbox: clip_utils.fuse_clips(clip_g, clip_seg, clip_bbox, self.embed_type, w_masked, w_global)

        self.model_card = config.get("model_card", "SigLIP-384")
        self.model, self.tokenizer, self.preprocess, clip_dim = clip_utils.load_clip_model(self.model_card, config.get("use_half", False))
        self.clip_dim=clip_dim        
        
        if self.model_card[:6] == "SigLIP":
            self.get_similarity = clip_utils.siglip_cosine_similarity

            logit_scale = config.get("logit_scale")
            if logit_scale:
                logit_scale = torch.Tensor([logit_scale]).to(self.device)
            else:
                logit_scale = self.model.logit_scale

            logit_bias = config.get("logit_bias")
            if logit_bias:
                logit_bias = torch.Tensor([logit_bias]).to(self.device)
            else:
                logit_bias = self.model.logit_bias

            self.similarity_args = (logit_scale, logit_bias)
        else:
            self.get_similarity = clip_utils.clip_cosine_similarity
            self.similarity_args = ()  
        self.to(self.device)

    @property
    def get_clip_dim(self) -> str:
        return self.clip_dim
            
    def to(self, device: str) -> None:
        """
        Move predictor model to either 'cpu' or 'cuda' device.
        Args:
            device (str): device to mode the model to.
        """
        if "cuda" in device:
            return self.cuda()
        else:
            return self.cpu()

    def cpu(self) -> None:
        """
        Move predictor model to cpu device.
        """
        self.device = "cpu"
        self.model.cpu()
        self.similarity_args = [x.cpu() for x in self.similarity_args]
        if self.embed_type == "learned":
            self.clips_fusion_model.cpu()
        

    def cuda(self) -> None:
        """
        Move predictor model to cuda default device.
        """
        self.device = "cuda"
        self.model.cuda()
        self.similarity_args = [x.cuda() for x in self.similarity_args]
        if self.embed_type == "learned":
            self.clips_fusion_model.cuda()

    @torch.no_grad
    def encode_image(self, input: torch.Tensor) -> torch.Tensor:
        """ Compute CLIP descriptor of an RGB image
        Args:
            - input (torch.Tensor): RGB image as tensor with shape (H,W,3) in range [0,1]
        Return:
            - clip_descriptor (torch.Tensor): as tensor with shape (self.clip_dim)
        """
        processed_input = self.preprocess(input)
        if self.config.get("use_half", False):
            processed_input = processed_input.half()
        return self.model.encode_image(processed_input)    

    @torch.no_grad
    def extract_clip(self, image: torch.Tensor, seg_images: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        """ Computes a CLIP vector for each mask of the segmented image.
        Args:
            - image (torch.Tensor): Full source RGB image with dimensions (H,W,3) and range 0-1.
            - seg_images (torch.Tensor): array of shape (N,6,h,w), with h < H and w < W. The first 3 channels of the second dimension store the segment with black background of a 2D instance, while the last 3 channels store the image of the minimum bounding box arround that 2D semgent with background.
            - return_all: if True returns the three computed descriptors of each image in seg_images instead of merging them.
        Return:
            - climp_embeds: list of numpy arrays with dim (N, self.clip_dim).        
        """
        if self.embed_type != "vanilla":
            if len(image.shape) ==3:
                image = image[None, ...]
            clip_g = torch.nn.functional.normalize(self.encode_image(image), p=2,dim=-1)

        if len(seg_images) == 0:
            return torch.tensor([], device = self.device)
        
        if self.embed_type == "vanilla":
            clip_embed = torch.nn.functional.normalize(self.encode_image(seg_images[:,:3]), p=2,dim=-1)
        else:
            n_clips = seg_images.shape[0]
            seg_images = torch.vstack([seg_images[:,:3], seg_images[:,3:]])
            clip_seg = torch.nn.functional.normalize(self.encode_image(seg_images), p=2,dim=-1)

            if return_all:
                clip_embed = torch.cat([clip_g.repeat(n_clips,1)[:,None], clip_seg[:n_clips][:,None], clip_seg[n_clips:][:,None] ],dim=1)
            else:
                clip_embed = self.clips_fusion(clip_g.repeat(n_clips,1), clip_seg[:n_clips], clip_seg[n_clips:,])
            
        if self.config.get("use_half", False):
            clip_embed = clip_embed.half()
        return clip_embed

    @torch.no_grad
    def get_txt_embedding(self, text_list: List[str]) -> torch.Tensor:
        """
        Compute text embeddings for a list of strings. Each element of the list is tokenized individually
        Args:
            - text_list (List[str]): A list of strings to be embedded.
        Returns:
            - embeds (torch.Tensor): A tensor containing the normalized embeddings for the input phrases.
        """

        tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in text_list]).to(self.device)
        embeds = self.model.encode_text(tok_phrases)
        embeds /= embeds.norm(dim=-1, keepdim=True)
        return embeds

    @torch.no_grad
    def get_embed_txt_similarity(self, ins_descriptors: torch.Tensor, txt_queries: List[str], templates: str | List[str] = ['{}']) -> torch.Tensor:
        """
        Computes independently the similarity between image embeddings and text queries.
        Args:
            - ins_descriptors (torch.Tensor): A tensor containing image embeddings.
            - txt_queries (List[str]): A list of text queries.
            - templates (str | List[str], optional): A template or a list of templates to use for classification. If it's a list, the classes embeddings will be an ensembles of the templates. 
        Returns:
            - sim_map (torch.Tensor): A tensor containing the similarity scores between each text query and the image embeddings.
        """

        n_queries = len(txt_queries)
        txt_embeds = torch.zeros((n_queries, ins_descriptors.shape[1]), device = ins_descriptors.device)
        if isinstance(templates, str):
            templates = [templates]
        queries = [[template.format(query) for template in templates] for query in txt_queries]

        for j in range(n_queries):
            # Compute the text embedding of each query individually to make them independent from other queries
            embed = self.get_txt_embedding(queries[j]).mean(0, keepdim=True)
            txt_embeds[j] = torch.nn.functional.normalize(embed, p=2, dim=-1)

        sim_map = self.get_similarity(txt_embeds, ins_descriptors, *self.similarity_args)
        return sim_map

