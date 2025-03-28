import torch.nn.functional as F
import torch.nn as nn
import torch


ACTIVATION_DICT = {
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "sigmoid": nn.Sigmoid
}

def block_mlp(i_dim, h_dim, o_dim, n_layers, act_key="leaky_relu"):
    activation = ACTIVATION_DICT[act_key]
    mlp_layers = [
        nn.Linear(i_dim, h_dim),
        activation(),
        ]
    for i in range(n_layers):
        mlp_layers.append(nn.Linear(h_dim, h_dim))
        mlp_layers.append(activation())
    mlp_layers.append(nn.Linear(h_dim, o_dim))

    return nn.Sequential(*mlp_layers)

class WeightsPredictorMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            batch_first= True,
            d_model=config["transformer"]["d_model"],
            nhead=config["transformer"].get("nhead",8),
            dim_feedforward=config["transformer"]["dim_feedforward"],
            dropout=0.1,
            activation='relu')
        self.att_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["transformer"]["n_layers"])       
        self.mlp = block_mlp(**config["mlp"])

    def forward(self, input_clips: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_clips (torch.Tensor): shape (B, N_clips, 1152)
        Return:
            merged clip (torch.Tensor): shape (B, 1152)
        """
        b, n_clips, clips_dim = input_clips.shape
        x = self.att_encoder(input_clips)
        weights = self.mlp(x.flatten(-2, -1))
        if weights.shape[-1] != 3:
            weights = weights.reshape(b, n_clips, clips_dim)
            weights = F.softmax(weights, dim=-2)
        else:
            weights = F.softmax(weights, dim=-1).unsqueeze(-1)
        clips = (input_clips*weights).sum(-2)
        clips = F.normalize(clips, dim=-1)
        return clips
