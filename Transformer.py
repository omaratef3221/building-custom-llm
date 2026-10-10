import torch
import torch.nn as nn 

from Encoder import EncoderBlock
from Decoder import DecoderBlock


class Transformer(nn.Module):
    def __init__(self, D_MODEL, H):
        super(Transformer, self).__init__()
        self.encoder = EncoderBlock(D_MODEL, H)
        self.decoder = DecoderBlock(D_MODEL, H)

    def forward(x, src_mask, tgt_mask):
        pass