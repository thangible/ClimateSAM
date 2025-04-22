import torch
import torch.nn as nn

class PromptEncoderWrapper(nn.Module):

    def __init__(self, ori_sam, fix: bool = True):
        super(PromptEncoderWrapper, self).__init__()
        self.sam_prompt_encoder = ori_sam.prompt_encoder
        if fix:
            for name, param in self.sam_prompt_encoder.named_parameters():
                param.requires_grad = False

    def forward(self, points=None, boxes=None, masks=None):
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points, boxes, masks)
        return sparse_embeddings, dense_embeddings

    def get_dense_pe(self):
        return self.sam_prompt_encoder.get_dense_pe()