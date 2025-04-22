import torch
from torch import nn
from model.image_encoder import ClimateSAMEncoder
from model.prompt_encoder import PromptEncoderWrapper
from model.mask_decoder import MaskDecoderHQ
from model.segment_anything_ext.build_sam import sam_model_registry


sam_ckpt_path_dict = dict(
    vit_b='./pretrained/sam_vit_b_01ec64.pth',
    vit_l='./pretrained/sam_vit_l_0b3195.pth',
    vit_h='./pretrained/sam_vit_h_4b8939.pth'
)

class ClimateSAM(nn.Module):
    """
    ClimateSAM is a class that implements a modified version of the Segment Anything Model (SAM) for climate data.
    It includes a Vision Transformer (ViT) encoder and a mask decoder, with additional features for climate data processing.
    """

    def __init__(self, model_type: str):
        super(ClimateSAM, self).__init__()
        assert model_type in ['vit_b', 'vit_l', 'vit_h'], f"invalid model_type: {model_type}!"
        
        # ORI SAM model
        self.ori_sam = sam_model_registry[model_type](sam_ckpt_path_dict[model_type])
        del self.ori_sam.mask_decoder # remove the mask decoder in original SAM to avoid redundant params in model object
        self.sam_img_size = (self.ori_sam.image_encoder.img_size, self.ori_sam.image_encoder.img_size)

        # ClimateSAM model
        self.image_encoder = ClimateSAMEncoder(ori_sam=self.ori_sam, fix=True)
        self.prompt_encoder = PromptEncoderWrapper(ori_sam=self.ori_sam, fix=True)
        self.mask_decoder = MaskDecoderHQ(
            model_type, self.ori_sam.mask_decoder.state_dict()
        )
        
    def train(self, mode: bool = True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training: only turn the image encoder to train mode
            for n, c in self.named_children():
                if n != 'image_encoder':
                    c.eval()
                else:
                    c.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)
        
        
        
