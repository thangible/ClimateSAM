import torch
import copy
from torch import nn
from model.image_encoder import ClimateSAMImageEncoder
from model.prompt_encoder import PromptEncoderWrapper
from model.mask_decoder import MaskDecoderHQ
from model.segment_anything_ext.build_sam import sam_model_registry
from typing import Union, List, Tuple


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

    def __init__(self, model_type: str, input_weights: List[float] = None):
        super(ClimateSAM, self).__init__()
        assert model_type in ['vit_b', 'vit_l', 'vit_h'], f"invalid model_type: {model_type}!"
        
        # ORI SAM model
        self.ori_sam = sam_model_registry[model_type](sam_ckpt_path_dict[model_type])
        del self.ori_sam.mask_decoder # remove the mask decoder in original SAM to avoid redundant params in model object
        self.sam_img_size = (self.ori_sam.image_encoder.img_size, self.ori_sam.image_encoder.img_size)

        # ClimateSAM model
        self.input_adapt = nn.Conv2d(17, 3, kernel_size=1, stride=1, padding=0)
        self.image_encoder = ClimateSAMImageEncoder(ori_sam=self.ori_sam, fix=True)
        self.prompt_encoder = PromptEncoderWrapper(ori_sam=self.ori_sam, fix=True)
        self.mask_decoder = MaskDecoderHQ(
            model_type, self.ori_sam.mask_decoder.state_dict()
        )
        
        #set weights for input adaptation:
        with torch.no_grad():
            # Zero out all weights
                self.input_adapt.weight.zero_()
                
                # Define the channels where you want high weights
                if input_weight is None:
                    input_weight = [0, 5, 7]
                
                # For instance, set those weights to 1.0 for every output channel
                for out_ch in range(self.input_adapt.weight.shape[0]):
                    for in_ch in high_channels:
                        self.input_adapt.weight[out_ch, in_ch, 0, 0] = 1.0    
        
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
                
                
    def forward(
            self,
            input: Union[List[torch.Tensor], None],
            hq_token_weight: torch.Tensor = None,
            return_all_hq_masks: bool = False
    ):
        img = self.input_adapt(input)
        image_embeddings, interm_embeddings = self.image_encoder(img)
        
        
        

    # def preprocess(self, imgs):
    #     ori_img_size = [(imgs[i].shape[-2], imgs[i].shape[-1]) for i in range(len(imgs))]
    #     imgs_return = copy.deepcopy(imgs)
    #     for i in range(len(ori_img_size)):
    #         # skip the one with the same size as SAM input
    #         if ori_img_size[i] == self.sam_img_size:
    #             continue

    #         if imgs_return is not None:
    #             # bilinear will produce non-deterministic gradients during training. For exact reproduction, please
    #             # change the mode from bilinear to nearest
    #             imgs_return[i] = F.interpolate(
    #                 imgs_return[i], self.sam_img_size, mode="bilinear", align_corners=False,
    #             )
    #             # Normalize colors to match the original SAM preprocessing
    #             imgs_return[i] = (imgs_return[i] - self.ori_sam.pixel_mean) / self.ori_sam.pixel_std

    #         h_scale = self.sam_img_size[0] / ori_img_size[i][0]
    #         w_scale = self.sam_img_size[1] / ori_img_size[i][1]
        
        
        
        
