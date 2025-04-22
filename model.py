import torch
import copy
from torch import nn
from model.image_encoder import ClimateSAMImageEncoder
from model.prompt_encoder import PromptEncoderWrapper
from model.mask_decoder import MaskDecoderHQ
from model.prompt_generator import PromptGenerator
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
        self.input_adapt = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0),
            nn.Upsample(size=self.sam_img_size, mode='bilinear', align_corners=False)
        )
        
        self.image_encoder = ClimateSAMImageEncoder(ori_sam=self.ori_sam, fix=True)
        self.prompt_generator = PromptGenerator()
        self.prompt_encoder = PromptEncoderWrapper(ori_sam=self.ori_sam, fix=True)
        self.mask_decoder = MaskDecoderHQ(
            model_type, self.ori_sam.mask_decoder.state_dict()
        )
        
        #set weights for input adaptation:
        # Zero out all weights 
        # Define the channels where you want high weights
        self.input_adapt.weight.zero_()  
        if input_weight is None:
            input_weight = [0, 1, 2] # for 'TMQ', 'U850', 'V850'
        # For instance, set those weights to 1.0 for every output channel
        for out_ch in range(self.input_adapt.weight.shape[0]):
            for in_ch in input_weight:
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
        mask = self.prompt_generator(interm_embeddings)
        tc_mask, ar_mask = mask[:, 0:1, :, :].squeeze(), mask[:, 1:2, :, :].squeeze()
        
        batch_size = len(image_embeddings)
        tc_mask_embeded = self.prompt_encoder(mask=tc_mask)
        ar_mask_embeded = self.prompt_encoder(mask=ar_mask)
        
        # _, _, masks = self.convert_raw_prompts_to_triple(
        #     point_coords=None, 
        #     point_labels=None,
        #     box_coords=None, 
        #     noisy_masks=mask, 
        #     batch_size=batch_size
        # )
        
        
    # @staticmethod
    # def convert_mask_to_triple(masks):
    #     for i in range(len(masks)):
    #         masks_idx = None
    #         if masks[i] is not None:
    #             masks_idx = masks[i]
    #             if len(masks_idx.shape) == 2:
    #                 masks_idx = masks_idx[None, None, :, :]
    #             if len(masks_idx.shape) == 3:
    #                 masks_idx = masks_idx[None, :, :]
    #             if len(masks_idx.shape) != 4:
    #                 raise RuntimeError(
    #                     "Each mask in the list must be in the shape of (N, 1, 256, 256) "
    #                     "where N is the number of output masks!"
    #                 )
    #             if masks_idx.size(1) != 1:
    #                 raise RuntimeError("Please only give one mask for each output!")
    #             if masks_idx.size(-2) != 256 or masks_idx.size(-1) != 256:
    #                 raise RuntimeError("Each mask must have width and height of 256!")
    #         masks[i] = masks_idx
                
        

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
        
        
        
        
