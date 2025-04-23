import torch
import copy
from torch import nn
from model.image_encoder import ClimateSAMImageEncoder
from model.prompt_encoder import PromptEncoderWrapper
from model.mask_decoder import MaskDecoderHQ
from model.prompt_generator import PromptGenerator
from model.segment_anything_ext.build_sam import sam_model_registry
from typing import Union, List, Tuple
import torch.nn.functional as F


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
        
        self.sam_img_size = (self.ori_sam.image_encoder.img_size, self.ori_sam.image_encoder.img_size)

        # ClimateSAM model
        self.input_adapt = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0),
            nn.Upsample(size=self.sam_img_size, mode='bilinear', align_corners=False)
        )
        
        self.mask_decoder = MaskDecoderHQ(
            model_type, self.ori_sam.mask_decoder.state_dict()
        )
        self.image_encoder = ClimateSAMImageEncoder(ori_sam=self.ori_sam, fix=True, hq_token=self.mask_decoder.hf_token.weight)
        self.prompt_generator = PromptGenerator()
        self.prompt_encoder = PromptEncoderWrapper(ori_sam=self.ori_sam, fix=True)
        
        #set weights for input adaptation:
        # Zero out all weights 
        # Define the channels where you want high weights
        with torch.no_grad():
            self.input_adapt[0].weight.zero_()  
            if input_weights is None:
                input_weights = [0, 1, 2] # for 'TMQ', 'U850', 'V850'
            # For instance, set those weights to 1.0 for every output channel
            for out_ch in range(self.input_adapt[0].weight.shape[0]):
                for in_ch in input_weights:
                    self.input_adapt[0].weight[out_ch, in_ch, 0, 0] = 1.0    
                
        del self.ori_sam.mask_decoder # remove the mask decoder in original SAM to avoid redundant params in model object
        
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
        imgs = self.input_adapt(input) # from 16x768x768 to 3x1024x1024
        ori_img_size = [(imgs[i].shape[-2], imgs[i].shape[-1]) for i in range(len(imgs))]
        # Normalize colors to match the original SAM preprocessing
        pixel_mean = self.ori_sam.pixel_mean.clone().detach().to(imgs.device).view(1, 3, 1, 1)
        pixel_std  = self.ori_sam.pixel_std.clone().detach().to(imgs.device).view(1, 3, 1, 1)
        imgs = (imgs - pixel_mean) / pixel_std
        
        # encode the images
        image_embeddings, interm_embeddings = self.image_encoder(imgs) # shape batch x [256, 64, 64] and 12 x torch.Size([batch, 64, 64, 768])
        batch_size = len(image_embeddings)
        # Print the shapes of the embeddings for debugging
        print(f"Image embeddings shape: {image_embeddings[0].shape}")
        print(f"Intermediate embeddings shape: {interm_embeddings[0].shape}")

        masks = self.prompt_generator(interm_embeddings) # shape: batch x 2 x 256 x 256
        tc_masks, ar_masks = torch.chunk(masks, 2, dim=1) # shape: batch x 1 x 256 x 256 each


        print(f"TC masks shape: {tc_masks.shape}")
        print(f"AR masks shape: {ar_masks.shape}")
        
        tc_sparse_embeddings, tc_dense_embeddings = self.prompt_encoder(masks=tc_masks)
        ar_sparse_embeddings, ar_dense_embeddings = self.prompt_encoder(masks=ar_masks)
        print(f"TC mask embedding shape: {tc_dense_embeddings.shape}")
        print(f"AR mask embedding shape: {ar_dense_embeddings.shape}")
        print(f"TC sparse embedding shape: {tc_sparse_embeddings.shape}")
        print(f"AR sparse embedding shape: {ar_sparse_embeddings.shape}")

        _, tc_pred_masks = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=[self.prompt_encoder.get_dense_pe() for _ in range(batch_size)],
            sparse_prompt_embeddings=tc_sparse_embeddings.unsqueeze(1),
            dense_prompt_embeddings=tc_dense_embeddings,
            multimask_output=False,
            interm_embeddings=interm_embeddings,
            hq_token_weight=hq_token_weight,
            return_all_hq_masks=return_all_hq_masks
        )
        
        _, ar_pred_masks = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=[self.prompt_encoder.get_dense_pe() for _ in range(batch_size)],
            sparse_prompt_embeddings=ar_sparse_embeddings.unsqueeze(1),
            dense_prompt_embeddings= ar_dense_embeddings,
            multimask_output=False,
            interm_embeddings=interm_embeddings,
            hq_token_weight=hq_token_weight,
            return_all_hq_masks=return_all_hq_masks
        )
        
        print(f"TC predicted masks shape: {tc_pred_masks[0].shape}")
        print(f"Length of TC predicted masks list: {len(tc_pred_masks)}")
        print(f"AR predicted masks shape: {ar_pred_masks[0].shape}")
        print(f"Length of AR predicted masks list: {len(ar_pred_masks)}")
        
        # rescale the mask size back to original image size
        tc_postprocess_masks_hq = [m_hq.clone() for m_hq in tc_pred_masks]
        for i in range(len(tc_postprocess_masks_hq)):
            tc_postprocess_masks_hq[i] = self.postprocess(output_masks=tc_postprocess_masks_hq[i], ori_img_size=ori_img_size[i])
        
        ar_postprocess_masks_hq = [m_hq.clone() for m_hq in ar_pred_masks]
        for i in range(len(ar_postprocess_masks_hq)):
            ar_postprocess_masks_hq[i] = self.postprocess(output_masks=ar_postprocess_masks_hq[i], ori_img_size=ori_img_size[i])
        
        # Print the shapes of the postprocessed masks for debugging
        for i, (tc_mask, ar_mask) in enumerate(zip(tc_postprocess_masks_hq, ar_postprocess_masks_hq)):
            print(f"Postprocessed TC mask {i} shape: {tc_mask.shape}")
            print(f"Postprocessed AR mask {i} shape: {ar_mask.shape}")
            
        return tc_postprocess_masks_hq, ar_postprocess_masks_hq
    
    @staticmethod
    def postprocess(output_masks: torch.Tensor, ori_img_size: Tuple):
        # rescale the mask size back to original image size
        output_mask_size = (output_masks.size(-2), output_masks.size(-1))
        if output_mask_size != ori_img_size:
            if len(output_masks.shape) == 3:
                output_masks = output_masks.unsqueeze(1)
            # bilinear will produce non-deterministic gradients during training. For exact reproduction, please
            # change the mode from bilinear to nearest
            output_masks = F.interpolate(
                output_masks, ori_img_size, mode="bilinear", align_corners=False,
            )
        return output_masks

        
        
        