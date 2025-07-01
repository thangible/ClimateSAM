import torch
import copy
from torch import nn
from model.image_encoder import ClimateSAMImageEncoder
from model.prompt_encoder import PromptEncoderWrapper
from model.mask_decoder import MaskDecoderHQ
from model.prompt_generator import PromptGenerator
from model.segment_anything_ext.build_sam import sam_model_registry
from typing import Union, List, Tuple, Optional
import torch.nn.functional as F
# from climatesam_util import extract_point_and_bbox_prompts_from_climatenet_mask
import numpy as np

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

    def __init__(self, model_type: str, input_weights: List[float] = None, verbose = False, use_prompt_generator = False, mlp_ratio = 0.25):
        super(ClimateSAM, self).__init__()
        assert model_type in ['vit_b', 'vit_l', 'vit_h'], f"invalid model_type: {model_type}!"
        self.verbose = verbose
        self.use_prompt_generator = use_prompt_generator
        # ORI SAM model
        self.ori_sam = sam_model_registry[model_type](sam_ckpt_path_dict[model_type])
        self.sam_img_size = (self.ori_sam.image_encoder.img_size, self.ori_sam.image_encoder.img_size)
        # ClimateSAM model
        self.input_adapt = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0),
        )
        
        self.mask_decoder = MaskDecoderHQ(
            model_type, self.ori_sam.mask_decoder.state_dict()
        )
        self.image_encoder = ClimateSAMImageEncoder(ori_sam=self.ori_sam, fix=True, hq_token=self.mask_decoder.hf_token.weight, mlp_ratio=mlp_ratio)
        if self.use_prompt_generator:
          self.prompt_generator = PromptGenerator(in_channels = self.image_encoder.sam_img_encoder.num_features)
        self.prompt_encoder = PromptEncoderWrapper(ori_sam=self.ori_sam, fix=True)
        
        #set weights for input adaptation:
        # Zero out all weights 
        # Define the channels where you want high weights
        with torch.no_grad():
            self.input_adapt[0].weight.zero_()  
            if input_weights is None:
                # Default input_weights correspond to indices of specific climate variables:
                # 'TMQ' (Total Precipitable Water Vapor), 'U850' (Zonal Wind at 850 hPa), 
                # and 'V850' (Meridional Wind at 850 hPa).
                input_weights = [0, 1, 2] # for 'TMQ', 'U850', 'V850'
            # For instance, set those weights to 1.0 for every output channel
            for out_ch in range(self.input_adapt[0].weight.shape[0]):
                for in_ch in input_weights:
                    self.input_adapt[0].weight[out_ch, in_ch, 0, 0] = 1.0    
        self.input_adapt[0].weight.requires_grad = False # freeze the input adaptation layer
                
        del self.ori_sam.mask_decoder # remove the mask decoder in original SAM to avoid redundant params in model object
        
    def train(self, mode: bool = True, phase: int = 1, verbose = False):
        # Set the global train/eval mode
        super().train(mode)

        # Freeze all parameters by default (phase-specific unfreezing follows)
        for param in self.parameters():
            param.requires_grad = False

        # Phase-specific training configurations
        if phase == 1:
            for n, c in self.named_children():
                if n not in ['image_encoder', 'mask_decoder']:
                    c.eval()
                else:
                    c.train(mode=mode)
            if verbose:
                print("Training image_encoder")

        elif phase == 2:
            # Phase 2: Train only prompt_encoder
            self.enable_prompt_generator()
            for n, c in self.named_children():
                if n not in ['prompt_generator']:
                    c.eval()
                else:
                    c.train(mode=mode)
            if verbose:
                print("Training prompt_encoder and prompt_generator")
            
            
        elif phase == 3:
            # Phase 3: Train only input_adapt
            self.enable_prompt_generator()
            for n, c in self.named_children():
                if n not in 'input_adapt':
                    c.eval()
                else:
                    c.train(mode = mode)
            if verbose:
                print("Training input_adapt")
            
        if verbose:
                  
            # Verify frozen/trainable parameters (for debugging)
            for n, c in self.named_children():
                total_params = sum(p.numel() for p in c.parameters())
                trainable_params = sum(p.numel() for p in c.parameters() if p.requires_grad)        
                print(f"{n.upper():<20} | Trainable: {str(c.training):<5} | Trainable params: {trainable_params:>9,}/{total_params:>12,} ({100*trainable_params/total_params:>5.2f}%)")
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Phase {phase}: Trainable params = {trainable_params}/{total_params} "
                f"({100*trainable_params/total_params:.2f}%)")
                    
                
    def forward(
            self,
            input: Union[List[torch.Tensor], None],
            hq_token_weight: torch.Tensor = None,
            return_all_hq_masks: bool = False,
            ar_point_prompts: List[Union[torch.Tensor, None]] = None,
            tc_point_prompts: List[Union[torch.Tensor, None]] = None,
            ar_bbox_prompts: List[Union[torch.Tensor, None]] = None,
            tc_bbox_prompts: List[Union[torch.Tensor, None]] = None,
            ar_mask_prompts: List[Union[torch.Tensor, None]] = None,
            tc_mask_prompts: List[Union[torch.Tensor, None]] = None
    ):
        ori_img_size = [(input[i].shape[-2], input[i].shape[-1]) for i in range(len(input))]
        input = self.interpolate_input(input) # from 16x768x1152 to 16x1024x1024
        
        # print(f"Input shape after interpolation: {input.shape}")
        imgs = input[:, :3, :, :] # from 16x1024x1024 to 3x1024x1024
        # imgs = self.input_adapt(input) # from 16x1024x1024 to 3x1024x1024
        
        # print(f"Input to imgs after adapt: {imgs.shape}")
        imgs = self.preprocess_images(imgs) # normalize the input images
        # print(f"Shape of imgs after preprocessing: {imgs.shape}")
        # encode the images
        image_input = imgs.clone().detach()
        image_embeddings, interm_embeddings = self.image_encoder(imgs) # shape batch x [256, 64, 64] and 12 x torch.Size([batch, 64, 64, 768])
        batch_size = len(image_embeddings)
        # Print the shapes of the embeddings for debugging
        # print(f"Image embeddings shape: {image_embeddings[0].shape}")
        # print(f"Intermediate embeddings shape: {interm_embeddings[0].shape}")

        ar_point_prompts, tc_point_prompts, ar_bbox_prompts, tc_bbox_prompts = self.preprocess_prompts(
            ar_point_prompts=ar_point_prompts,
            tc_point_prompts=tc_point_prompts,
            ar_bbox_prompts=ar_bbox_prompts,
            tc_bbox_prompts=tc_bbox_prompts,
            ori_img_size=ori_img_size
        )
        
        if self.use_prompt_generator:
            tc_masks, ar_masks = self.prompt_generator(interm_embeddings) # shape: batch x 2 x 256 x 256
            ar_point_prompts, tc_point_prompts, ar_bbox_prompts, tc_bbox_prompts = None, None, None, None
        # tc_masks, ar_masks = torch.chunk(masks, 2, dim=1) # shape: batch x 1 x 256 x 256 each

        # print(f"TC masks shape: {tc_masks.shape}")
        # print(f"AR masks shape: {ar_masks.shape}")
            
        tc_sparse_embeddings, tc_dense_embeddings = [], []
        ar_sparse_embeddings, ar_dense_embeddings = [], []
        for batch_idx in range(batch_size):
            # print(f"ar_point_prompts shape: {ar_point_prompts[batch_idx][0].shape}")
            # print(f"tc_point_prompts shape: {tc_point_prompts[batch_idx][0].shape}")
            # print(f"ar_point_prompts label shape: {ar_point_prompts[batch_idx][1].shape}")
            # print(f"tc_point_prompts label shape: {tc_point_prompts[batch_idx][1].shape}")
            # print(f"ar_bbox_prompts shape: {ar_bbox_prompts[batch_idx].shape}")
            # print(f"tc_bbox_prompts shape: {tc_bbox_prompts[batch_idx].shape}")
            current_tc_sparse_embedding, current_tc_dense_embeddings = self.prompt_encoder(
                points=tc_point_prompts[batch_idx] if tc_point_prompts is not None else None,
                boxes=tc_bbox_prompts[batch_idx] if tc_bbox_prompts is not None else None,
                masks= tc_mask_prompts[batch_idx] if tc_mask_prompts is not None else None,
            )
            
            current_ar_sparse_embedding, current_ar_dense_embeddings = self.prompt_encoder(
                points=ar_point_prompts[batch_idx] if ar_point_prompts is not None else None,
                boxes=ar_bbox_prompts[batch_idx] if ar_bbox_prompts is not None else None,
                masks= ar_mask_prompts[batch_idx] if ar_mask_prompts is not None else None,
            )
            
            tc_sparse_embeddings.append(current_tc_sparse_embedding)
            ar_sparse_embeddings.append(current_ar_sparse_embedding)
            tc_dense_embeddings.append(current_tc_dense_embeddings)
            ar_dense_embeddings.append(current_ar_dense_embeddings)

        
        # print(f"TC mask embedding shape: {tc_dense_embeddings.shape}")
        # print(f"AR mask embedding shape: {ar_dense_embeddings.shape}")
        # print(f"TC sparse embedding shape: {tc_sparse_embeddings.shape}")
        # print(f"AR sparse embedding shape: {ar_sparse_embeddings.shape}")

        _, tc_pred_masks = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=[self.prompt_encoder.get_dense_pe() for _ in range(batch_size)],
            sparse_prompt_embeddings=tc_sparse_embeddings,
            dense_prompt_embeddings=tc_dense_embeddings,
            multimask_output=False,
            interm_embeddings=interm_embeddings,
            hq_token_weight=hq_token_weight,
            return_all_hq_masks=return_all_hq_masks
        )
        
        _, ar_pred_masks = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=[self.prompt_encoder.get_dense_pe() for _ in range(batch_size)],
            sparse_prompt_embeddings=ar_sparse_embeddings,
            dense_prompt_embeddings= ar_dense_embeddings,
            multimask_output=False,
            interm_embeddings=interm_embeddings,
            hq_token_weight=hq_token_weight,
            return_all_hq_masks=return_all_hq_masks
        )
        
        # print(f"TC predicted masks shape: {tc_pred_masks[0].shape}")
        # print(f"Length of TC predicted masks list: {len(tc_pred_masks)}")
        # print(f"AR predicted masks shape: {ar_pred_masks[0].shape}")
        # print(f"Length of AR predicted masks list: {len(ar_pred_masks)}")
        
        # rescale the mask size back to original image size
        tc_postprocess_masks_hq = [m_hq.clone() for m_hq in tc_pred_masks]
        for i in range(len(tc_postprocess_masks_hq)):
            tc_postprocess_masks_hq[i] = self.postprocess(output_masks=tc_postprocess_masks_hq[i], ori_img_size=ori_img_size[i])
        
        ar_postprocess_masks_hq = [m_hq.clone() for m_hq in ar_pred_masks]
        for i in range(len(ar_postprocess_masks_hq)):
            ar_postprocess_masks_hq[i] = self.postprocess(output_masks=ar_postprocess_masks_hq[i], ori_img_size=ori_img_size[i])
        
        # Print the shapes of the postprocessed masks for debugging
        # for i, (tc_mask, ar_mask) in enumerate(zip(tc_postprocess_masks_hq, ar_postprocess_masks_hq)):
        #     print(f"Postprocessed TC mask {i} shape: {tc_mask.shape}")
        #     print(f"Postprocessed AR mask {i} shape: {ar_mask.shape}")
        
        if not self.training:
            tc_postprocess_masks_hq = self.assemble_raw_masks(tc_postprocess_masks_hq) 
            ar_postprocess_masks_hq = self.assemble_raw_masks(ar_postprocess_masks_hq)
        return tc_postprocess_masks_hq, ar_postprocess_masks_hq, image_input
    
    def enable_prompt_generator(self):
        if not hasattr(self, 'prompt_generator'):
            self.prompt_generator = PromptGenerator(
                in_channels=self.image_encoder.sam_img_encoder.num_features
            )
        self.use_prompt_generator = True

    def disable_prompt_generator(self):
        if hasattr(self, 'prompt_generator'):
            del self.prompt_generator
        self.use_prompt_generator = False
    
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

    def interpolate_input(self, input: torch.Tensor):
        # Check if input size matches self.sam_img_size
        if input.shape[-2:] != self.sam_img_size:
            input = F.interpolate(input, size=self.sam_img_size, mode='bilinear', align_corners=False)
            
        # print(f"Input shape after preprocessing: {input.shape}")
        
        return input
    
    def preprocess_images(self, input: Union[List[torch.Tensor], None]):
        # Normalize colors to match the original SAM preprocessing
        pixel_mean = self.ori_sam.pixel_mean.clone().detach().to(input.device).view(1, 3, 1, 1)
        pixel_std  = self.ori_sam.pixel_std.clone().detach().to(input.device).view(1, 3, 1, 1)
        input = (input - pixel_mean) / pixel_std
        return input
        
    def preprocess_prompts(self, ar_point_prompts = None,
                           tc_point_prompts = None,
                            ar_bbox_prompts = None,
                            tc_bbox_prompts = None,
                            ori_img_size = None):
        
        for i in range(len(ori_img_size)):
            h_scale = self.sam_img_size[0] / ori_img_size[i][0]
            w_scale = self.sam_img_size[1] / ori_img_size[i][1]
        
            if tc_point_prompts is not None:
                if tc_point_prompts[i] is not None:
                    tc_point, tc_label = tc_point_prompts[i] 
                    tc_point[:,:, 0]  *= w_scale
                    tc_point[:,:, 1]  *= h_scale
                    tc_point = torch.round(tc_point)
                    tc_point_prompts[i] = (tc_point, tc_label)
            if ar_point_prompts is not None:
                if ar_point_prompts[i] is not None:
                    ar_point, ar_label = ar_point_prompts[i]
                    ar_point[:,:, 0]  *= w_scale
                    ar_point[:,:, 1]  *= h_scale
                    ar_point = torch.round(ar_point)
                    ar_point_prompts[i] = (ar_point, ar_label)
            if tc_bbox_prompts is not None:
                if tc_bbox_prompts[i] is not None:
                    tc_bbox_prompts[i][..., 0]  *= w_scale
                    tc_bbox_prompts[i][..., 1]  *= h_scale
                    tc_bbox_prompts[i][..., 2]  *= w_scale
                    tc_bbox_prompts[i][..., 3]  *= h_scale
                    tc_bbox_prompts[i] = torch.round(tc_bbox_prompts[i])
            if ar_bbox_prompts is not None:
                if ar_bbox_prompts[i] is not None:
                    ar_bbox_prompts[i][..., 0]  *= w_scale
                    ar_bbox_prompts[i][..., 1]  *= h_scale
                    ar_bbox_prompts[i][..., 2]  *= w_scale
                    ar_bbox_prompts[i][..., 3]  *= h_scale
                    ar_bbox_prompts[i] = torch.round(ar_bbox_prompts[i])


        
        return ar_point_prompts, tc_point_prompts, ar_bbox_prompts, tc_bbox_prompts
    
     
    def discretize_mask(self, masks_logits):
        return torch.gt(masks_logits, self.ori_sam.mask_threshold).float()
    
    def assemble_raw_masks(self, raw_masks: List):
        # Order: discretize -> sum over all output masks -> clamp the values larger than 1 -> stack into a batch
        masks = []
        for r_m in raw_masks:
            # discretize the logits into a 0-1 mask
            r_m = self.discretize_mask(r_m)
            # sum up the prediced masks by all the prompts of a single image
            r_m = torch.sum(r_m, dim=0, keepdim=True)
            masks.append(torch.clamp(r_m, max=1.0))
        return masks
            

        
        