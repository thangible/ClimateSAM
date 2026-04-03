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
import torch.utils.checkpoint as checkpoint
import wandb
import matplotlib.pyplot as plt
from model.input_adapter import ClimateInputAdapter, LinearClimateInputAdapter

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

    def __init__(self, model_type: str, input_weights: List[float] = None, verbose = False, use_prompt_generator = False, mlp_ratio = 0.25, use_checkpoint = True, enable_wandb_logging = False):
        
        super(ClimateSAM, self).__init__()
        
        assert model_type in ['vit_b', 'vit_l', 'vit_h'], f"invalid model_type: {model_type}!"
        self.verbose = verbose
        self.use_prompt_generator = use_prompt_generator
        self.enable_wandb_logging = enable_wandb_logging
        
        # ORI SAM model
        self.ori_sam = sam_model_registry[model_type](sam_ckpt_path_dict[model_type])
        self.sam_img_size = (self.ori_sam.image_encoder.img_size, self.ori_sam.image_encoder.img_size)
        
        # ClimateSAM model
        self.input_adapter = LinearClimateInputAdapter(in_channels=16, out_channels=3)
        
        self.mask_decoder = MaskDecoderHQ(
            model_type, self.ori_sam.mask_decoder.state_dict()
        )
        
        # Pass use_checkpoint parameter to image encoder
        self.image_encoder = ClimateSAMImageEncoder(
            ori_sam=self.ori_sam, 
            fix=True,
            hq_token_ar=self.mask_decoder.hf_token_ar.weight,
            hq_token_tc=self.mask_decoder.hf_token_tc.weight, 
            mlp_ratio=mlp_ratio,
            use_checkpoint=use_checkpoint  # Add this line
        )
        if self.use_prompt_generator:
            num_features_map = {
                'vit_b': 12,
                'vit_l': 24,
                'vit_h': 32  # Assuming ViT-H has 32 layers
            }
            feature_per_block = {
                'vit_b': 3,
                'vit_l': 6,
                'vit_h': 9  # Assuming ViT-H has 4 features per block
            }
            self.prompt_generator = PromptGenerator(num_features=num_features_map[model_type],
                                                    features_per_block=feature_per_block[model_type])
        self.prompt_encoder = PromptEncoderWrapper(ori_sam=self.ori_sam, fix=True)
        
        #set weights for input adaptation:
        # Zero out all weights 
        # Define the channels where you want high weights
        # with torch.no_grad():
        #     torch.nn.init.normal_(self.input_adapter[0].weight, mean=0.0, std=0.02)
        #     if input_weights is None:
        #         # Default input_weights correspond to indices of specific climate variables:
        #         # 'TMQ' (Total Precipitable Water Vapor), 'U850' (Zonal Wind at 850 hPa), 
        #         # and 'V850' (Meridional Wind at 850 hPa).
        #         input_weights = [0, 1, 2] # for 'TMQ', 'U850', 'V850'
        #     # For instance, set those weights to 1.0 for every output channel
        #     for out_ch in range(self.input_adapter[0].weight.shape[0]):
        #         for in_ch in input_weights:
        #             self.input_adapter[0].weight[out_ch, in_ch, 0, 0] = 1.0    
        # # self.input_adapter[0].weight.requires_grad = False # freeze the input adaptation layer
                
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
                if n not in ['image_encoder', 'mask_decoder' ]:
                    c.eval()
                else:
                    c.train(mode=mode)
            if verbose:
                print("Training image_encoder")
                
        if phase == 2:
            for n, c in self.named_children():
                if n not in ['image_encoder', 'mask_decoder', 'input_adapter']:
                    c.eval()
                else:
                    c.train(mode=mode)
                    if n == 'input_adapter':
                        for p in c.parameters():
                            p.requires_grad = True
            if verbose:
                print("Training image_encoder ")
                
        elif phase == 3:
            # Phase 3: Train only prompt_encoder
            self.enable_prompt_generator()
            for n, c in self.named_children():
                if n not in ['prompt_generator']:
                    c.eval()
                else:
                    c.train(mode=mode)
                    if n == 'prompt_generator':
                        for param in c.parameters():
                            param.requires_grad = True
            if verbose:
                print("Training prompt_encoder and prompt_generator")

            
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

    
        
    def encode_images(self, input: Union[List[torch.Tensor], None]):
        """
        Separate image encoding step that processes input and returns image embeddings and intermediate features.
        
        Args:
            input: Input tensor batch
            
        Returns:
            tuple: (image_embeddings, interm_embeddings, preprocessed_input, original_image_sizes)
        """
        ori_img_size = [(input[i].shape[-2], input[i].shape[-1]) for i in range(len(input))]
        
        # Preprocess input
        input = self.interpolate_input(input)  # from 16x768x1152 to 16x1024x1024
        imgs = input[:, :3, :, :] # from 16x1024x1024 to 3x1024x1024
        imgs = self.input_adapter(input)  # from 16x1024x1024 to 3x1024x1024
        imgs = self.preprocess_images(imgs)  # normalize the input images
        
        # Encode the images
        image_input = imgs.clone().detach()
        image_embeddings, interm_embeddings = self.image_encoder(imgs)
        # shape batch x [256, 64, 64] and 12 x torch.Size([batch, 64, 64, 768])
        return image_embeddings, interm_embeddings, image_input, ori_img_size


    def  forward(
            self,
            image_input,
            image_embeddings: torch.Tensor,
            interm_embeddings: List[torch.Tensor],
            ori_img_size: List[Tuple],
            hq_token_weight_ar: torch.Tensor = None,
            hq_token_weight_tc: torch.Tensor = None,
            return_all_hq_masks: bool = False,
            ar_point_prompts: List[Union[torch.Tensor, None]] = None,
            tc_point_prompts: List[Union[torch.Tensor, None]] = None,
            ar_bbox_prompts: List[Union[torch.Tensor, None]] = None,
            tc_bbox_prompts: List[Union[torch.Tensor, None]] = None,
            ar_mask_prompts: List[Union[torch.Tensor, None]] = None,
            tc_mask_prompts: List[Union[torch.Tensor, None]] = None
    ):


        batch_size = len(image_embeddings)

        ar_point_prompts, tc_point_prompts, ar_bbox_prompts, tc_bbox_prompts = self.preprocess_prompts(
            ar_point_prompts=ar_point_prompts,
            tc_point_prompts=tc_point_prompts,
            ar_bbox_prompts=ar_bbox_prompts,
            tc_bbox_prompts=tc_bbox_prompts,
            ori_img_size=ori_img_size
        )
        
        # # Log prompt shapes after preprocessing only if debugging
        # if self.enable_wandb_logging:
        #     self.log_prompt_shapes(ar_point_prompts, tc_point_prompts, ar_bbox_prompts, tc_bbox_prompts, "forward_preprocessed")
        
        if self.use_prompt_generator:
            tc_masks, ar_masks = self.prompt_generator(interm_embeddings) # shape: batch x 2 x 256 x 256
            ar_point_prompts, tc_point_prompts, ar_bbox_prompts, tc_bbox_prompts = None, None, None, None
            

            
        tc_sparse_embeddings, tc_dense_embeddings = [], []
        ar_sparse_embeddings, ar_dense_embeddings = [], []
        for batch_idx in range(batch_size):
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

        # # Log prompt embeddings only if debugging
        # if self.enable_wandb_logging:
        #     self.log_embeddings(tc_sparse_embeddings, tc_dense_embeddings, "forward/tc_prompt")
        #     self.log_embeddings(ar_sparse_embeddings, ar_dense_embeddings, "forward/ar_prompt")

        _, tc_pred_masks = self.mask_decoder(
            type = 'TC',  # either 'TQ' or 'AR'
            image_embeddings=image_embeddings,
            image_pe=[self.prompt_encoder.get_dense_pe() for _ in range(batch_size)],
            sparse_prompt_embeddings=tc_sparse_embeddings,
            dense_prompt_embeddings=tc_dense_embeddings,
            multimask_output=False,
            interm_embeddings=interm_embeddings,
            hq_token_weight=hq_token_weight_tc,
            return_all_hq_masks=return_all_hq_masks
        )
        
        _, ar_pred_masks = self.mask_decoder(
            type = 'AR',  # either 'TQ' or 'AR'
            image_embeddings=image_embeddings,
            image_pe=[self.prompt_encoder.get_dense_pe() for _ in range(batch_size)],
            sparse_prompt_embeddings= ar_sparse_embeddings,
            dense_prompt_embeddings= ar_dense_embeddings,
            multimask_output=False,
            interm_embeddings=interm_embeddings,
            hq_token_weight=hq_token_weight_ar,
            return_all_hq_masks=return_all_hq_masks
        )
        
        # Log predicted masks only if debugging
        # if self.enable_wandb_logging:
        #     self.log_masks(tc_pred_masks, "forward/tc_predicted", log_images=self.training, max_images=2)
        #     self.log_masks(ar_pred_masks, "forward/ar_predicted", log_images=self.training, max_images=2)
        
        # rescale the mask size back to original image size
        tc_postprocess_masks_hq = [m_hq.clone() for m_hq in tc_pred_masks]
        for i in range(len(tc_postprocess_masks_hq)):
            tc_postprocess_masks_hq[i] = self.postprocess(output_masks=tc_postprocess_masks_hq[i], ori_img_size=ori_img_size[i])
        
        ar_postprocess_masks_hq = [m_hq.clone() for m_hq in ar_pred_masks]
        for i in range(len(ar_postprocess_masks_hq)):
            ar_postprocess_masks_hq[i] = self.postprocess(output_masks=ar_postprocess_masks_hq[i], ori_img_size=ori_img_size[i])
        
        # Log postprocessed masks only if debugging
        # if self.enable_wandb_logging:
        #     self.log_masks(tc_postprocess_masks_hq, "forward/tc_postprocessed", log_images=self.training, max_images=2)
        #     self.log_masks(ar_postprocess_masks_hq, "forward/ar_postprocessed", log_images=self.training, max_images=2)
        
        if not self.training:
            tc_postprocess_masks_hq = self.assemble_raw_masks(tc_postprocess_masks_hq) 
            ar_postprocess_masks_hq = self.assemble_raw_masks(ar_postprocess_masks_hq)
        
        # Clear unnecessary variables early
        # del imgs  # Delete after use
        torch.cuda.empty_cache()
        
        # Process embeddings in chunks if needed
        batch_size = len(image_embeddings)
        
        # Clear intermediate variables
        del image_embeddings, interm_embeddings
        torch.cuda.empty_cache()
        
        return tc_postprocess_masks_hq, ar_postprocess_masks_hq, image_input
    
    @torch.no_grad()
    def save_image_embeddings(self,
            save_path: str,
            input: Union[List[torch.Tensor], None],
            gt_mask: Union[List[torch.Tensor], None]
    ):
        input = self.interpolate_input(input) # from 16x768x1152 to 16x1024x1024
        
        imgs = input[:, :3, :, :] # from 16x1024x1024 to 3x1024x1024
        imgs = self.input_adapter(input) # from 16x1024x1024 to 3x1024x1024
        imgs = self.preprocess_images(imgs) # normalize the input images
        
        # encode the images
        image_input = imgs.clone().detach()
        image_embeddings, interm_embeddings = self.image_encoder(imgs) # shape batch x [256, 64, 64] and 12 x torch.Size([batch, 64, 64, 768])
        batch_size = len(image_embeddings)
        
        torch.save({
            'batch_size': batch_size,
            'input': input,
            'image_embeddings': image_embeddings,
            'interm_embeddings': interm_embeddings,
            'image_input': image_input,
            'gt_mask': gt_mask
        }, save_path)
        print(f"Image embeddings saved to {save_path}")


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
                output_masks, ori_img_size, mode="nearest"
            )
        return output_masks

    def interpolate_input(self, input: torch.Tensor):
        # Check if input size matches self.sam_img_size
        if input.shape[-2:] != self.sam_img_size:
            input = F.interpolate(input, size=self.sam_img_size, mode='nearest')
            
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
        batch_num = len(ori_img_size)
        for i in range(batch_num):
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

    @torch.no_grad()
    def set_infer_img(self, input: Union[List[torch.Tensor], torch.Tensor]):
        """Set and preprocess images for inference, storing features for reuse."""
        if isinstance(input, torch.Tensor):
            if len(input.shape) == 3:
                input = [input]
            elif len(input.shape) == 4:
                input = [input[i] for i in range(input.shape[0])]
            else:
                raise RuntimeError(f"Unsupported input shape: {input.shape}")
        elif not isinstance(input, list):
            raise RuntimeError("Input must be tensor or list of tensors")
        
        self.ori_infer_img_size = [(img.shape[-2], img.shape[-1]) for img in input]
        self.ori_infer_img = input
        
        # Preprocess images
        input = self.interpolate_input(torch.stack(input))
        imgs = input[:, :3, :, :]  # Take first 3 channels
        imgs = self.preprocess_images(imgs)
        
        # Store features for reuse
        self.img_features, self.interm_features = self.image_encoder(imgs)
        return imgs, self.img_features, self.interm_features

    @torch.no_grad()
    def infer(
        self,
        ar_point_prompts: List[Union[torch.Tensor, None]] = None,
        tc_point_prompts: List[Union[torch.Tensor, None]] = None,
        ar_bbox_prompts: List[Union[torch.Tensor, None]] = None,
        tc_bbox_prompts: List[Union[torch.Tensor, None]] = None,
        ar_mask_prompts: List[Union[torch.Tensor, None]] = None,
        tc_mask_prompts: List[Union[torch.Tensor, None]] = None,
        return_all_hq_masks: bool = False
    ):
        """Perform inference using precomputed image features."""
        if not hasattr(self, 'img_features') or not hasattr(self, 'interm_features'):
            raise RuntimeError("Must call set_infer_img() before infer()")
        
        batch_size = len(self.img_features)
        
        # # Log inference info only if debugging
        # if self.enable_wandb_logging and wandb.run:
        #     wandb.log({
        #         "infer/batch_size": batch_size,
        #         "infer/img_features_shapes": [list(feat.shape) for feat in self.img_features],
        #         "infer/interm_features_count": len(self.interm_features)
        #     })
        
        # Preprocess prompts
        ar_point_prompts, tc_point_prompts, ar_bbox_prompts, tc_bbox_prompts = self.preprocess_prompts(
            ar_point_prompts=ar_point_prompts,
            tc_point_prompts=tc_point_prompts,
            ar_bbox_prompts=ar_bbox_prompts,
            tc_bbox_prompts=tc_bbox_prompts,
            ori_img_size=self.ori_infer_img_size
        )
        
        # # Log prompt shapes for inference only if debugging
        # if self.enable_wandb_logging:
        #     self.log_prompt_shapes(ar_point_prompts, tc_point_prompts, ar_bbox_prompts, tc_bbox_prompts, "infer")
        
        # Handle prompt generator if enabled
        if self.use_prompt_generator:
            tc_masks, ar_masks = self.prompt_generator(self.interm_features)
            ar_point_prompts, tc_point_prompts, ar_bbox_prompts, tc_bbox_prompts = None, None, None, None
        
        # Encode prompts
        tc_sparse_embeddings, tc_dense_embeddings = [], []
        ar_sparse_embeddings, ar_dense_embeddings = [], []
        
        for batch_idx in range(batch_size):
            current_tc_sparse, current_tc_dense = self.prompt_encoder(
                points=tc_point_prompts[batch_idx] if tc_point_prompts is not None else None,
                boxes=tc_bbox_prompts[batch_idx] if tc_bbox_prompts is not None else None,
                masks=tc_mask_prompts[batch_idx] if tc_mask_prompts is not None else None,
            )
            
            current_ar_sparse, current_ar_dense = self.prompt_encoder(
                points=ar_point_prompts[batch_idx] if ar_point_prompts is not None else None,
                boxes=ar_bbox_prompts[batch_idx] if ar_bbox_prompts is not None else None,
                masks=ar_mask_prompts[batch_idx] if ar_mask_prompts is not None else None,
            )
            
            tc_sparse_embeddings.append(current_tc_sparse)
            ar_sparse_embeddings.append(current_ar_sparse)
            tc_dense_embeddings.append(current_tc_dense)
            ar_dense_embeddings.append(current_ar_dense)
        
        # # Log prompt embeddings only if debugging
        # if self.enable_wandb_logging:
        #     self.log_embeddings(tc_sparse_embeddings, tc_dense_embeddings, "infer/tc_prompt")
        #     self.log_embeddings(ar_sparse_embeddings, ar_dense_embeddings, "infer/ar_prompt")
        
        # Decode masks
        _, tc_pred_masks = self.mask_decoder(
            type='TC',
            image_embeddings=self.img_features,
            image_pe=[self.prompt_encoder.get_dense_pe() for _ in range(batch_size)],
            sparse_prompt_embeddings=tc_sparse_embeddings,
            dense_prompt_embeddings=tc_dense_embeddings,
            multimask_output=False,
            interm_embeddings=self.interm_features,
            return_all_hq_masks=return_all_hq_masks
        )
        
        _, ar_pred_masks = self.mask_decoder(
            type='AR',
            image_embeddings=self.img_features,
            image_pe=[self.prompt_encoder.get_dense_pe() for _ in range(batch_size)],
            sparse_prompt_embeddings=ar_sparse_embeddings,
            dense_prompt_embeddings=ar_dense_embeddings,
            multimask_output=False,
            interm_embeddings=self.interm_features,
            return_all_hq_masks=return_all_hq_masks
        )
        
        # Log predicted masks only if debugging
        # if self.enable_wandb_logging:
        #     self.log_masks(tc_pred_masks, "infer/tc_predicted", log_images=False, max_images=2)
        #     self.log_masks(ar_pred_masks, "infer/ar_predicted", log_images=False, max_images=2)
        
        # Postprocess masks to original image size
        tc_postprocess_masks = []
        ar_postprocess_masks = []
        
        for i in range(len(tc_pred_masks)):
            tc_postprocess_masks.append(
                self.postprocess(tc_pred_masks[i].clone(), self.ori_infer_img_size[i])
            )
            ar_postprocess_masks.append(
                self.postprocess(ar_pred_masks[i].clone(), self.ori_infer_img_size[i])
            )
        
        # Log postprocessed masks only if debugging
        if self.enable_wandb_logging:
            self.log_masks(tc_postprocess_masks, "infer/tc_postprocessed", log_images=True, max_images=2)
            self.log_masks(ar_postprocess_masks, "infer/ar_postprocessed", log_images=True, max_images=2)
        
        # Assemble raw masks (discretize and combine)
        tc_postprocess_masks = self.assemble_raw_masks(tc_postprocess_masks)
        ar_postprocess_masks = self.assemble_raw_masks(ar_postprocess_masks)

        if self.enable_wandb_logging:
            self.log_masks(tc_postprocess_masks, "infer/tc_postprocessed", log_images=True, max_images=2, binary=True)
            self.log_masks(ar_postprocess_masks, "infer/ar_postprocessed", log_images=True, max_images=2, binary=True)

        return tc_postprocess_masks, ar_postprocess_masks
    
    def log_masks(self, masks, prefix="", log_images=False, max_images=2, binary=False):
        """Log mask shapes and optionally visualizations

        Args:
            masks: list of torch.Tensor masks
            prefix: prefix for wandb logging keys
            log_images: whether to log mask images
            max_images: max number of images to log
            binary: if True, visualize mask as binary (0/1), else use colormap
        """
        if not self.enable_wandb_logging or not wandb.run:
            return

        mask_info = {}

        for i, mask in enumerate(masks):
            mask_info[f"{prefix}_mask_{i}_shape"] = list(mask.shape)
            mask_info[f"{prefix}_mask_{i}_min"] = mask.min().item()
            mask_info[f"{prefix}_mask_{i}_max"] = mask.max().item()
            mask_info[f"{prefix}_mask_{i}_mean"] = mask.mean().item()

            if log_images and i < max_images:
                mask_np = mask.detach().cpu().numpy()
                if len(mask_np.shape) == 4:
                    mask_np = mask_np[0, 0]
                elif len(mask_np.shape) == 3:
                    mask_np = mask_np[0]

                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                if binary:
                    im = ax.imshow(mask_np, cmap='gray', vmin=0, vmax=1)
                else:
                    im = ax.imshow(mask_np, cmap='viridis')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f'{prefix} Mask {i}')
                ax.axis('off')

                mask_info[f"{prefix}_mask_{i}_image"] = wandb.Image(fig)
                plt.close(fig)

        wandb.log(mask_info)



