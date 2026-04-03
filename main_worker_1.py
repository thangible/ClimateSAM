import os
import time
import random

def main():
    # os.system("python train_script/official/train_tune_token_concat.py --config input_config_concat --sam_type vit_b  --image_encoder_mlp_ratio 0.5 --run_name tune_token_concat_vit_b_128_mlp05")
    # time.sleep(10)
    # os.system("python train_script/official/train_tune_token_concat.py --config input_config_concat --sam_type vit_b  --image_encoder_mlp_ratio 0.75 --run_name tune_token_concat_vit_b_128_mlp075")

    # time.sleep(10)
    # os.system("python train_script/official/train_tune_token_concat.py --config input_config_concat --sam_type vit_l  --image_encoder_mlp_ratio 1 --run_name tune_token_concat_vit_l_128_mlp075 --focal_weight 10")
    # time.sleep(10)
    # os.system("python train_script/official/train_tune_token_concat.py --config input_config_concat --sam_type vit_b  --image_encoder_mlp_ratio 1 --run_name tune_token_concat_vit_b_128_mlp1 --focal_weight 10")
    ## LEARNING RATE FOR CONCAT
    # os.system("python train_script/official/train_tune_token_concat.py --config input_config_concat --sam_type vit_b  --image_encoder_mlp_ratio 0.5 --run_name tune_token_concat_vit_b_128_mlp05 --lr 1e-2")
    # time.sleep(10)
    # LEARNING RATE FOR INFUSED
    # os.system("python train_script/official/train_adaptation.py --config input_config_concat_2 --sam_type vit_b --image_encoder_mlp_ratio 0.5 --run_name infused_token_vit_b_mlp05_1e2 --lr 1e-2")
    # time.sleep(10)
    
    
    
    # INFUSED TEST MLP RATIO VIT B
    os.system("python train_script/official/train_adaptation.py --config input_config_official  --sam_type vit_b --image_encoder_mlp_ratio 1 --run_name infused_token_vit_b_mlp1_CORRECTED --smooth_label")
    time.sleep(15)
    
    os.system("python train_script/official/train_adaptation.py --config input_config_official --sam_type vit_b --image_encoder_mlp_ratio 1 --run_name infused_token_vit_b_mlp1_CORRECTED_NOSMOOTH")
    time.sleep(15)
    # os.system("python train_script/official/train_adaptation.py --config input_config_official --sam_type vit_b --image_encoder_mlp_ratio 0.75 --run_name infused_token_vit_b_mlp075")
    # time.sleep(15)
    os.system("python train_script/official/train_adaptation.py --config input_config_official --sam_type vit_b --image_encoder_mlp_ratio 0.5 --run_name infused_token_vit_b_mlp05_CORRECTED --smooth_label")
    time.sleep(15)
    
        
    # SINGLE SAM VIT B
    os.system("python train_script/official/train_sam_single.py --config input_config_official --sam_type vit_b --image_encoder_mlp_ratio 01 --run_name single_sam_vit_b_CORRECTED --smooth_label")

    
    # LORA SINGLE VIT B
    os.system("python train_script/official/train_lora_sam.py --config input_config_official --sam_type vit_b --lora_r 64 --run_name lora_single_vit_b_r64_CORRECTED --gradient_accumulation_steps 16 --smooth_label")
    time.sleep(15)
    os.system("python train_script/official/train_lora_sam.py --config input_config_official --sam_type vit_b --lora_r 32 --run_name lora_single_vit_b_r32_CORRECTED --gradient_accumulation_steps 16 --smooth_label")
    time.sleep(15)
    
    # # LORA DUAL VIT L
    # os.system("python train_script/official/train_lora_sam.py --config input_config_official --sam_type vit_l --lora_r 32 --run_name lora_single_vit_l_r32 --gradient_accumulation_steps 16")
    # time.sleep(15)
    # os.system("python train_script/official/train_lora_sam.py --config input_config_official --sam_type vit_l --lora_r 16 --run_name lora_single_vit_l_r16 --gradient_accumulation_steps 16")

    # # INFUSED TEST MLP RATIO VIT L
    # os.system("python train_script/official/train_adaptation.py --config input_config_official --sam_type vit_l --image_encoder_mlp_ratio 0.5 --run_name infused_token_vit_l_mlp05")
    # time.sleep(15)
    # os.system("python train_script/official/train_adaptation.py --config input_config_official --sam_type vit_l --image_encoder_mlp_ratio 0.75  --run_name infused_token_vit_l_mlp075")
    # time.sleep(15)


if __name__ == "__main__":
    main()