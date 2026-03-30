import os
import time
import random

def main():
    # os.system("python train_script/official/train_tune_token_concat.py --config input_config_concat --sam_type vit_b  --image_encoder_mlp_ratio 0.5 --run_name tune_token_concat_vit_b_128_mlp05")
    # time.sleep(10)
    # os.system("python train_script/official/train_tune_token_concat.py --config input_config_concat --sam_type vit_b  --image_encoder_mlp_ratio 0.75 --run_name tune_token_concat_vit_b_128_mlp075")

    # time.sleep(10)
    # os.system("python script/official/train_adaptation.py --config input_config_concat_2 --sam_type vit_l  --run_name infused_token_vit_l_mlp075_bce_notrans")
    
    # TEST ML RATIO CONCAT VIT L
    # os.system("python train_script/official/train_tune_token_concat.py --config input_config_concat --sam_type vit_l  --image_encoder_mlp_ratio 0.5 --run_name tune_token_concat_vit_l_mlp05")
    # time.sleep(10)
    # TEST ML RATIO CONCAT VIT B
    os.system("python train_script/official/train_tune_token_concat.py --config input_config_concat_2 --sam_type vit_b  --image_encoder_mlp_ratio 0.5 --run_name tune_token_concat_vit_b_mlp05")
    time.sleep(10)
    os.system("python train_script/official/train_tune_token_concat.py --config input_config_concat_2 --sam_type vit_b  --image_encoder_mlp_ratio 0.75 --run_name tune_token_concat_vit_b_mlp075")
    time.sleep(10)
    os.system("python train_script/official/train_tune_token_concat.py --config input_config_concat --sam_type vit_b  --image_encoder_mlp_ratio 0.75 --run_name tune_token_concat_vit_b_mlp1")
    # TRAIN SINGLE TOKEN
    # os.system("python train_script/official/train_sam_single.py --config input_config_concat_2 --sam_type vit_b  --image_encoder_mlp_ratio 0.5 --run_name tune_single_token_vit_b_mlp05")
    # time.sleep(10)
    # os.system("python train_script/official/train_sam_single.py --config input_config_concat_2 --sam_type vit_l  --image_encoder_mlp_ratio 0.5 --run_name tune_single_token_vit_l_mlp05")
    # time.sleep(10)
if __name__ == "__main__":
    main()