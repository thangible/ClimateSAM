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
    os.system("python train_script/official/train_tune_token_concat.py --config input_config_concat --sam_type vit_l  --image_encoder_mlp_ratio 0.75 --run_name tune_token_concat_vit_l_128_mlp075_continue --focal_weight 10")

    

if __name__ == "__main__":
    main()