import os
import time
import random

# python train_script/official/train_generator.py --config input_config_generator_test --fuse_channels 128 --sam_type vit_b --run_name test --encoder_weights_name best_weights/infused_token_vitb_mlp1_best --image_encoder_mlp_ratio 1

def main():

    os.system("python train_script/official/train_generator.py --config input_config_generator --fuse_channels 128 --sam_type vit_b --run_name generator_256_vit_b --encoder_weights_name infused_token_vitb_mlp1_best ")


    # os.system("python train_script/official/train_generator_token.py --config input_config_generator --fuse_channels 256 --sam_type vit_l --run_name generator_token_256_vit_l")
    # time.sleep(10)
    # os.system("python train_script/official/train_generator_token.py --config input_config_generator --fuse_channels 128 --sam_type vit_l --run_name generator_token_512_vit_l")

if __name__ == "__main__":
    main()