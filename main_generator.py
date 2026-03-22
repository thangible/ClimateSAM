import os
import time
import random

def main():
    os.system("python train_script/official/train_generator_token.py --config input_config_generator --fuse_channels 256 --sam_type vit_l --run_name generator_token_256_vit_l")
    # time.sleep(10)
    # os.system("python train_script/official/train_generator_token.py --config input_config_generator --fuse_channels 128 --sam_type vit_l --run_name generator_token_512_vit_l")

if __name__ == "__main__":
    main()