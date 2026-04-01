import os
import time
import random

def main():
    # LORA DUAL VIT B
    os.system("python train_script/official/train_lora_sam_dual.py --config input_config_official --sam_type vit_b --lora_r 32 --run_name lora_dual_vit_b_r32")
    time.sleep(10)
    os.system("python train_script/official/train_lora_sam_dual.py --config input_config_official --sam_type vit_b --lora_r 64 --run_name lora_dual_vit_b_r64")
    time.sleep(10)
    
    # LORA DUAL VIT L
    os.system("python train_script/official/train_lora_sam_dual.py --config input_config_official --sam_type vit_l --lora_r 32 --run_name lora_dual_vit_l_r32")
    time.sleep(10)
    os.system("python train_script/official/train_lora_sam_dual.py --config input_config_official --sam_type vit_l --lora_r 16 --run_name lora_dual_vit_l_r16")

if __name__ == "__main__":
    main()