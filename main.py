import os
import time

def main():
    # os.system('python train_script/official/traiin_lora_sam.py --config lora --run_name TRAIN_LORA_SAM_VIT_L --sam_type vit_l')
    # time.sleep(30)
    # os.system('python train_script/official/traiin_lora_sam.py --config lora --run_name TRAIN_LORA_SAM_VIT_B --sam_type vit_b')
    # time.sleep(30)
    os.system('python train_script/official/traiin_lora_sam.py --config lora --run_name TRAIN_LORA_SAM_VIT_H --sam_type vit_h')
    time.sleep(30)
    
    # os.system('python train_script/official/train_adaptation.py --run_name TRAIN_ADAPTATION_VIT_L_BB50')
    # time.sleep(30)



#     os.system('python train_complete.py --config input_config_prompter --run_name TRAIN_COMPLETE_PROMPTER_128_VIT_L --fuse_channels 128')
#     time.sleep(30)
#     os.system('python train_complete.py --config input_config_prompter --run_name TRAIN_COMPLETE_PROMPTER_64_VIT_L --fuse_channels 64')
#     time.sleep(30)
#     os.system('python train_complete.py --config input_config_prompter --run_name TRAIN_COMPLETE_PROMPTER_32_VIT_L --fuse_channels 32')
#     time.sleep(30)
#     os.system('python train_complete.py --config input_config_prompter --run_name TRAIN_COMPLETE_PROMPTER_256_VIT_L --fuse_channels 256')
if __name__ == "__main__":
    main()
