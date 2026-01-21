import os
import time

def main():
    # os.system('python3 train_unet.py --config input_config --run_name train_cgnet ')
    # time.sleep(30)
    
    # os.system('python3 test.py --config input_config --run_name test_cgnet_point_1_0 --prompt_type point --max_epoch_num 2 --positive_point_num 1 --negative_point_num 1')
    # time.sleep(10)
    # os.system('python3 test.py --config input_config --run_name test_cgnet_point_5_0 --prompt_type point --max_epoch_num 2 --positive_point_num 5 --negative_point_num 1')
    # time.sleep(10)
    # os.system('python3 test.py --config input_config --run_name test_cgnet_point_10_0 --prompt_type point --max_epoch_num 2 --positive_point_num 10 --negative_point_num 1')
    # time.sleep(10)
    # os.system('python3 test.py --config input_config --run_name test_cgnet_point_20_0 --prompt_type point --max_epoch_num 2 --positive_point_num 20 --negative_point_num 1')
    # time.sleep(10)
    # os.system('python3 test.py --config input_config --run_name test_cgnet_point_5_5 --prompt_type point --max_epoch_num 2 --positive_point_num 5 --negative_point_num 5')
    # time.sleep(10)
    # os.system('python3 test.py --config input_config --run_name test_cgnet_point_10_5 --prompt_type point --max_epoch_num 2 --positive_point_num 10 --negative_point_num 5')
    # time.sleep(10)
    # os.system('python3 test.py --config input_config --run_name test_cgnet_point_10_10 --prompt_type point --max_epoch_num 2 --positive_point_num 10 --negative_point_num 10')
    # time.sleep(10)
    # os.system('python3 test.py --config input_config --run_name test_cgnet_point_20_10 --prompt_type point --max_epoch_num 2 --positive_point_num 20 --negative_point_num 10')

    # time.sleep(30)
    # os.system('python3 test.py --config input_config --run_name test_cgnet_bbox --prompt_type bbox --max_epoch_num 2')
    # time.sleep(30)
    # os.system('python3 test.py --config input_config --run_name test_cgnet_mask --prompt_type mask --max_epoch_num 2')



    os.system('python train_complete.py --config input_config_prompter --run_name TRAIN_COMPLETE_PROMPTER_128_VIT_L --fuse_channels 128')
    time.sleep(30)
    os.system('python train_complete.py --config input_config_prompter --run_name TRAIN_COMPLETE_PROMPTER_64_VIT_L --fuse_channels 64')
    time.sleep(30)
    os.system('python train_complete.py --config input_config_prompter --run_name TRAIN_COMPLETE_PROMPTER_32_VIT_L --fuse_channels 32')
    time.sleep(30)
    os.system('python train_complete.py --config input_config_prompter --run_name TRAIN_COMPLETE_PROMPTER_256_VIT_L --fuse_channels 256')
if __name__ == "__main__":
    main()
