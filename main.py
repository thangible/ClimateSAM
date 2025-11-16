import os
import time

def main():
    os.system('python3 train_unet.py --config input_config --run_name train_cgnet ')
    time.sleep(30)
    os.system('python3 test.py --config input_config --run_name test_cgnet_point --prompt_type point --max_epoch_num 2')
    time.sleep(30)
    os.system('python3 test.py --config input_config --run_name test_cgnet_bbox --prompt_type bbox --max_epoch_num 2')
    time.sleep(30)
    os.system('python3 test.py --config input_config --run_name test_cgnet_mask --prompt_type mask --max_epoch_num 2')


if __name__ == "__main__":
    main()
