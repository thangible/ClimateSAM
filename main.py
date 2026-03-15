import os
import time

def main():
    # os.system('python train_script/official/test_generator.py --run_name "GENERATOR_TEST" --config generator_test')
    # time.sleep(30)
    epoch_num = 20
    lr_list = [1e-4, 1e-5]
    
    bce_weight_list = [1, 10]
    tversky_weight_list = [1]
    
    tversky_pairs = [
    (0.3, 0.7), # Focus on Recall (High Beta - good for tiny objects)
    (0.5, 0.5), # Balanced (Standard Dice)
    (0.7, 0.3)  # Focus on Precision (High Alpha - reduces fake detections)
]
    for lr in lr_list:
        for bce_weight in bce_weight_list:
            for tversky_weight in tversky_weight_list:
                for alpha, beta in tversky_pairs:
                    run_name = f"HP_TUNING_lr{lr}_bce{bce_weight}_tversky{tversky_weight}_alpha{alpha}_beta{beta}"
                    config_name = "hp_mode"
                    os.system(f'python train_script/official/train_adaptation.py --run_name "{run_name}" --config {config_name} '
                              f'--lr {lr} --bce_weight_ar {bce_weight} --bce_weight_tc {bce_weight} '
                              f'--alpha_ar_tversky {alpha} --beta_ar_tversky {beta} --alpha_tc_tversky {alpha} --beta_tc_tversky {beta} '
                              f'--max_epoch_num {epoch_num}')
                    time.sleep(30)  # Sleep for 30 seconds between runs to avoid resource contention
    


if __name__ == "__main__":
    main()
