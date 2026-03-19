import os
import time

def main():
    # epoch_num = 50
    # # Testing a wider LR range and a 'middle' ground
    # lr_list = [1e-4, 5e-5, 1e-5]
    
    # # Testing the balance between the two specific classes
    # # AR is easier to find, TC is 8x smaller/harder
    # class_balance_configs = [
    #     (1, 1),  # Equal importance
    #     (1, 5),  # TC is 5x more important than AR
    #     (1, 10)  # TC is 10x more important than AR
    # ]
    
    # # Tversky Pairs: (Alpha, Beta)
    # tversky_pairs = [
    #     (0.3, 0.7), # Focus on Recall (Tiny objects)
    #     (0.2, 0.8), # Extremely aggressive Recall (Tiny TCs)
    #     (0.5, 0.5)  # Balanced
    # ]
    
    # # Focal Configs: (Focal_Weight, Gamma_AR, Gamma_TC)
    # focal_configs = [
    #     (1.0, 2.0, 5.0),  # Moderate focus
    #     (5.0, 3.0, 8.0),  # High focus (Best for 257:1 ratio)
    #     (10.0, 4.0, 10.0) # Extreme focus for microscopic targets
    # ]

    # for lr in lr_list:
    #     for bce_ar, bce_tc in class_balance_configs:
    #         for f_weight, g_ar, g_tc in focal_configs:
    #             for alpha, beta in tversky_pairs:
                    
    #                 # Shortening run_name to avoid OS path length limits
    #                 run_name = f"EXP_LR{lr}_BCE{bce_tc}_FW{f_weight}_G{g_tc}_A{alpha}"
                    
    #                 cmd = (
    #                     f'python train_script/official/train_adaptation.py '
    #                     f'--run_name "{run_name}" '
    #                     f'--config hp_mode '
    #                     f'--lr {lr} '
    #                     f'--max_epoch_num {epoch_num} '
    #                     f'--bce_weight 10 ' # Base BCE weight
    #                     f'--bce_weight_ar {bce_ar} '
    #                     f'--bce_weight_tc {bce_tc} '
    #                     f'--focal_weight {f_weight} '
    #                     f'--gamma_ar {g_ar} '
    #                     f'--gamma_tc {g_tc} '
    #                     f'--alpha_ar_tversky {alpha} '
    #                     f'--beta_ar_tversky {beta} '
    #                     f'--alpha_tc_tversky {alpha} '
    #                     f'--beta_tc_tversky {beta} '
    #                     f'--tversky_weight 1.0'
    #                 )
                    
    #                 print(f"\n>>> RUNNING: {run_name}")
    #                 os.system(cmd)
    #                 time.sleep(15)
    os.system('python train_script/official/train_adaptation.py --sam_type vit_b')
    time.sleep(15)
    os.system('python train_script/official/train_adaptation.py --sam_type vit_h')
    time.sleep(15)
    os.system('python train_script/official/train_lora_sam.py --config lora ')
    time.sleep(15)
    os.system('python train_script/official/train_generator.py ')


if __name__ == "__main__":
    main()