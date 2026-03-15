import os
import time

def main():
    os.system('python train_script/official/test_generator.py --run_name "GENERATOR_TEST" --config generator_test')
    
    epoch_num = 20
    lr_list = [1e-4, 1e-5, 1e-6]
    
    bce_weight_list = [0, 1, 10]
    focal_weight_list = [0, 1]
    tversky_weight_list = [0, 1]
    alpha_ar_tversky_list = [0.3, 0.5, 0.7]
    beta_ar_tversky_list = [0.7, 0.5, 0.3]
    alpha_tc_tversky_list = [0.3, 0.5, 0.7]
    beta_tc_tversky_list = [0.7, 0.5, 0.3]
    gamma_ar_list = [3]
    gamma_tc_list = [8]
    alpha_ar_list = [0.75]
    alpha_tc_list = [0.996]



    for bce_weight in bce_weight_list:
        for focal_weight in focal_weight_list:
            for tversky_weight in tversky_weight_list:
                for alpha_ar_tversky in alpha_ar_tversky_list:
                    for beta_ar_tversky in beta_ar_tversky_list:
                        for alpha_tc_tversky in alpha_tc_tversky_list:
                            for beta_tc_tversky in beta_tc_tversky_list:
                                for gamma_ar in gamma_ar_list:
                                    for gamma_tc in gamma_tc_list:
                                        for alpha_ar in alpha_ar_list:
                                            for alpha_tc in alpha_tc_list:
                                                for lr in lr_list:
                                                    if bce_weight == 0 and focal_weight == 0 and tversky_weight == 0:
                                                        continue
                                                    
                                                    run_name = f'ADAPTATION bce_{bce_weight}_focal_{focal_weight}_tversky_{tversky_weight}_alpha_ar_tversky_{alpha_ar_tversky}_beta_ar_tversky_{beta_ar_tversky}_alpha_tc_tversky_{alpha_tc_tversky}_beta_tc_tversky_{beta_tc_tversky}_gamma_ar_{gamma_ar}_gamma_tc_{gamma_tc}_alpha_ar_{alpha_ar}_alpha_tc_{alpha_tc}_lr_{lr}'
                                                    
                                                    os.system(f'python train_script/official/train_adaptation.py --run_name {run_name} --fuse_channels 256 --bce_weight {bce_weight} --focal_weight {focal_weight} --alpha_ar_tversky {alpha_ar_tversky} --beta_ar_tversky {beta_ar_tversky} --alpha_tc_tversky {alpha_tc_tversky} --beta_tc_tversky {beta_tc_tversky} --gamma_ar {gamma_ar} --gamma_tc {gamma_tc} --alpha_ar {alpha_ar} --alpha_tc {alpha_tc} --lr {lr} --epoch_num {epoch_num} ')
                                                    time.sleep(30)


if __name__ == "__main__":
    main()
