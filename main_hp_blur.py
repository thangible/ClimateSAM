import os
import time
import itertools

def main():
    epoch_num = 15
    log_file = "search_best_kernel_search_log_official.txt"

    # LOCKED OPTIMAL HYPERPARAMETERS
    atc_t, btc_t = 0.3, 0.7
    aar_t, bar_t = 0.5, 0.5
    atc_f, gtc_f = 0.95, 5.0
    aar_f, gar_f = 0.85, 5.0
    fw, tw, bcew = 1.0, 1.0, 0.0

    # SEARCH SPACE
    tc_kernels = [3, 11, 21]
    tc_sigmas = [1, 5, 10]
    
    ar_kernels = [9, 21, 51]
    ar_sigmas = [2.0, 10.0, 20.0]

    # BASELINES (Used to lock one class while searching the other)
    base_ar_k, base_ar_s = 9, 2.0
    base_tc_k, base_tc_s = 3, 1.0

    count = 0
    # 1 Baseline + (9 TC runs) + (9 AR runs)
    total_samples = 1 + (len(tc_kernels) * len(tc_sigmas)) + (len(ar_kernels) * len(ar_sigmas))
    
    print(f"Starting Independent Grid Search for Label Smoothing...")
    print(f"Total target runs: {total_samples}")

    # ==========================================
    # PHASE 0: BASELINE (NO SMOOTHING)
    # ==========================================
    print("\n" + "="*40)
    print("PHASE 0: BASELINE (NO SMOOTHING)")
    print("="*40)
    
    count += 1
    cmd_baseline = (
        f"python train_script/official/train_adaptation.py --config hp_mode --sam_type vit_b "
        f"--alpha_ar_tversky {aar_t} --beta_ar_tversky {bar_t} "
        f"--alpha_tc_tversky {atc_t} --beta_tc_tversky {btc_t} "
        f"--alpha_ar {aar_f} --gamma_ar {gar_f} "
        f"--alpha_tc {atc_f} --gamma_tc {gtc_f} "
        f"--focal_weight {fw} --tversky_weight {tw} --bce_weight {bcew} "
        f"--max_epoch_num {epoch_num} --run_name hparam_smooth_BASELINE_0"
    )

    print(f"\n--- Iteration {count}/{total_samples} [PHASE 0 | BASELINE] ---")
    print("Kernel: NONE | Sigma: NONE")
    
    with open(log_file, "a") as f:
        f.write(f"Iter {count} [PHASE 0 | BASELINE]: {cmd_baseline}\n")

    os.system(cmd_baseline)
    time.sleep(5)


    # ==========================================
    # PHASE 1: SEARCH TC (LOCK AR TO BASELINE)
    # ==========================================
    print("\n" + "="*40)
    print("PHASE 1: TC PARAMETER SEARCH")
    print("="*40)
    
    for tc_k, tc_s in itertools.product(tc_kernels, tc_sigmas):
        count += 1
        ar_k, ar_s = base_ar_k, base_ar_s
        
        cmd = (
            f"python train_script/official/train_adaptation.py --config hp_mode --sam_type vit_b "
            f"--alpha_ar_tversky {aar_t} --beta_ar_tversky {bar_t} "
            f"--alpha_tc_tversky {atc_t} --beta_tc_tversky {btc_t} "
            f"--alpha_ar {aar_f} --gamma_ar {gar_f} "
            f"--alpha_tc {atc_f} --gamma_tc {gtc_f} "
            f"--focal_weight {fw} --tversky_weight {tw} --bce_weight {bcew} "
            f"--smooth_label "
            f"--tc_kernel_size {tc_k} --ar_kernel_size {ar_k} "
            f"--tc_sigma {tc_s} --ar_sigma {ar_s} "
            f"--max_epoch_num {epoch_num} --run_name hparam_smooth_TC_{count}"
        )

        is_aggressive = "AGGRESSIVE_TC" if (tc_k >= 11 or tc_s >= 5.0) else "STANDARD"
        
        print(f"\n--- Iteration {count}/{total_samples} [PHASE 1 | {is_aggressive}] ---")
        print(f"Kernel: TC({tc_k}) AR({ar_k}) | Sigma: TC({tc_s}) AR({ar_s})")
        
        with open(log_file, "a") as f:
            f.write(f"Iter {count} [PHASE 1 | {is_aggressive}]: {cmd}\n")

        os.system(cmd)
        time.sleep(5)

    # ==========================================
    # PHASE 2: SEARCH AR (LOCK TC TO BASELINE)
    # ==========================================
    print("\n" + "="*40)
    print("PHASE 2: AR PARAMETER SEARCH")
    print("="*40)

    for ar_k, ar_s in itertools.product(ar_kernels, ar_sigmas):
        count += 1
        tc_k, tc_s = base_tc_k, base_tc_s
        
        cmd = (
            f"python train_script/official/train_adaptation.py --config hp_mode --sam_type vit_b "
            f"--alpha_ar_tversky {aar_t} --beta_ar_tversky {bar_t} "
            f"--alpha_tc_tversky {atc_t} --beta_tc_tversky {btc_t} "
            f"--alpha_ar {aar_f} --gamma_ar {gar_f} "
            f"--alpha_tc {atc_f} --gamma_tc {gtc_f} "
            f"--focal_weight {fw} --tversky_weight {tw} --bce_weight {bcew} "
            f"--smooth_label "
            f"--tc_kernel_size {tc_k} --ar_kernel_size {ar_k} "
            f"--tc_sigma {tc_s} --ar_sigma {ar_s} "
            f"--max_epoch_num {epoch_num} --run_name hparam_smooth_AR_{count}"
        )

        is_aggressive = "AGGRESSIVE_AR" if (ar_k >= 31 or ar_s >= 10.0) else "STANDARD"
        
        print(f"\n--- Iteration {count}/{total_samples} [PHASE 2 | {is_aggressive}] ---")
        print(f"Kernel: TC({tc_k}) AR({ar_k}) | Sigma: TC({tc_s}) AR({ar_s})")
        
        with open(log_file, "a") as f:
            f.write(f"Iter {count} [PHASE 2 | {is_aggressive}]: {cmd}\n")

        os.system(cmd)
        time.sleep(5)

if __name__ == "__main__":
    main()