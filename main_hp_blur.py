import os
import time
import random
import itertools

def main():
    epoch_num = 15
    total_samples = 40 
    log_file = "search_best_kernel_search_log_official.txt"

    # LOCKED OPTIMAL HYPERPARAMETERS
    atc_t, btc_t = 0.3, 0.7
    aar_t, bar_t = 0.5, 0.5
    atc_f, gtc_f = 0.95, 5.0
    aar_f, gar_f = 0.85, 5.0
    fw, tw, bcew = 1.0, 1.0, 0.0

    # SEARCH: Kernel Sizes (Must be ODD numbers)
    # TCs are tiny, keep them tight. ARs are massive, push them hard.
    tc_kernels = [3, 11, 21]
    ar_kernels = [9, 21, 51]

    # SEARCH: Sigmas (Spread of the blur)
    tc_sigmas = [1, 5, 10]
    ar_sigmas = [2.0, 10.0, 20]

    executed = set()
    count = 0

    print(f"Starting Random Search for Label Smoothing...")
    print(f"Total target runs: {total_samples}")

    while count < total_samples:
        tc_k = random.choice(tc_kernels)
        ar_k = random.choice(ar_kernels)
        tc_s = random.choice(tc_sigmas)
        ar_s = random.choice(ar_sigmas)
        
        combo = (tc_k, ar_k, tc_s, ar_s)
        
        if combo in executed:
            continue
            
        executed.add(combo)
        count += 1

        # Command builder with the new parameters
        cmd = (
            f"python train_script/official/train_adaptation.py --config input_config_official --sam_type vit_b "
            f"--alpha_ar_tversky {aar_t} --beta_ar_tversky {bar_t} "
            f"--alpha_tc_tversky {atc_t} --beta_tc_tversky {btc_t} "
            f"--alpha_ar {aar_f} --gamma_ar {gar_f} "
            f"--alpha_tc {atc_f} --gamma_tc {gtc_f} "
            f"--focal_weight {fw} --tversky_weight {tw} --bce_weight {bcew} "
            f"--smooth_label "  # <-- Enable the flag
            f"--tc_kernel_size {tc_k} --ar_kernel_size {ar_k} "
            f"--tc_sigma {tc_s} --ar_sigma {ar_s} "
            f"--max_epoch_num {epoch_num} --run_name hparam_smooth_v1_{count}"
        )

        # Flag aggressive AR blurring for the console output
        is_aggressive = "AGGRESSIVE_AR" if (ar_k >= 31 or ar_s >= 10.0) else "STANDARD"
        
        print(f"\n--- Iteration {count}/{total_samples} [{is_aggressive}] ---")
        print(f"Kernel: TC({tc_k}) AR({ar_k}) | Sigma: TC({tc_s}) AR({ar_s})")
        
        with open(log_file, "a") as f:
            f.write(f"Iter {count} [{is_aggressive}]: {cmd}\n")

        os.system(cmd)
        time.sleep(5)

if __name__ == "__main__":
    main()