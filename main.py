import os
import time
import random

def main():
    # Configuration
    epoch_num = 50
    total_samples = 40  # Set how many random combinations to test
    log_file = "hparam_search_log.txt"

    # Search Space
    tversky_pairs_tc = [(0.3, 0.7), (0.2, 0.8), (0.5, 0.5)]
    tversky_pairs_ar = [(0.7, 0.3), (0.8, 0.2), (0.5, 0.5)]
    
    # Reduced gamma range to prevent gradient instability (Max 8.0)
    focal_configs_ar = [(2.0, 5.0), (3.0, 8.0), (4.0, 7.0)]
    focal_configs_tc = [(2.0, 5.0), (3.0, 8.0), (4.0, 7.0)]
    
    loss_weight_pairs = [(1.0, 1.0), (0.5, 1.5), (1.5, 0.5)]

    # Tracking executed combinations to avoid duplicates
    executed = set()

    print(f"Starting Random Search: {total_samples} iterations.")

    for i in range(total_samples):
        # Randomly sample from your lists
        atc_t, btc_t = random.choice(tversky_pairs_tc)
        aar_t, bar_t = random.choice(tversky_pairs_ar)
        aar_f, gar_f = random.choice(focal_configs_ar)
        atc_f, gtc_f = random.choice(focal_configs_tc)
        # tw, fw = random.choice(loss_weight_pairs)
        tw, fw = 1.0, 1.0  # Keep weights constant for stability

        combo = (atc_t, btc_t, aar_t, bar_t, aar_f, gar_f, atc_f, gtc_f, tw, fw)
        
        if combo in executed:
            continue
        executed.add(combo)

        cmd = (
            f"python train_script/official/train_adaptation.py --config hp_mode "
            f"--alpha_ar_tversky {aar_t} --beta_ar_tversky {bar_t} "
            f"--alpha_tc_tversky {atc_t} --beta_tc_tversky {btc_t} "
            f"--focal_weight {fw} --gamma_ar {gar_f} --gamma_tc {gtc_f} "
            f"--tversky_weight {tw}"
        )

        print(f"\n--- Running Iteration {i+1}/{total_samples} ---")
        print(cmd)
        
        # Log the attempt
        with open(log_file, "a") as f:
            f.write(f"Iter {i+1}: {cmd}\n")

        os.system(cmd)
        time.sleep(15)

if __name__ == "__main__":
    main()