import os
import time
import random

def main():
    # Configuration
    epoch_num = 30
    total_samples = 20
    log_file = "hparam_search_log_part_2.txt"

    # Search Space Refined for ClimateNet Imbalance
    # TC: 0.5% pixels - Prioritize high Beta for Recall [cite: 68, 76]
    tversky_pairs_tc = [(0.3, 0.7)] 
    
    # AR: 5.7% pixels - Balanced approach [cite: 68, 76]
    tversky_pairs_ar = [(0.5, 0.5)]
    
    # Focal Configs: (Alpha, Gamma)
    # TC: 257:1 Ratio - Requires high Alpha and Gamma 
    focal_configs_tc = [(0.99, 5.0)]
    
    # AR: 17:1 Ratio - Moderate Alpha 
    focal_configs_ar = [(0.90, 2.0)]
    
    fw_configs = [10.0, 20.0, 50.0]  # Focal weights to test

    # Tracking executed combinations
    executed = set()

    print(f"Starting Random Search: {total_samples} iterations targeted.")

    count = 0
    while count < total_samples:
        # Randomly sample from your refined lists
        atc_t, btc_t = random.choice(tversky_pairs_tc)
        aar_t, bar_t = random.choice(tversky_pairs_ar)
        aar_f, gar_f = random.choice(focal_configs_ar)
        atc_f, gtc_f = random.choice(focal_configs_tc)
        fw = random.choice(fw_configs)
        
        # Constant weights for loss stability in Adaptation Phase [cite: 212]
        tw, bcew = 1.0, 0.0

        combo = (atc_t, btc_t, aar_t, bar_t, aar_f, gar_f, atc_f, gtc_f)
        
        if combo in executed:
            continue
            
        executed.add(combo)
        count += 1

        cmd = (
            f"python train_script/official/train_adaptation.py --config hp_mode "
            f"--alpha_ar_tversky {aar_t} --beta_ar_tversky {bar_t} "
            f"--alpha_tc_tversky {atc_t} --beta_tc_tversky {btc_t} "
            f"--alpha_ar {aar_f} --gamma_ar {gar_f} "
            f"--alpha_tc {atc_f} --gamma_tc {gtc_f} "
            f"--focal_weight {fw} --tversky_weight {tw} --bce_weight {bcew} --max_epoch_num {epoch_num} --run_name hparam_search_{count}"
        )

        print(f"\n--- Running Iteration {count}/{total_samples} ---")
        print(f"TC Tversky (a={atc_t}, b={btc_t}) | AR Focal (a={aar_f}, g={gar_f})")
        
        # Log the specific attempt
        with open(log_file, "a") as f:
            f.write(f"Iter {count}: {cmd}\n")

        # Execute training
        exit_code = os.system(cmd)
        
        if exit_code != 0:
            print(f"Warning: Iteration {count} failed with exit code {exit_code}")

        # Short cooldown to ensure file handles/GPU memory are cleared
        time.sleep(10)

if __name__ == "__main__":
    main()