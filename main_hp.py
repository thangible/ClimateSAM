import os
import time
import random

def main():
    epoch_num = 15
    total_samples = 100
    log_file = "hparam_search_log_official.txt"

    # SEARCH: Tversky TC
    tversky_configs_tc = [
        (0.3, 0.7),   # Thesis Optimal
        (0.2, 0.8),   # High Recall
        (0.4, 0.6),   # Precision leaning
        (0.05, 0.95)  # STRESS TEST
    ]

    # SEARCH: Tversky AR
    tversky_configs_ar = [
        (0.5, 0.5),   # Thesis Optimal
        (0.4, 0.6),   # High Recall
        (0.6, 0.4),   # Precision leaning
        (0.3, 0.7)    # STRESS TEST
    ]

    # SEARCH: Focal TC
    focal_configs_tc = [
        (0.99, 5.0),  # Thesis Optimal
        (0.95, 5.0),  # Recent Search Winner
        (0.99, 2.5),  # Stability focus
        (0.995, 8.0)  # STRESS TEST
    ]
    
    # SEARCH: Focal AR
    focal_configs_ar = [
        (0.90, 2.0),  # Thesis Optimal
        (0.95, 3.0),  # Recent Search Winner
        (0.80, 1.0),  # High performance secondary
        (0.85, 5.0)   # STRESS TEST
    ]

    # SEARCH: Focal Weight
    fw_config = [1, 10, 50]

    # PRE-POPULATED: (atc_t, aar_t, atc_f, gtc_f, aar_f, gar_f, fw)
    # This prevents re-running the 16 combinations from the previous CSV
    executed = set()

    count = 0
    while count < total_samples:
        atc_t, btc_t = random.choice(tversky_configs_tc)
        aar_t, bar_t = random.choice(tversky_configs_ar)
        atc_f, gtc_f = random.choice(focal_configs_tc)
        aar_f, gar_f = random.choice(focal_configs_ar)
        fw = random.choice(fw_config)
        
        tw, bcew = 1.0, 0.0
        combo = (atc_t, aar_t, atc_f, gtc_f, aar_f, gar_f, fw)
        
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
            f"--focal_weight {fw} --tversky_weight {tw} --bce_weight {bcew} "
            f"--max_epoch_num {epoch_num} --run_name hparam_v4_{count}"
        )

        is_bad = "STRESS_TEST" if (atc_t == 0.05 or gtc_f == 8.0 or gar_f == 5.0 or fw == 50) else "VALID_SEARCH"
        
        print(f"\n--- Iteration {count}/{total_samples} [{is_bad}] ---")
        print(f"Tversky: TC({atc_t}) AR({aar_t}) | Focal: TC({atc_f}, {gtc_f}) AR({aar_f}, {gar_f}) | FW: {fw}")
        
        with open(log_file, "a") as f:
            f.write(f"Iter {count} [{is_bad}]: {cmd}\n")

        os.system(cmd)
        time.sleep(5)

if __name__ == "__main__":
    main()