import os
import time
import random

def main():
    epoch_num = 15
    total_samples = 20
    log_file = "hparam_search_log_offical.txt"

    # FIXED: Optimal Tversky pairs from Table 4.6 [cite: 687, 688]
    atc_t, btc_t = 0.3, 0.7  # Fixed for high TC recall without instability
    aar_t, bar_t = 0.5, 0.5  # Fixed for balanced AR detection

    # SEARCH: Testing Alpha sensitivity while keeping Gamma near optimal values
    # TC: Testing if 0.99 is the ceiling or if 0.95/0.995 offers better balance
    focal_configs_tc = [
        (0.99, 5.0),   # Thesis Optimal [cite: 687]
        (0.95, 5.0),   # Testing lower Alpha (more background weight)
        (0.995, 4.0),   # Testing higher Alpha with slightly lower Gamma
        (0.995, 6.0)    # Testing higher Alpha with slightly higher Gamma for precision
    ]
    
    # AR: Testing if 0.90 is the ceiling or if 0.85/0.95 is better
    focal_configs_ar = [
        (0.90, 2.0),   # Thesis Optimal [cite: 687]
        (0.85, 2.0),   # Testing lower Alpha
        (0.95, 3.0),   # Testing higher Alpha/Gamma for precision
        (0.80, 1.0)    # Testing much lower Alpha to see if it helps with recall
    ]
    
    # FIXED: Focal weight 10 is the "sweet spot" for structural discovery 
    fw = 10 

    executed = set()
    count = 0
    while count < total_samples:
        aar_f, gar_f = random.choice(focal_configs_ar)
        atc_f, gtc_f = random.choice(focal_configs_tc)
        
        tw, bcew = 1.0, 0.0
        combo = (aar_f, gar_f, atc_f, gtc_f)
        
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
            f"--max_epoch_num {epoch_num} --run_name hparam_search_{count}"
        )

        print(f"\n--- Running Iteration {count}/{total_samples} ---")
        print(f"Testing Focal: TC(a={atc_f}, g={gtc_f}) | AR(a={aar_f}, g={gar_f})")
        
        with open(log_file, "a") as f:
            f.write(f"Iter {count}: {cmd}\n")

        os.system(cmd)
        time.sleep(10)

if __name__ == "__main__":
    main()