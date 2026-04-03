import os
import time
import itertools

def main():
    # Basic training parameters
    epoch_num = 26
    log_file = "search_lr_decay_log.txt"
    run_name = "LR_Decay_Search"
    

    # SEARCH SPACE
    base_lrs = [1e-3, 5e-4, 1e-4]
    
    # Factor relative to base_lr (e.g., 1e-3 * 1.0 = 1e-3)
    adapter_factors = [1e-3, 1e-4, 1e-5] 
    
    # Factor relative to base_lr (e.g., 1e-3 * 0.01 = 1e-5)
    encoder_factors = [1e-1, 1e-2, 1e-3, 1e-4]

    # Generate all combinations
    all_combinations = list(itertools.product(base_lrs, adapter_factors, encoder_factors))
    
    # Filter: Ensure Adapter LR > Encoder LR
    valid_combinations = [
        (lr, af, ef) for lr, af, ef in all_combinations if af < ef
    ]

    total_samples = len(valid_combinations)
    print(f"Starting LR and Decay Grid Search...")
    print(f"Total valid runs: {total_samples}")

    for count, (lr, af, ef) in enumerate(valid_combinations):
        # Calculate actual LRs for the name
        # actual_adapter_lr = lr * af
        # actual_encoder_lr = lr * ef
        
        run_name = f"LR_{lr}_Adap_{af}_Enc_{ef}"
        
        # Construct the command
        # Note: Ensure your train_adaptation.py is updated to accept --adapter_lr and --encoder_lr
        cmd = (
            f"python train_script/official/train_adaptation.py --config hp_mode --sam_type vit_b --image_encoder_mlp_ratio 1 --wandb "
            f"--lr {lr} "
            f"--adapter_decay {af} "
            f"--encoder_decay {ef} "
            f"--max_epoch_num {epoch_num} --run_name {run_name} --hp_mode "
            f"--smooth_label"
        )

        print(f"\n--- Iteration {count+1}/{total_samples} ---")
        print(f"Base LR: {lr} | Adapter LR: {af} | Encoder LR: {ef}")
        
        with open(log_file, "a") as f:
            f.write(f"Iter {count+1}: {cmd}\n")

        # Execute
        os.system(cmd)
        
        # Brief cooldown for GPU memory clearance
        time.sleep(5)
        
        

if __name__ == "__main__":
    main()