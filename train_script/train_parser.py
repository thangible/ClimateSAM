import argparse
import configargparse
import wandb

def parse():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='input_config', help='config file path')
    parser.add_argument(
        '--exp_dir', default='./exp', type=str,
        help="The directory to save the best checkpoint file. Default to be ./exp"
    )
    parser.add_argument(
        '--data_dir', default='./data', type=str,
        help="The directory that the datasets are placed. Default to be ./data"
    )
    parser.add_argument(
        '--num_workers', default=None, type=int,
        help="The num_workers argument used for the training and validation dataloaders. "
             "Default to be 1 for one-shot and 4 for 16- and full-shot."
    )
    parser.add_argument(
        '--train_bs', default= 16, type=int,
        help="The batch size for the training dataloader. Default to be 1 for one-shot and 4 for 16- and full-shot."
    )
    parser.add_argument(
        '--val_bs', default=None, type=int,
        help="The batch size for the validation dataloader. Default to be 1 for one-shot and 4 for 16- and full-shot."
    )

    parser.add_argument(
        '--shot_num', default=None, type=int, choices=[1, 16],
        help="The number of your target setting. For one-shot please give --shot_num 1. "
             "For 16-shot please give --shot_num 16. For full-shot please leave it blank. "
             "Default to be full-shot."
    )
    parser.add_argument(
        '--sam_type', default='vit_l', type=str, choices=['vit_b', 'vit_l', 'vit_h'],
        help='The type of the backbone SAM model. Default to be vit_l.'
    )
    
    parser.add_argument(
        '--max_epoch_num', default=50, type=int,
        help="The maximum number of epochs for training. Default is 50."
    )
    
    parser.add_argument(
        '--lr', default=1e-3, type=float,
        help="Learning rate for the optimizer. Default is 1e-3."
    )
    parser.add_argument(
        '--weight_decay', default=1e-4, type=float,
        help="Weight decay for the optimizer. Default is 1e-4."
    )
    parser.add_argument(
        '--save_model', action='store_true',
        help="Flag to save the best model. Default is False."
    )
    parser.add_argument(
        '--valid_per_epochs', default=1, type=int,
        help="Validation frequency in terms of epochs. Default is 1."
    )
    
    parser.add_argument(
        '--wandb', action='store_true',
        help="Flag to enable Weights & Biases logging. Default is False."
    )
    parser.add_argument(
        '--project_name', type=str, default="climate-sam",
        help="Project name for Weights & Biases logging."
    )
    parser.add_argument(
        '--run_name', type=str,
        help="Run name for Weights & Biases logging."
    )
    parser.add_argument(
        '--debugging', action='store_true',
        help="Flag to enable debugging mode. Default is False."
    )
    parser.add_argument(
        '--load_pretrained', action='store_true',
        help="Flag to load a pretrained model. Default is False."
    )

    parser.add_argument(
        '--pretrained_name', type=str,
        help="Name of the pretrained model to load."
    )
    
    parser.add_argument(
        '--bce_weight_ar', default=10, type=float,
        help="Weight for the BCE loss for AR. Default is 1.0."
    )
    
    parser.add_argument(
        '--bce_weight_tc', default=200, type=float,
        help="Weight for the BCE loss for TC. Default is 1.0."
    )
    
    parser.add_argument(
        '--gamma_ar', default=2, type=float,
        help="Gamma parameter for the Focal loss for AR. Default is 2.0."
    )
    
    parser.add_argument(
        '--gamma_tc', default=2, type=float,
        help="Gamma parameter for the Focal loss for TC. Default is 2.0."
    )
    
    parser.add_argument(
        '--alpha_ar', default=0.9, type=float,
        help="Alpha parameter for the Focal loss for AR. Default is 0.25."
    )
    
    parser.add_argument(
        '--alpha_tc', default=0.98, type=float,
        help="Alpha parameter for the Focal loss for TC. Default is 0.25."
    )
    
    parser.add_argument(
        '--alpha_ar_tversky', default=0.1, type=float,
        help="Alpha parameter for the Tversky loss for AR. Default is 0.1."
    )

    parser.add_argument(
        '--beta_ar_tversky', default=0.9, type=float,
        help="Beta parameter for the Tversky loss for AR. Default is 0.9."
    )

    parser.add_argument(
        '--alpha_tc_tversky', default=0.8, type=float,
        help="Alpha parameter for the Tversky loss for TC. Default is 0.8."
    )

    parser.add_argument(
        '--beta_tc_tversky', default=0.2, type=float,
        help="Beta parameter for the Tversky loss for TC. Default is 0.2."
    )

    parser.add_argument(
        '--phase', default='1', type=int, choices=[1, 2, 3, 4],
        help="Phase 1 for image_encoder, Phase 2 for prompt generator, phase 3 for input adapter, phase 4 for all"
    )
    
    parser.add_argument(
        '--image_encoder_mlp_ratio', default=0.25, type=float,
        help="MLP ratio for the image encoder. Default is 0.25."
    )
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                   help='Number of steps to accumulate gradients before updating')

    parser.add_argument(
        '--focal_weight', default=1.0, type=float,
        help="Weight for the Focal loss. Default is 1.0."
    )

    parser.add_argument(
        '--tversky_weight', default=3.0, type=float,
        help="Weight for the Tversky loss. Default is 3.0."
    )
    
    parser.add_argument(
        '--bce_weight', default=1.0, type=float,
        help="Weight for the BCE loss. Default is 1.0."
    )
    
    parser.add_argument(
        '--augmented', action='store_true',
        help="Flag to enable data augmentation. Default is False."
    )
    
    parser.add_argument(
        '--positive_point_num', default=5, type=int,
        help="Number of positive points for prompt generation. Default is 5."
    )

    parser.add_argument(
        '--negative_point_num', default=5, type=int,
        help="Number of negative points for prompt generation. Default is 5."
    )

    parser.add_argument(
        '--prompt_type', default='point', type=str,
        help="Type of prompt to generate. Default is 'point'."
    )
    
    parser.add_argument(
        '--fuse_channels', default=128, type=int,
        help="Number of channels to fuse in the prompt generator. Default is 128."
    )
    
    parser.add_argument(
        '--data_type', type=str, default='ar', 
        help="Type of data to train on. Choices are 'ar' for atmospheric rivers and 'tc' for tropical cyclones."
    )

    args = parser.parse_args()

    return args