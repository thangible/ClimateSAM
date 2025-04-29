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
    args = parser.parse_args()

    return args