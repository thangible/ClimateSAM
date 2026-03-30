import random
from .cgnet_module import CGNetModule
import torch
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
from .create_prompt_from_mask import extract_point_and_bbox_prompts_from_pred_masks
from .prompt_maker import PromptMaker
import wandb

class CGNetPrompter:
    def __init__(self, weights_path, device, worker_args):
        self.cgnet_model = CGNetModule(classes=3, channels=4)
        self.cgnet_model.load_state_dict(torch.load(weights_path, map_location=device))
        self.cgnet_model.to(device) 
        self.exp_dir = worker_args.exp_dir
        self.device = device
        self.optimizer = torch.optim.Adam(self.cgnet_model.parameters(), lr=1e-4)
        self.prompt_maker = PromptMaker(prompt_type=worker_args.prompt_type, positive_point_num=worker_args.positive_point_num, negative_point_num=worker_args.negative_point_num)
        # WandB config passed from train script
        self.wandb = getattr(worker_args, 'wandb', False)
        self.run_name = getattr(worker_args, 'run_name', None)

    def train(self, dataloader, epochs):
        self.cgnet_model.train()
        best_ious = 0
        for epoch in range(1, epochs):
            print(f'Epoch {epoch}:')
            epoch_loader = tqdm(dataloader)
            aggregate_cm = np.zeros((3,3))
            epoch_loss_sum = 0.0
            epoch_loss_count = 0

            for batch in epoch_loader:
                
                features = batch['cgnet_input'].to(device=self.device, dtype=torch.float32)
                # labels = [x.to(device=self.device, dtype=torch.float32) for x in batch['gt_mask']]
                labels = torch.stack([x.to(self.device, dtype=torch.long) for x in batch['gt_mask']])
                outputs = torch.softmax(self.cgnet_model(features), 1)

                # Update training CM
                predictions = torch.max(outputs, 1)[1]
                aggregate_cm += get_cm(predictions, labels, 3)

                # Pass backward
                loss = dice_bce_loss(outputs, labels)
                epoch_loader.set_description(f'Loss: {loss.item()}')
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad() 

                # accumulate loss for epoch logging
                try:
                    epoch_loss_sum += float(loss.item())
                except:
                    epoch_loss_sum += loss.detach().cpu().item()
                epoch_loss_count += 1

            avg_epoch_loss = epoch_loss_sum / epoch_loss_count if epoch_loss_count > 0 else 0.0

            if epoch % 5 == 1:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(labels[0].cpu().numpy(), cmap='viridis')
                axes[0].set_title('Ground Truth')
                axes[1].imshow(predictions[0].cpu().numpy(), cmap='viridis')
                axes[1].set_title('Predictions')
                plt.show()
                plot_path = os.path.join(self.exp_dir, f"epoch_{epoch}_plot.png")
                fig.savefig(plot_path)
                plt.close(fig)
                print(f"Saved image at {plot_path}")

                # Log example image to WandB
                if self.wandb:
                    try:
                        wandb.log({"train/example_prediction": wandb.Image(plot_path)}, step=epoch)
                    except Exception as e:
                        print(f"WandB image log failed: {e}")
                
                
            print('Epoch stats:')
            print(aggregate_cm)
            ious = get_iou_perClass(aggregate_cm)[1:]
            mean_iou = ious.mean()
            if mean_iou > best_ious and epoch > 10:
                best_ious = mean_iou
                self.save_model()
                print(f"New best model saved with mean IoU: {best_ious}")
                if self.wandb:
                    try:
                        save_path = os.path.join(self.exp_dir, f"cgnet_weight.pth")
                        wandb.save(save_path)
                    except Exception as e:
                        print(f"WandB save failed: {e}")

            # Log epoch scalars to WandB
            if self.wandb:
                log_dict = {
                    "train/avg_loss": avg_epoch_loss,
                    "train/mean_iou": float(mean_iou),
                    "epoch": epoch,
                    "train/iou_class_1": float(ious[0]),
                    "train/iou_class_2": float(ious[1])
                }
                try:
                    wandb.log(log_dict, step=epoch)
                except Exception as e:
                    print(f"WandB scalar log failed: {e}")

            print('IOUs: ', ious, ', mean: ', ious.mean())
            
    def save_model(self,):
        '''
        Save model weights and config to a directory.
        '''
        save_path = os.path.join(self.exp_dir, f"cgnet_weight.pth")
        torch.save(self.cgnet_model.state_dict(), save_path)
        
        
    @torch.no_grad()
    def get_aux_mask(self, bacth_input):
        '''
        Given an input image, return the auxiliary mask from CGNet.
        '''
        self.cgnet_model.eval()
        with torch.no_grad():
            output = self.cgnet_model(bacth_input)
            outputs = torch.softmax(output, 1)
        preds = torch.max(outputs, 1)[1]
        return preds
    
    @torch.no_grad()
    def get_prompts(self, batch_input, prompt_type='point', noisy_mask_threshold=0.5):
        '''
        Given an input image, return the prompts from CGNet.
        '''
        pred_masks = self.get_aux_mask(batch_input)
        prompt_dict = self.prompt_maker.make_prompts(pred_masks)
        return prompt_dict



    
    
            
            

def jaccard_loss(logits, true, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    true_1_hot = torch.eye(num_classes, device=true.device)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

def dice_bce_loss(logits, true, eps=1e-7):
    """Computes the Dice loss combined with Binary Cross-Entropy (BCE) loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        loss: the combined Dice and BCE loss.
    """
    num_classes = logits.shape[1]
    true_1_hot = torch.eye(num_classes, device=true.device)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    
    # Dice Loss
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    
    # BCE Loss
    bce_loss = F.cross_entropy(logits, true.squeeze(1))
    
    # Combined Loss
    loss = (1 - dice_loss) + bce_loss
    return loss

def get_iou_perClass(confM):
    """
    Takes a confusion matrix confM and returns the IoU per class
    """
    unionPerClass = confM.sum(axis=0) + confM.sum(axis=1) - confM.diagonal()
    iouPerClass = np.zeros(3)
    for i in range(0,3):
        if unionPerClass[i] == 0:
            iouPerClass[i] = 1
        else:
            iouPerClass[i] = confM.diagonal()[i] / unionPerClass[i]
    return iouPerClass
        
def get_cm(pred, gt, n_classes=3):
    cm = np.zeros((n_classes, n_classes))
    for i in range(len(pred)):
        pred_tmp = pred[i].int()
        gt_tmp = gt[i].int()

        for actual in range(n_classes):
            for predicted in range(n_classes):
                is_actual = torch.eq(gt_tmp, actual)
                is_pred = torch.eq(pred_tmp, predicted)
                cm[actual][predicted] += len(torch.nonzero(is_actual & is_pred))
            
    return cm