from .cgnet_module import CGNetModule
import torch
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
from .create_prompt_from_mask import extract_point_and_bbox_prompts_from_pred_masks


class CGNetPrompter:
    def __init__(self, weights_path, device, worker_args):
        self.cgnet_model = CGNetModule(classes=3, channels=4)
        self.cgnet_model.load_state_dict(torch.load(weights_path, map_location=device))
        self.cgnet_model.to(device) 
        self.exp_dir = worker_args.exp_dir
        self.device = device
        self.optimizer = torch.optim.Adam(self.cgnet_model.parameters(), lr=1e-4)

    def train(self, dataloader, epochs):
        self.cgnet_model.train()
        for epoch in range(1, epochs):
            print(f'Epoch {epoch}:')
            epoch_loader = tqdm(dataloader)
            aggregate_cm = np.zeros((3,3))

            for batch in epoch_loader:
                
                features = batch['cgnet_input'].to(device=self.device, dtype=torch.float32)
                # labels = [x.to(device=self.device, dtype=torch.float32) for x in batch['gt_mask']]
                labels = torch.stack([x.to(self.device, dtype=torch.long) for x in batch['gt_mask']])
                outputs = torch.softmax(self.cgnet_model(features), 1)

                # Update training CM
                predictions = torch.max(outputs, 1)[1]
                aggregate_cm += get_cm(predictions, labels, 3)

                # Pass backward
                loss = jaccard_loss(outputs, labels)
                epoch_loader.set_description(f'Loss: {loss.item()}')
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad() 

            if epoch == 1:
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
                
                
            print('Epoch stats:')
            print(aggregate_cm)
            ious = get_iou_perClass(aggregate_cm)
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
        prompt_dict = extract_point_and_bbox_prompts_from_pred_masks(preds=pred_masks, device=self.device, prompt_type=prompt_type)

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