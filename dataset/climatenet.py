import os
import random
import xarray as xr
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
from .transforms  import Compose
from .climatenet_util import extract_point_and_bbox_prompts_from_climatenet_mask

class ClimateDataset(Dataset):
    def __init__(self, data_dir, train_flag=True, reset_flag=False, transforms=None,  **prompt_kwargs):
        """
        Parameters:
            data_dir (str): Directory containing the .nc files.
            train_flag (bool): Whether the dataset is used for training.
            transforms (list): A list of transforms to apply.
            prompt_kwargs: Additional keyword arguments for prompt generation.
        """
        train_path = os.path.join(data_dir, "train")
        test_path = os.path.join(data_dir, "test")
        sub_dir = train_path if train_flag else test_path

        self.files = [os.path.join(sub_dir, f) for f in sorted(os.listdir(sub_dir)) if f.endswith(".nc")]
        if len(self.files) == 0:
            raise ValueError(f"No .nc files found in directory: {sub_dir}")

        self.train_flag = train_flag
        self.transforms = Compose(transforms) if transforms else None
        
        # Store prompt generation parameters.
        self.prompt_kwargs = prompt_kwargs
        
        prompt_kwargs = prompt_kwargs.copy()
        shot_num = prompt_kwargs.pop("shot_num", None)
        if shot_num is not None:
            self.files = self.files[:shot_num]
            
            
        self.reset_flag = reset_flag
        self.climatenet_label = None
        self.variables = ['TMQ', 'U850', 'V850', 'UBOT', 'VBOT', 'QREFHT', 'PS', 'PSL', 
                        'T200', 'T500', 'PRECT', 'TS', 'TREFHT', 'Z1000', 'Z200', 'ZBOT']
        
        
        # Define the path to save the mean and std values.
        self.mean_std_path = os.path.join(data_dir, "mean_std.npy")
        self.mean_std_dict = self.calculate_stats() 

    def __getitem__(self, index):
        # Use filename as the unique index name.
        file_path = self.files[index]
        index_name = os.path.basename(file_path)

        # Load the .nc file.
        dataset = xr.load_dataset(file_path)
        prompt_kwargs = self.prompt_kwargs.copy() 
        
        # Generate the binary mask from the dataset.
        mask = self.get_labels(dataset)  # see function below
        
        # Generate inputs
        data = dataset.to_array().sel(variable=self.variables).values.squeeze()
                
        # Apply Z-normalization to the data
        data = self.minmax_per_channel_to_image(data)
        
        rgb_image = self.to_image(dataset, var_1='TMQ', var_2='U850', var_3='V850')
        # Return a dictionary that matches the expected format.
        
        
        prompt_type = random.choice(['bbox', 'point', 'mask']) if self.train_flag else random.choice(['point', 'bbox'])
        prompt_dict = extract_point_and_bbox_prompts_from_climatenet_mask(mask = mask, prompt_type = prompt_type)
        self.prompt_check(prompt_dict)
        
        
        
        return {
            "input": data,
            "gt_mask": mask,     # binary mask.
            "index_name": index_name,
            
            "ar_point_prompts": prompt_dict['ar_point_prompts'],
            "tc_point_prompts": prompt_dict['tc_point_prompts'],
            
            "ar_bbox_prompts": prompt_dict['ar_bbox_prompts'],
            "tc_bbox_prompts": prompt_dict['tc_bbox_prompts'],
            
            "ar_mask_prompts": prompt_dict['ar_mask_prompts'],
            "tc_mask_prompts": prompt_dict['tc_mask_prompts'],
            
            "ar_object_masks" : prompt_dict['ar_object_masks'],
            "tc_object_masks" : prompt_dict['tc_object_masks']
        }
        
    def prompt_check(self, prompt_dict):
        """
        Check if prompt coordinates are within valid image dimensions.
        Print warnings if x > 1152 or y > 768.
        """
        max_x = 1152
        max_y = 768
        
        # Check AR point prompts
        if prompt_dict['ar_point_prompts'][0] is not None:
            ar_points = prompt_dict['ar_point_prompts'][0]
            for i, point in enumerate(ar_points):
                x, y = point[0]  # point is in format [[x, y]]
                if x > max_x or y > max_y:
                    print(f"AR point prompt {i} out of bounds: x={x}, y={y} (max: x={max_x}, y={max_y})")
        
        # Check TC point prompts
        if prompt_dict['tc_point_prompts'][0] is not None:
            tc_points = prompt_dict['tc_point_prompts'][0]
            for i, point in enumerate(tc_points):
                x, y = point[0]  # point is in format [[x, y]]
                if x > max_x or y > max_y:
                    print(f"TC point prompt {i} out of bounds: x={x}, y={y} (max: x={max_x}, y={max_y})")
        
        # Check AR bbox prompts
        if prompt_dict['ar_bbox_prompts'] is not None:
            ar_bboxes = prompt_dict['ar_bbox_prompts']
            for i, bbox in enumerate(ar_bboxes):
                x1, y1, x2, y2 = bbox[0]  # bbox is in format [[x1, y1, x2, y2]]
                if x1 > max_x or x2 > max_x or y1 > max_y or y2 > max_y:
                    print(f"AR bbox prompt {i} out of bounds: x1={x1}, y1={y1}, x2={x2}, y2={y2} (max: x={max_x}, y={max_y})")
        
        # Check TC bbox prompts
        if prompt_dict['tc_bbox_prompts'] is not None:
            tc_bboxes = prompt_dict['tc_bbox_prompts']
            for i, bbox in enumerate(tc_bboxes):
                x1, y1, x2, y2 = bbox[0]  # bbox is in format [[x1, y1, x2, y2]]
                if x1 > max_x or x2 > max_x or y1 > max_y or y2 > max_y:
                    print(f"TC bbox prompt {i} out of bounds: x1={x1}, y1={y1}, x2={x2}, y2={y2} (max: x={max_x}, y={max_y})")
            
        
        
    def calculate_stats(self):
        """
        Calculate the mean and std of the data across all the files.
        """
        if os.path.exists(self.mean_std_path) and self.reset_flag is False:
            
            stats = np.load(self.mean_std_path, allow_pickle=True).item()
            if self.train_flag:
                print(f"Loading mean/std from {self.mean_std_path}")
                print(f"AR Ratio (negative/positive): {stats['ar_ratio']}, TC Ratio: {stats['tc_ratio']}")
            return stats

        print("Calculating mean/std from scratch...")
        means = []
        stds = []
        ar_ratios = []
        tc_ratios = []
        
        for file in self.files:
            dataset = xr.load_dataset(file)
            data = dataset.to_array().sel(variable=self.variables).values.squeeze()
            means.append(np.mean(data, axis=(1,2)))  # Mean for each of the 16 channels
            stds.append(np.std(data, axis=(1,2)))    # Std for each of the 16 channels
            
            mask = dataset['LABELS'].values
            ar_mask = mask == 2
            tc_mask = mask == 1
            ar_ones_count = np.sum(ar_mask == 1) + 1
            tc_ones_count = np.sum(tc_mask == 1) + 1
            ar_zeros_count = np.sum(ar_mask == 0)
            tc_zeros_count = np.sum(tc_mask == 0)
            ar_ratio = ar_zeros_count / (ar_ones_count)
            tc_ratio = tc_zeros_count / (tc_ones_count)
            ar_ratios.append(ar_ratio)
            tc_ratios.append(tc_ratio)
        
        # Calculate the overall mean and std for each channel across all files
        mean_dict = np.mean(means, axis=0)
        std_dict = np.mean(stds, axis=0)
        mean_ar_ratio = np.median(ar_ratios)
        mean_tc_ratio = np.median(tc_ratios)
        
        result = {"mean": mean_dict, "std": std_dict, "ar_ratio": mean_ar_ratio, "tc_ratio": mean_tc_ratio}
        np.save(self.mean_std_path, result)
        
        # Return a dictionary with channel-wise mean and std
        return result
    
    def z_normalize(self, data):
        """
        Normalize the data using Z-normalization: (X - mean) / std
        """
        mean = self.mean_std_dict["mean"]
        std = self.mean_std_dict["std"]
        
        # Z-normalization for each channel
        normalized_data = (data - mean) / std
        
        return normalized_data
    
    def z_normalize_and_scale(self, data):
        """
        Normalize the data using Z-normalization: (X - mean) / std, then scale it to [0, 255].
        """
        # Z-normalize the data
        mean = self.mean_std_dict["mean"][:, np.newaxis, np.newaxis]
        std = self.mean_std_dict["std"][:, np.newaxis, np.newaxis]
        normalized_data = (data - mean) / std

        # Scale to [0, 255]
        normalized_data_min = normalized_data.min(axis=(0, 1), keepdims=True)
        normalized_data_max = normalized_data.max(axis=(0, 1), keepdims=True)
        
        # Clip values to ensure they stay within the range [0, 1] before multiplying by 255
        epsilon = 1e-8  # Small value to prevent division by zero
        scaled_data = np.clip((normalized_data - normalized_data_min) / (normalized_data_max - normalized_data_min + epsilon), 0, 1) * 255
        
        # Convert to uint8 for image representation
        scaled_data = scaled_data.astype(np.uint8)
        
        return scaled_data

    def __len__(self):
        return len(self.files)
        
    def get_file_names(self, index_name):
        return os.path.splitext(index_name)[0] # file name without the .nc extension,
    
    def get_variables(self):
        return self.variables
        

    def to_image(self, dataset, var_1='TMQ', var_2='U850', var_3='V850'):
        """
        Convert the dataset into an RGB image using three selected variables.
        """
        # Assume dataset.to_array() gives an array with a "variable" dimension.
        features = dataset.to_array()
        # Select the variables (you may need to adjust this if your dataset is structured differently).
        var1 = features.sel(variable=var_1).values
        var2 = features.sel(variable=var_2).values
        var3 = features.sel(variable=var_3).values

        # Ensure variables are 2D (H, W) before stacking
        var1 = np.squeeze(var1)
        var2 = np.squeeze(var2)
        var3 = np.squeeze(var3)
        
        # Stack the channels to form an RGB image.
        rgb_image = np.stack([var1, var2, var3], axis=-1)
        # Normalize the image to 0-255.
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        rgb_image = (rgb_image * 255).astype(np.uint8)

        # Remove the batch dimension if it exists (1, H, W, C) → (H, W, C)
        if rgb_image.shape[0] == 1:
            rgb_image = np.squeeze(rgb_image, axis=0) 

        return rgb_image
    
    def minmax_per_channel_to_image(self, data):
        """
        Normalize the data using min-max normalization.
        """
        # Min-max normalization
        data_min = data.min(axis=(1, 2), keepdims=True)
        data_max = data.max(axis=(1, 2), keepdims=True)
        normalized_data = (data - data_min) / (data_max - data_min + 1e-8)
        normalized_data = (normalized_data * 255).astype(np.uint8)
        return normalized_data

    def get_labels(self, dataset, label_name= None):
        """
        Extract and binarize the segmentation mask from the dataset.
        """
        # if label_name == 'cyclone':
        #     mask_description = 1
        # elif label_name == 'river':
        #     mask_description = 2
        # else:
        #     raise ValueError(f"Unknown label name: {label_name}")
            
        mask = dataset['LABELS'].values
        # if label_name is not None:
        #     mask = (mask == mask_description).astype(np.uint8)  # Convert to a binary mask.
        # mask = np.ascontiguousarray(mask)
        # mask = cv2.UMat(mask)  # Ensure the mask is a numpy array
        # print("Mask shape:", mask.shape)
        return mask


    @classmethod
    def collate_fn(cls, batch):
        """
        Custom collate function to batch ClimateDataset samples.
        Handles image/mask tensors without assuming same spatial shape.
        """
        batch_dict = {key: [] for key in batch[0].keys()}
        
        # Fix: Single loop to process all samples
        for sample in batch:
            if sample is not None:
                for key, value in sample.items():
                    batch_dict[key].append(value)
    
        # Convert inputs and masks to tensors
        batch_dict['input'] = torch.stack([torch.from_numpy(inp).float() for inp in batch_dict['input']])
        batch_dict['gt_mask'] = [torch.from_numpy(mask).long() for mask in batch_dict['gt_mask']]
    
        return batch_dict

