import os
import random
import xarray as xr
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
from .transforms  import Compose, HorizontalFlip, VerticalFlip, RandomHorizontalRoll
from .climatenet_util import extract_point_and_bbox_prompts_from_climatenet_mask
from model.prompt.cgnet import CGNetPrompter

class ClimateDataset(Dataset):
    def __init__(self, data_dir, train_flag=True, reset_flag=False, augmented=False, generate_prompt=False, enlarge_ratio = [0,0], prompt_type = None, **prompt_kwargs):
        """
        Parameters:
            data_dir (str): Directory containing the .nc files.
            train_flag (bool): Whether the dataset is used for training.
            transforms (list): A list of transforms to apply.
            prompt_kwargs: Additional keyword arguments for prompt generation.
        """
        self.train_path = os.path.join(data_dir, "train")
        self.test_path = os.path.join(data_dir, "test")
        sub_dir = self.train_path if train_flag else self.test_path

        self.files = [os.path.join(sub_dir, f) for f in sorted(os.listdir(sub_dir)) if f.endswith(".nc")]
        if len(self.files) == 0:
            raise ValueError(f"No .nc files found in directory: {sub_dir}")

        self.train_flag = train_flag
        self.augmented = augmented
        self.cg_prompter = None
        self.generate_prompt = generate_prompt
        self.transforms = Compose([HorizontalFlip(p = 0.5), 
                                  VerticalFlip(p = 0.5), 
                                  RandomHorizontalRoll(p = 0.5, shift_limit=(0.5))]) if self.train_flag and self.augmented else None
        self.enlarge_ratio = enlarge_ratio
        self.prompt_type = prompt_type

        # self.transforms = None

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
        self.spatial_priors_path = os.path.join(data_dir, "spatial_priors.npy")
        self.cgnet_fields = {
            "TMQ": {"mean": 19.21859, "std": 15.81723},
            "U850": {"mean": 1.55302, "std": 8.29764},
            "V850": {"mean": 0.25413, "std": 6.23163},
            "PSL": {"mean": 100814.414, "std": 1461.2227},
        }
        
        # Load or compute mean/std for all variables (used for z-normalization + scaling)
        if os.path.exists(self.mean_std_path) and not self.reset_flag:
            self.mean_std_dict = np.load(self.mean_std_path, allow_pickle=True).item()
        else:
            self.mean_std_dict = self.calculate_stats()

        prompt_type_str = prompt_type if prompt_type is not None else "random"
        print("ClimateDataset initialized with {} samples. Enlarge Ratio {}, Prompt Type {}".format(len(self.files), self.enlarge_ratio, prompt_type_str))

    # def get_cg_prompter(self, worker_args, device):
    #     cg_prompter = CGNetPrompter(weights_path='pretrained/weights_cgnet.pth', device=device, worker_args=worker_args)
    #     self.cg_prompter = cg_prompter
    #     print("CGNet prompter initialized.")
        
        
        

    def __getitem__(self, index):
        # Use filename as the unique index name.
        file_path = self.files[index]
        index_name = os.path.basename(file_path)

        # Load the .nc file.
        dataset = xr.load_dataset(file_path)
        
        # CG INPUT
        cgnet_input = dataset[list(self.cgnet_fields)].to_array()
        for variable_name, stats in self.cgnet_fields.items():
            var = cgnet_input.sel(variable=variable_name).values
            var -= stats['mean']
            var /= stats['std']
        cgnet_input = cgnet_input.transpose('time', 'variable', 'lat', 'lon').values
        
        # SAM INPUT
        sam_input = dataset.to_array().sel(variable=self.variables).values.squeeze()
        # z-normalize per-channel using precomputed mean/std and scale to [0,255]
        sam_input = self.z_normalize_and_scale(sam_input)
        mask = self.get_labels(dataset)  # see function below
        
        # Apply transforms (if any)
        if self.transforms:
            mask_before_shape = mask.shape
            data_before_shape = sam_input.shape 
            transform_dict = self.transforms(sam_input, mask, cgnet_input)
            sam_input, mask, cgnet_input = transform_dict['input'], transform_dict['mask'], transform_dict['extra']
            assert sam_input.shape == data_before_shape, f"Data shape changed after transforms: {sam_input.shape} vs {data_before_shape}"
            assert mask.shape == mask_before_shape, f"Mask shape changed after transforms: {mask.shape} vs {mask_before_shape}"

        # rgb_image = self.to_image(dataset, var_1='TMQ', var_2='U850', var_3='V850')
        # Return a dictionary that matches the expected format.

        
        if self.generate_prompt:
            if self.prompt_type is not None:
                prompt_type = self.prompt_type
            else:
                prompt_type = random.choice(['bbox', 'point', 'mask']) if self.train_flag else random.choice(['point', 'bbox'])
            prompt_dict = extract_point_and_bbox_prompts_from_climatenet_mask(mask=mask, prompt_type=prompt_type, enlarge_ratio=self.enlarge_ratio)
            
        else:
            prompt_dict = {
                'ar_point_prompts': (None, None),
                'tc_point_prompts': (None, None),
                'ar_bbox_prompts': None,
                'tc_bbox_prompts': None,
                'ar_mask_prompts': None,
                'tc_mask_prompts': None,
                'ar_object_masks' : None,
                'tc_object_masks' : None,
                'ar_centroids': None,
                'tc_centroids': None,
            }
        # self.prompt_check(prompt_dict)
        
        return {
            "input": sam_input,
            'cgnet_input': cgnet_input,
            "gt_mask": mask,     # binary mask.
            "index_name": index_name,
            
            "ar_point_prompts": prompt_dict['ar_point_prompts'],
            "tc_point_prompts": prompt_dict['tc_point_prompts'],
            
            "ar_bbox_prompts": prompt_dict['ar_bbox_prompts'],
            "tc_bbox_prompts": prompt_dict['tc_bbox_prompts'],
            
            "ar_mask_prompts": prompt_dict['ar_mask_prompts'],
            "tc_mask_prompts": prompt_dict['tc_mask_prompts'],
            
            "ar_object_masks" : prompt_dict['ar_object_masks'],
            "tc_object_masks" : prompt_dict['tc_object_masks'],
            "ar_centroids": prompt_dict['ar_centroids'],
            "tc_centroids": prompt_dict['tc_centroids'],
        }
            
        
        
    def calculate_stats(self):
        """
        Calculate per-variable mean and std across all files and save to self.mean_std_path.
        This is robust to datasets that include a time dimension.
        Returns a dict with keys 'mean', 'std', 'norm_min', 'norm_max' (all arrays length = len(self.variables)).
        """
        # If file exists and reset_flag is False, load it
        if os.path.exists(self.mean_std_path) and not self.reset_flag:
            stats = np.load(self.mean_std_path, allow_pickle=True).item()
            return stats

        means = []
        stds = []

        for file in self.files:
            try:
                ds = xr.load_dataset(file)
                data = ds.to_array().sel(variable=self.variables).values.squeeze()

                # Determine where the channel dimension is and compute per-channel mean/std
                # Possible shapes: (channels, H, W) or (time, channels, H, W)
                if data.ndim == 3 and data.shape[0] == len(self.variables):
                    # (channels, H, W)
                    per_channel_mean = np.mean(data, axis=(1, 2))
                    per_channel_std = np.std(data, axis=(1, 2))
                elif data.ndim == 4 and data.shape[1] == len(self.variables):
                    # (time, channels, H, W) -> average over time and spatial dims
                    per_channel_mean = np.mean(data, axis=(0, 2, 3))
                    per_channel_std = np.std(data, axis=(0, 2, 3))
                else:
                    # Fallback: try to move channel axis to front if possible
                    chan_axis = None
                    for i, s in enumerate(data.shape):
                        if s == len(self.variables):
                            chan_axis = i
                            break
                    if chan_axis is None:
                        raise ValueError(f"Unable to locate channel axis for file {file} with shape {data.shape}")
                    data_moved = np.moveaxis(data, chan_axis, 0)
                    per_channel_mean = np.mean(data_moved, axis=tuple(range(1, data_moved.ndim)))
                    per_channel_std = np.std(data_moved, axis=tuple(range(1, data_moved.ndim)))

                means.append(per_channel_mean)
                stds.append(per_channel_std)
            except Exception as e:
                # Skip files that fail to load and continue
                print(f"Warning: failed to process {file} for stats ({e}) - skipping")
                continue

        if len(means) == 0:
            raise RuntimeError("No valid files found to compute mean/std")

        mean_arr = np.mean(np.stack(means, axis=0), axis=0)
        std_arr = np.mean(np.stack(stds, axis=0), axis=0)

        # compute dataset-level normalized min/max (one extra pass)
        eps = 1e-12
        norm_mins = []
        norm_maxs = []
        for file in self.files:
            try:
                ds = xr.load_dataset(file)
                data = ds.to_array().sel(variable=self.variables).values.squeeze()

                # move channel axis to front so shape becomes (channels, ...)
                if data.ndim == 3 and data.shape[0] == len(self.variables):
                    data_moved = data
                elif data.ndim == 4 and data.shape[1] == len(self.variables):
                    # average over time first to reduce variability then keep channels,H,W
                    data_moved = np.mean(data, axis=0)
                else:
                    chan_axis = None
                    for i, s in enumerate(data.shape):
                        if s == len(self.variables):
                            chan_axis = i
                            break
                    if chan_axis is None:
                        continue
                    data_moved = np.moveaxis(data, chan_axis, 0)

                mean_b = mean_arr[:, np.newaxis, np.newaxis]
                std_b = std_arr[:, np.newaxis, np.newaxis] + eps
                normalized = (data_moved - mean_b) / std_b

                # per-channel min/max across remaining dims
                mins = normalized.min(axis=tuple(range(1, normalized.ndim)))
                maxs = normalized.max(axis=tuple(range(1, normalized.ndim)))
                norm_mins.append(mins)
                norm_maxs.append(maxs)
            except Exception:
                continue

        if len(norm_mins) > 0:
            norm_min_arr = np.min(np.stack(norm_mins, axis=0), axis=0)
            norm_max_arr = np.max(np.stack(norm_maxs, axis=0), axis=0)
        else:
            # sensible default clipping range for z-scores
            norm_min_arr = np.full_like(mean_arr, -3.0)
            norm_max_arr = np.full_like(mean_arr, 3.0)

        result = {"mean": mean_arr, "std": std_arr, "norm_min": norm_min_arr, "norm_max": norm_max_arr}
        np.save(self.mean_std_path, result)
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
        Z-normalize per-channel using dataset mean/std, then scale per-channel to [0,255]
        using dataset-level normalized min/max for consistent mapping.
        Input shape expected: (channels, H, W)
        """
        eps = 1e-12
        mean = self.mean_std_dict["mean"][:, np.newaxis, np.newaxis]
        std = self.mean_std_dict["std"][:, np.newaxis, np.newaxis] + eps
        normalized = (data - mean) / std

        # Use stored dataset-level normalized min/max if available
        norm_min = self.mean_std_dict.get("norm_min")
        norm_max = self.mean_std_dict.get("norm_max")
        if norm_min is not None and norm_max is not None:
            norm_min_b = norm_min[:, np.newaxis, np.newaxis]
            norm_max_b = norm_max[:, np.newaxis, np.newaxis]
        else:
            # fallback to per-sample min/max
            norm_min_b = normalized.min(axis=(1, 2), keepdims=True)
            norm_max_b = normalized.max(axis=(1, 2), keepdims=True)

        scaled = np.clip((normalized - norm_min_b) / (norm_max_b - norm_min_b + eps), 0.0, 1.0) * 255.0
        return scaled.astype(np.uint8)

    def __len__(self):
        return len(self.files)
        
    def get_file_names(self, index_name):
        return os.path.splitext(index_name)[0] # file name without the .nc extension,
    
    def get_variables(self):
        return self.variables
        

    # def to_image(self, dataset, var_1='TMQ', var_2='U850', var_3='V850'):
    #     """
    #     Convert the dataset into an RGB image using three selected variables.
    #     """
    #     # Assume dataset.to_array() gives an array with a "variable" dimension.
    #     features = dataset.to_array()
    #     # Select the variables (you may need to adjust this if your dataset is structured differently).
    #     var1 = features.sel(variable=var_1).values
    #     var2 = features.sel(variable=var_2).values
    #     var3 = features.sel(variable=var_3).values

    #     # Ensure variables are 2D (H, W) before stacking
    #     var1 = np.squeeze(var1)
    #     var2 = np.squeeze(var2)
    #     var3 = np.squeeze(var3)
        
    #     # Stack the channels to form an RGB image.
    #     rgb_image = np.stack([var1, var2, var3], axis=-1)
    #     # Normalize the image to 0-255.
    #     rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    #     rgb_image = (rgb_image * 255).astype(np.uint8)

    #     # Remove the batch dimension if it exists (1, H, W, C) → (H, W, C)
    #     if rgb_image.shape[0] == 1:
    #         rgb_image = np.squeeze(rgb_image, axis=0) 
        

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
    
    def generate_grid_prompts(self, mask, num_points):
        """
        Generate uniformly spaced grid points as prompts within the mask.
        """
        # Get the shape of the mask
        height, width = mask.shape
        padding = 10
        
        prompts = []

        # Generate grid points
        y_coords = np.linspace(padding, height - padding - 1, num_points, dtype=int)
        x_coords = np.linspace(padding, width - padding - 1, num_points, dtype=int)

        for y in y_coords:
            for x in x_coords:
                prompts.append([(x, y)])

        points = torch.from_numpy(np.stack(prompts, axis=0)).to(torch.float32)
        labels = torch.ones(points.shape[0], dtype=torch.float32).unsqueeze(1)  

        return (points, labels)
    
    

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
        batch_dict['cgnet_input'] = torch.cat([torch.from_numpy(cg_inp).float() for cg_inp in batch_dict['cgnet_input']], dim=0)
        # batch_dict['cgnet_input'] =  torch.Tensor(xr.concat(batch_dict['cgnet_input'], dim='time').values)
    
        return batch_dict

    def get_hemisphere_extremal_points(self):
        """
        Finds the top-most and bottom-most points for Tropical Cyclones (TC) and 
        Atmospheric Rivers (AR) in the Northern and Southern hemispheres.
        
        Returns:
            dict: Nested dictionary containing pixel coordinates (y, x), 
                  geographic coordinates (lat, lon), and the source filename.
        """
        # Initialize the output dictionary
        extremes = {
            'tc': {
                'north': {'top': None, 'bottom': None},
                'south': {'top': None, 'bottom': None}
            },
            'ar': {
                'north': {'top': None, 'bottom': None},
                'south': {'top': None, 'bottom': None}
            }
        }

        # Helper to pick a representative x for a given y (mean x at that y)
        def rep_point_at_y(pt_array, y_val):
            xs = pt_array[pt_array[:, 0] == y_val, 1]
            x_rep = int(xs.mean()) if xs.size else int(pt_array[0, 1])
            return int(y_val), x_rep

        print(f"Scanning {len(self.files)} files for extremal points...")

        for file_path in self.files:
            dataset = xr.load_dataset(file_path)
            
            # Get the label mask and lat/lon arrays
            m = self.get_labels(dataset)
            if hasattr(m, 'numpy'):
                m = m.numpy()
            m = m.astype(np.int32)
            
            lats = dataset.lat.values
            lons = dataset.lon.values
            
            H, W = m.shape
            half_row = H // 2

            for cls_val, cls_key in zip((1, 2), ('tc', 'ar')):
                pts = np.argwhere(m == cls_val)  # rows (y), cols (x)
                if pts.size == 0:
                    continue

                # north = top half (rows < half_row), south = bottom half (rows >= half_row)
                north_pts = pts[pts[:, 0] < half_row]
                south_pts = pts[pts[:, 0] >= half_row]

                # --- NORTH hemisphere updates ---
                if north_pts.size:
                    y_top = int(north_pts[:, 0].min())
                    y_bot = int(north_pts[:, 0].max())
                    yt, xt = rep_point_at_y(north_pts, y_top)
                    yb, xb = rep_point_at_y(north_pts, y_bot)

                    # Update most top (smallest y) in north
                    if extremes[cls_key]['north']['top'] is None or yt < extremes[cls_key]['north']['top']['pixel'][0]:
                        extremes[cls_key]['north']['top'] = {
                            'pixel': (yt, xt),
                            'latlon': (float(lats[yt]), float(lons[xt])),
                            'file': os.path.basename(file_path)
                        }
                    
                    # Update most bottom (largest y) in north
                    if extremes[cls_key]['north']['bottom'] is None or yb > extremes[cls_key]['north']['bottom']['pixel'][0]:
                        extremes[cls_key]['north']['bottom'] = {
                            'pixel': (yb, xb),
                            'latlon': (float(lats[yb]), float(lons[xb])),
                            'file': os.path.basename(file_path)
                        }

                # --- SOUTH hemisphere updates ---
                if south_pts.size:
                    y_top_s = int(south_pts[:, 0].min())
                    y_bot_s = int(south_pts[:, 0].max())
                    yt_s, xt_s = rep_point_at_y(south_pts, y_top_s)
                    yb_s, xb_s = rep_point_at_y(south_pts, y_bot_s)

                    # Update most top (smallest y) in south
                    if extremes[cls_key]['south']['top'] is None or yt_s < extremes[cls_key]['south']['top']['pixel'][0]:
                        extremes[cls_key]['south']['top'] = {
                            'pixel': (yt_s, xt_s),
                            'latlon': (float(lats[yt_s]), float(lons[xt_s])),
                            'file': os.path.basename(file_path)
                        }
                    
                    # Update most bottom (largest y) in south
                    if extremes[cls_key]['south']['bottom'] is None or yb_s > extremes[cls_key]['south']['bottom']['pixel'][0]:
                        extremes[cls_key]['south']['bottom'] = {
                            'pixel': (yb_s, xb_s),
                            'latlon': (float(lats[yb_s]), float(lons[xb_s])),
                            'file': os.path.basename(file_path)
                        }
            
            dataset.close()

        return extremes
    
    def calculate_spatial_priors(self):
        """
        Creates a 2D heatmap tracking the exact locations where TCs and ARs occur
        across the entire training dataset.
        """
        if os.path.exists(self.spatial_priors_path) and not self.reset_flag:
            return np.load(self.spatial_priors_path, allow_pickle=True).item()

        print("Building spatial occurrence heatmaps from training data... (This only happens once)")
        
        # Get shape from the first file
        ds = xr.load_dataset(self.files[0])
        m = self.get_labels(ds)
        shape = m.shape
        ds.close()

        tc_heatmap = np.zeros(shape, dtype=np.int32)
        ar_heatmap = np.zeros(shape, dtype=np.int32)

        # Strictly use training files to prevent validation data leakage
        train_files = [os.path.join(self.train_path, f) for f in sorted(os.listdir(self.train_path)) if f.endswith(".nc")]

        from tqdm import tqdm
        for file in tqdm(train_files, desc="Mapping Historical TC/AR Hotspots"):
            try:
                ds = xr.load_dataset(file)
                m = self.get_labels(ds)
                
                # Accumulate occurrences
                tc_heatmap += (m == 1).astype(np.int32)
                ar_heatmap += (m == 2).astype(np.int32)
                ds.close()
            except Exception as e:
                print(f"Skipped {file} during prior building: {e}")
                continue
        
        priors = {'tc': tc_heatmap, 'ar': ar_heatmap}
        np.save(self.spatial_priors_path, priors)
        return priors
    def generate_smart_grid_prompts(self, class_type, grid_size=(32, 32), jitter_amount=0.5, min_occurrences=1):
        """
        Generates a grid restricted to historical hotspots, with random variations.
        
        Args:
            class_type (str): 'tc' or 'ar'
            grid_size (tuple): The base uniform grid density (y_steps, x_steps)
            jitter_amount (float): How much random shift to apply (0.0 to 1.0). 
                                   0.5 means a point can shift up to half a grid cell.
            min_occurrences (int): Drop grid points where historical occurrences are less than this.
        """
        
        if os.path.exists(self.spatial_priors_path) and not self.reset_flag:
            spatial_priors =  np.load(self.spatial_priors_path, allow_pickle=True).item()
        else:
            spatial_priors = self.calculate_spatial_priors()
        
        heatmap = spatial_priors[class_type]
        H, W = heatmap.shape

        # Create uniform grid baseline
        y_steps = np.linspace(0, H - 1, grid_size[0])
        x_steps = np.linspace(0, W - 1, grid_size[1])

        # Calculate pixel distance between grid points for jitter scaling
        dy = (H - 1) / (grid_size[0] - 1)
        dx = (W - 1) / (grid_size[1] - 1)

        valid_points = []

        for y in y_steps:
            for x in x_steps:
                y_int, x_int = int(y), int(x)
                
                # Filter Step: Does this point fall in a valid hotspot?
                if heatmap[y_int, x_int] >= min_occurrences:
                    
                    # Randomness Step: Add jitter
                    y_jitter = y + random.uniform(-dy * jitter_amount, dy * jitter_amount)
                    x_jitter = x + random.uniform(-dx * jitter_amount, dx * jitter_amount)

                    # Clamp to image boundaries
                    y_final = np.clip(y_jitter, 0, H - 1)
                    x_final = np.clip(x_jitter, 0, W - 1)

                    valid_points.append([x_final, y_final])

        if not valid_points:
            return None, None

        # Format for SAM: (N, 1, 2)
        points = np.array(valid_points)[:, np.newaxis, :] 
        points = torch.from_numpy(points).to(torch.float32)
        
        # Positive point labels
        labels = torch.ones(points.shape[0], dtype=torch.float32).unsqueeze(1)

        return points, labels
