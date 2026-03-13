import os
import torch
import numpy as np
from types import SimpleNamespace

import pytest

# Import the main_worker from train.py
from train import main_worker

# Define a dummy dataset that returns two samples.
class DummyDataset:
    def __init__(self, **kwargs):
        self.collate_fn = lambda batch: batch
        self.data = [self.get_dummy_item() for _ in range(2)]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    def get_dummy_item(self):
        # Create a dummy input of shape (3, 32, 32)
        input_tensor = torch.rand(3, 32, 32)
        # Create a dummy ground truth mask: use 0 (background), 1 and 2 for two classes.
        # This dummy mask has one pixel for each class.
        mask = torch.zeros(32, 32, dtype=torch.uint8)
        mask[10, 10] = 1  # class 1 (TC)
        mask[20, 20] = 2  # class 2 (AR)
        return {'input': input_tensor, 'gt_masks': mask, 'index_name': 'dummy'}

# Define a dummy ClimateSAM model that returns dummy outputs.
def dummy_climatesam(*args, **kwargs):
    class DummyModel(torch.nn.Module):
        def forward(self, x, ar_point_prompts=None, tc_point_prompts=None, ar_bbox_prompts=None, tc_bbox_prompts=None):
            batch_size = x.shape[0]
            # Create dummy outputs with shape (batch, 1, 32, 32)
            dummy_ar = torch.randn(batch_size, 1, 32, 32)
            dummy_tc = torch.randn(batch_size, 1, 32, 32)
            # Return (tc, ar, images)
            return dummy_tc, dummy_ar, x.unsqueeze(0) if x.ndim == 3 else x
    return DummyModel()

# Dummy metrics: simply returns a fixed mIoU.
def dummy_streamsegmetrics(*args, **kwargs):
    class DummyMetrics:
        def update(self, preds, targets, names):
            pass
        def compute(self):
            return [{'Mean Foreground IoU': 0.5}]
        def reset(self):
            pass
    return DummyMetrics()

# Dummy batch_to_cuda that moves tensors to the given device.
def dummy_batch_to_cuda(batch, device):
    batch['input'] = batch['input'].to(device)
    batch['gt_masks'] = batch['gt_masks'].to(device)
    return batch

@pytest.fixture(autouse=True)
def prepare_env(monkeypatch, tmp_path):
    # Patch ClimateDataset in train.py to our DummyDataset
    monkeypatch.setattr("train.ClimateDataset", lambda **kwargs: DummyDataset())
    # Patch ClimateSAM in train.py to our dummy model
    monkeypatch.setattr("train.ClimateSAM", dummy_climatesam)
    # Patch StreamSegMetrics in train.py to our dummy metrics
    monkeypatch.setattr("train.StreamSegMetrics", lambda class_names: dummy_streamsegmetrics())
    # Patch batch_to_cuda so that it just moves data to the device.
    monkeypatch.setattr("train.batch_to_cuda", dummy_batch_to_cuda)
    # Disable wandb logging (if necessary)
    monkeypatch.setattr("train.wandb", type("DummyWandb", (), {"log": lambda *args, **kwargs: None, "init": lambda *args, **kwargs: None, "save": lambda *args, **kwargs: None})())

def test_main_worker():
    # Create a dummy args object with the required attributes.
    args = SimpleNamespace(
        max_epoch_num=1,
        shot_num=1,
        train_bs=1,
        val_bs=1,
        num_workers=0,
        wandb=False,
        valid_per_epochs=1,
        data_dir="dummy_dir",
        used_gpu=["0"],
        gpu_num=1,
        run_name="test_run",
        exp_dir=str(os.path.join(os.getcwd(), "test_outputs")),
        sam_type="dummy",
        lr=1e-3,
        weight_decay=1e-4,
        dist_url="tcp://127.0.0.1:29500"
    )
    # Create necessary output directories.
    os.makedirs(os.path.join(args.exp_dir, args.run_name, "images"), exist_ok=True)
    
    # Run main_worker with worker_id 0
    main_worker(0, args)
    
    # Check that the dummy model produced images saved in the output directory.
    images_dir = os.path.join(args.exp_dir, args.run_name, "images")
    # Since the validation is run during epoch 1 and step 0, expect at least one image file.
    saved_files = os.listdir(images_dir)
    assert len(saved_files) >= 1, "No validation images were saved."