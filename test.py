from model.prompt.cgnet_module import CGNetModule
import torch

model = CGNetModule(classes=3, channels=4)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

def fmt(n):
    return f"{n:,} ({n/1e6:.3f} M)"

print("Model:", model.__class__.__name__)
print("Total params:     ", fmt(total_params))
print("Trainable params: ", fmt(trainable_params))
print("Non-trainable:    ", fmt(total_params - trainable_params))