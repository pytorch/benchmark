import torch
from torchbenchmark.models import timm_vision_transformer_large
# model = build_model("1.5B").cuda()
model = timm_vision_transformer_large.Model("train", 'cuda', batch_size=16)
print(torch.cuda.memory_allocated())
count = 0
for param in model.model.parameters():
    count += param.numel()
print(count)
model.cfg.optimizer.zero_grad()
amp_context = model.amp_context
with amp_context():
    output = model.model(model.example_inputs)
if isinstance(output, tuple):
    output = output[0]
target = model._gen_target(output.shape[0])
loss = model.cfg.loss(output, target)
loss.backward()
model.cfg.optimizer.step()
print(torch.cuda.memory_allocated())
with amp_context():
    output = model.model(model.example_inputs)
if isinstance(output, tuple):
    output = output[0]
target = model._gen_target(output.shape[0])
loss = model.cfg.loss(output, target)
print(torch.cuda.memory_allocated())
