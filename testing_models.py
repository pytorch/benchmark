from statistics import mean, median, stdev
import torch
import matplotlib.pyplot as plt
import importlib
import torchbenchmark
# model = build_model("1.5B").cuda()
KB = 2**20
# models = ['torchbenchmark.models.hf_Bert.Model', 'torchbenchmark.models.hf_BertLarge.Model', \
#     'torchbenchmark.models.hf_GPT2_large.Model', 'torchbenchmark.models.hf_T5_large.Model', \
#         'torchbenchmark.models.timm_vision_transformer_large.Model', 'torchbenchmark.models.hf_GPT2.Model', \
#             'torchbenchmark.models.hf_T5.Model']
models = ['torchbenchmark.models.hf_T5_xl.Model']
for model_name in models:
    pos = model_name.rfind(".")
    pos2 = model_name.rfind(".", 0, pos)
    m_name = model_name[pos2+1: pos]
    print(m_name)

    param_tensor_sizes= []
    module = importlib.import_module(model_name[:pos])
    model_class = getattr(module, model_name[(pos+1):])
    model = model_class("train", 'cuda')
    print(model_name)
    ele_count = 0
    ten_count = 0
    for param in model.model.parameters():
        ele_count += param.numel()
        param_tensor_sizes.append((param.numel()*4)/KB)
        ten_count += 1
    avg_size = mean(param_tensor_sizes)
    std_dev = stdev(param_tensor_sizes)
    median_size = median(param_tensor_sizes)
    max_size = max(param_tensor_sizes)
    # plt.hist(param_tensor_sizes)
    # plt.xlabel("Tensor Sizes (MBs)")
    # plt.ylabel('Count')
    # plt.savefig(f'{model_name}_distribution.png')
    # plt.close()

    print(f'Parameters: {ele_count} Tensors: {ten_count}')
    print(f'Median Size: {median_size} Avg Size: {avg_size} Std Dev: {std_dev} Max Size: {max_size}')

    print(torch.cuda.memory_allocated())
    loss = model.model(**model.example_inputs).loss
    print(torch.cuda.memory_allocated())
    loss.backward()
    model.optimizer.step()
    print(torch.cuda.memory_allocated())

    loss = model.model(**model.example_inputs).loss
    print(torch.cuda.memory_allocated())



# model.cfg.optimizer.zero_grad()
# amp_context = model.amp_context
# with amp_context():
#     output = model.model(model.example_inputs)
# if isinstance(output, tuple):
#     output = output[0]
# target = model._gen_target(output.shape[0])
# loss = model.cfg.loss(output, target)
# loss.backward()
# model.cfg.optimizer.step()
# print(torch.cuda.memory_allocated())
# with amp_context():
#     output = model.model(model.example_inputs)
# if isinstance(output, tuple):
#     output = output[0]
# target = model._gen_target(output.shape[0])
# loss = model.cfg.loss(output, target)
# print(torch.cuda.memory_allocated())
