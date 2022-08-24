from collections import defaultdict
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import os.path

MB = 2**20
FLOAT_MB = 2**18

current_size = 0
size = 5*(2**18)
data_sizes = []
for i in range(45):
    if(i == 20):
        size = 20 * (FLOAT_MB)
    elif(i==30):
        size = 50 * (FLOAT_MB)
    current_size += size
    size_in_mb = (current_size * 4)// MB
    data_sizes.append(size_in_mb)

network_types = ['efa', 'ena']


dir_name = '/shared/sanket/logs/ddp_latency/'
models = ['hf_Bert', 'hf_BertLarge', 'hf_GPT2_large', 'hf_T5_large', 'timm_vision_transformer_large', 'hf_GPT2', 'hf_T5']

model_batch_size = {'hf_Bert': 32, 'hf_BertLarge': 16, 'hf_GPT2_large': 4, 'hf_T5_large': 4, \
        'timm_vision_transformer_large': 16, 'hf_GPT2': 24, 'hf_T5': 12}
markers = ["d", "v", "s", "*", "^", "o"]
colors= ['blue', 'orange', 'green', 'red', 'purple', 'brown']
# -----------------------------BWD/FULL Latency Scatter Plot------------------------------
# for model in models:
#     b_size = model_batch_size[model]
#     for net_type in network_types:
#         fig, ax = plt.subplots(figsize=(6,4))
#         # bwd_latencies = defaultdict(list)
#         bwd_latencies = {}
#         nodes = [i for i in range(1,25)]
#         for num_nodes in nodes:
#             num_tasks = num_nodes * 8
#             rank_latencies = []
#             SKIP_NODE = False
#             for rank in range(num_tasks):
#                 filename = f"{dir_name}ddp_{model}_{b_size}_{net_type}_{num_tasks}_{rank}.data"
#                 print(filename)
#                 if(os.path.isfile(filename)): 
#                     with open(filename, 'r', encoding='UTF-8') as file:
#                         while (line := file.readline().rstrip()):
#                             vals = [float(x) for x in line.split(",")]
#                             mean_bwd = vals[2]
#                             # mean_fwd = vals[0]
#                             # mean_opt = vals[4]
#                             # rank_latencies.append(mean_bwd+mean_fwd+mean_opt)
#                             rank_latencies.append(mean_bwd)
#                 else:
#                     SKIP_NODE = True
#                     break

#             if(not SKIP_NODE):
#                 bwd_latencies[num_nodes]= mean(rank_latencies)
#         ax.scatter(bwd_latencies.keys(), bwd_latencies.values(), s=15, marker="^")
#         x = [int(i) for i in bwd_latencies.keys()]
#         y = [float(i) for i in bwd_latencies.values()]
#         z = np.polyfit(x, y, 3)
#         p = np.poly1d(z)
#         plt.xlabel("Nodes")
#         plt.ylabel("Iteration Latency (ms)")
#         plt.title(f"DDP Backward Pass Latency for {model}")
#         plt.plot(nodes,p(nodes),ls="--", lw=0.5, color="red")
#         fig.savefig(f"ddp_{model}_{net_type}_bwd.png")

#-----------------------------FULL LATENCY STCKAED BARS-----------------------
nodes = [i for i in range(1,25)]
nodes = [1,2,4,8,12,16,20]
width = 0.35
for model in models:
    b_size = model_batch_size[model]
    for net_type in network_types:
        fig, ax = plt.subplots(figsize=(6,4))
        # bwd_latencies = defaultdict(list)
        bwd_latencies = {}
        fwd_latencies = {}
        opt_latencies = {}
        for num_nodes in nodes:
            num_tasks = num_nodes * 8
            rank_bwd_latencies = []
            rank_fwd_latencies = []
            rank_opt_latencies = []
            SKIP_NODE = False
            for rank in range(num_tasks):
                filename = f"{dir_name}ddp_{model}_{b_size}_{net_type}_{num_tasks}_{rank}.data"
                print(filename)
                if(os.path.isfile(filename)): 
                    with open(filename, 'r', encoding='UTF-8') as file:
                        while (line := file.readline().rstrip()):
                            vals = [float(x) for x in line.split(",")]
                            mean_bwd = vals[2]
                            mean_fwd = vals[0]
                            mean_opt = vals[4]
                            # rank_latencies.append(mean_bwd+mean_fwd+mean_opt)
                            rank_bwd_latencies.append(mean_bwd)
                            rank_fwd_latencies.append(mean_fwd)
                            rank_opt_latencies.append(mean_opt)
                else:
                    SKIP_NODE = True
                    break

            if(not SKIP_NODE):
                bwd_latencies[num_nodes]= mean(rank_bwd_latencies)
                fwd_latencies[num_nodes]= mean(rank_fwd_latencies)
                opt_latencies[num_nodes]= mean(rank_opt_latencies)
        labels = [str(n) for n in fwd_latencies.keys()]
        fwd_vals = [val for val in fwd_latencies.values()]
        bwd_vals = [val for val in bwd_latencies.values()]
        fb_vals = [val1+val2 for val1, val2 in zip(fwd_vals, bwd_vals)]
        opt_vals = [val for val in opt_latencies.values()]
        ax.bar(labels, fwd_vals, width=width, label='Fwd latency')
        ax.bar(labels, bwd_vals, width=width, bottom = fwd_vals, label='Bwd latency')
        ax.bar(labels, opt_vals, width=width, bottom = fb_vals, label='Opt latency')
        ax.set_xlabel("Num Nodes")
        ax.set_ylabel("Iteration Latency (ms)")
        ax.legend()
        ax.set_title(f"DDP Latency Breakdown for {model}")
        fig.savefig(f"ddp_stacked_{model}_{net_type}.png")