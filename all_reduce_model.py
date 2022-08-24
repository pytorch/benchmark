
from collections import defaultdict
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
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
nodes = [i for i in range(1,25)]
dir_name = '/shared/sanket/logs/all_reduce/'
#------------BOX PLOTS -------------------------------
# nodes = [1,2,4,8,16,24,32]
# for net_type in network_types:
#     for num_nodes in nodes:
#         fig, ax = plt.subplots(figsize=(7,5))
#         num_tasks = num_nodes * 8
#         rank_latencies = defaultdict(lambda: defaultdict(list))
#         for rank in range(num_tasks):
#             latencies = defaultdict(list)
#             filename = f"{dir_name}all_red_{net_type}_{num_tasks}_{rank}.data"
#             print(filename)
#             with open(filename, 'r', encoding='UTF-8') as file:
#                 while (line := file.readline().rstrip()):
#                     vals = [x for x in line.split(",")]
#                     data_size = int(vals[0])
#                     latency = float(vals[1])
#                     latencies[data_size].append(latency)
#             rank_latencies[rank] = latencies
#         c_latencies = []
#         for d_size in data_sizes:
#             meta_list = []
#             for rank in range(num_tasks):
#                 meta_list.append(rank_latencies[rank][d_size])
#             t_meta_list = list(map(list, zip(*meta_list)))
#             if(d_size == 5):
#                 t_meta_list.pop(0)
#             comm_latency = [min(l) for l in t_meta_list]
#             straggle_latency = [max(l) for l in t_meta_list]
#             # straggle_latency = [a - b for a,b in zip(straggle_latency, comm_latency)]
#             # print(f"Data Size: {d_size}")
#             # print(f"Comm Latency: {comm_latency}")
#             # print(f"Straggle Latency: {straggle_latency}")
#             c_latencies.append(straggle_latency)
#         ax.boxplot(c_latencies, showfliers=False)
        
#         ax.set_xticks([r+1 for r in range(0,45) if(r%5==4)])
#         ax.set_xticklabels([data_sizes[r] for r in range(0,45) if(r%5==4)])
#         plt.xlabel("Tensor Size (MBs)")
#         plt.ylabel("All Reduce Latency (ms)")
#         plt.title(f"All Reduce Communication Latency for {num_tasks} GPUs")
#         fig.savefig(f"all_red_box_{num_nodes}_{net_type}_straggle.png")

#-------------- SCATTER LINE PLOTS
markers = ["d", "v", "s", "*", "^", "o"]
colors= ['blue', 'orange', 'green', 'red', 'purple', 'brown']
nodes = [2,4,9,14,20,32]
for net_type in network_types:
    fig, ax = plt.subplots(figsize=(6,4))
    for num_nodes,marker,color in zip(nodes,markers,colors):

        num_tasks = num_nodes * 8
        rank_latencies = defaultdict(lambda: defaultdict(list))
        for rank in range(num_tasks):
            latencies = defaultdict(list)
            filename = f"{dir_name}all_red_{net_type}_{num_tasks}_{rank}.data"
            print(filename)
            with open(filename, 'r', encoding='UTF-8') as file:
                while (line := file.readline().rstrip()):
                    vals = [x for x in line.split(",")]
                    data_size = int(vals[0])
                    latency = float(vals[1])
                    latencies[data_size].append(latency)
            rank_latencies[rank] = latencies
        c_latencies = []
        for d_size in data_sizes:
            meta_list = []
            for rank in range(num_tasks):
                meta_list.append(rank_latencies[rank][d_size])
            t_meta_list = list(map(list, zip(*meta_list)))
            if(d_size == 5):
                t_meta_list.pop(0)
            comm_latency = [min(l) for l in t_meta_list]
            straggle_latency = [max(l) for l in t_meta_list]
            print(f"Data Size: {d_size}")
            print(f"Comm Latency: {comm_latency}")
            print(f"Straggle Latency: {straggle_latency}")
            # straggle_latency = [a - b for a,b in zip(straggle_latency, comm_latency)]
            print(f"Straggle Latency Diff: {straggle_latency}")
            c_latencies.append(mean(straggle_latency))
        ax.scatter(data_sizes[:20], c_latencies[:20], s=12, marker=marker, label=f"{num_nodes}  Nodes")
        z = np.polyfit(data_sizes[:20], c_latencies[:20], 2)
        p = np.poly1d(z)
        plt.plot(data_sizes[:20],p(data_sizes[:20]),ls="--", lw=0.5, color=color)
        plt.legend()
        plt.xlabel("Tensor Size (MBs)")
        plt.ylabel("All Reduce Latency (ms)")
        # ax.set_xticks([r+1 for r in range(0,45) if(r%5==4)])
        # ax.set_xticklabels([data_sizes[r] for r in range(0,45) if(r%5==4)])
    fig.savefig(f"all_red_line_straggle_lower_25_{net_type}.png")
#------------HISTOGRAMS -------------------------------
# hist_data_sizes = [25, 100, 500, 1000]
# hist_nodes = [2, 8, 16, 32]

# nodes = [2,8,16,32]
# for net_type in network_types:
#     for num_nodes in nodes:
#         fig, ax = plt.subplots(figsize=(10,7))
#         num_tasks = num_nodes * 8
#         rank_latencies = defaultdict(lambda: defaultdict(list))
#         for rank in range(num_tasks):
#             latencies = defaultdict(list)
#             filename = f"{dir_name}all_red_{net_type}_{num_tasks}_{rank}.data"
#             # print(filename)
#             with open(filename, 'r', encoding='UTF-8') as file:
#                 while (line := file.readline().rstrip()):
#                     vals = [x for x in line.split(",")]
#                     data_size = int(vals[0])
#                     latency = float(vals[1])
#                     latencies[data_size].append(latency)
#             rank_latencies[rank] = latencies
#         c_latencies = []
#         for d_size in data_sizes:
#             meta_list = []
#             for rank in range(num_tasks):
#                 meta_list.append(rank_latencies[rank][d_size])
#             t_meta_list = list(map(list, zip(*meta_list)))
#             if(d_size == 5):
#                 t_meta_list.pop(0)
#             comm_latency = [min(l) for l in t_meta_list]
#             straggle_latency = [max(l) for l in t_meta_list]
#             straggle_latency = [a - b for a,b in zip(straggle_latency, comm_latency)]
#             if(d_size in hist_data_sizes and num_nodes in hist_nodes):
#                 print(f"Nodes: {num_nodes} Data Size: {d_size} Net Type: {net_type}")
#                 print("Communication Latency")
#                 print(f"Mean: {mean(comm_latency)} Std Dev:{stdev(comm_latency)}")
#                 plt.hist(comm_latency,density=True, bins=5, alpha=0.6)
#                 xmin, xmax = plt.xlim()
#                 x = np.linspace(xmin, xmax, 100)
#                 p = norm.pdf(x, mean(comm_latency), stdev(comm_latency))
#                 plt.plot(x, p, 'k', linewidth=2)
#                 plt.xlabel("All Reduce Communication Latency (ms)")
#                 plt.ylabel('Frequency')
#                 plt.suptitle(f"Mean: {mean(comm_latency):.2f} Std Dev:{stdev(comm_latency):.2f}")
#                 plt.savefig(f'Comm_latency_{num_nodes}_{d_size}_{net_type}_distribution.png')
#                 plt.close()
#                 plt.hist(straggle_latency,density=True, bins=5, alpha=0.6)
#                 xmin, xmax = plt.xlim()
#                 x = np.linspace(xmin, xmax, 100)
#                 p = norm.pdf(x, mean(straggle_latency), stdev(straggle_latency))
#                 plt.plot(x, p, 'k', linewidth=2)
#                 plt.xlabel("All Reduce Straggle Latency (ms)")
#                 plt.ylabel('Frequency')
#                 plt.suptitle((f"Mean: {mean(straggle_latency):.2f} Std Dev:{stdev(straggle_latency):.2f}"))
#                 plt.savefig(f'Comm_latency_{num_nodes}_{d_size}_{net_type}_straggle_distribution.png')
#                 plt.close()

#                 print("Straggle Latency")
#                 print(f"Mean: {mean(straggle_latency)} Std Dev:{stdev(straggle_latency)}")

#             c_latencies.append(comm_latency)
        

