# TODO: Is any of this leftover code useful?

# cpu_pin(cpu)

# class Benchmark(object):
#    # default_params = []
#    # params = make_params()
#    # param_names = ['config']

#
#
# def start_stats(common_name, framework_name, fname, mag, count, tv):
#     status = ""
#     status += "tag: {:<15}".format(common_name)
#     status += "fname: {:<15}".format(framework_name)
#     status += "{:<15}".format(fname)
#     status += " memory: {:<10}".format("O(10^" + str(mag) + ")KB")
#     status += " count: {:<6}".format(count)
#     status += " size: {:<20}".format(list(tv.size()))
#     status += " stride: {:<60}".format(list(map(lambda x: "{:>7}".format(x), list(tv.stride()))))
#     status += " numel: {:<9}".format(tv.numel())
#     return status
#
# def finish_stats(dtype, dim, elapsed):
#     status = ""
#     status += " type: {:<18}".format(dtype)
#     status += " dim: {:<5}".format(dim)
#     status += " elapsed: {:8.4f}".format(elapsed)
#     return status
#
# def lambda_benchmark(common_name, types, fun, name, framework_name, cast):
#     goal_size = 1000
#     onek = 1000
#     goal = onek * 1000 * goal_size
#     for dtype in types:
#         for cont in [True, False]:
#             for trans in [True, False]:
#                 for mag in [4, 5]:
#                     for dim in [4]:
#                         size_ = int(onek * math.pow(10, mag))
#                         count = goal / size_
#                         tv = make_tensor(size_, dtype, cont, 3, trans)
#                         status = start_stats(common_name, framework_name, name, mag, count, tv)
#                         gc.collect()
#                         gc.collect()
#                         fun(tv)
#                         gc.collect()
#                         gc.collect()
#                         tstart = timer()
#                         for _ in range(count):
#                             fun(tv)
#                         elapsed = timer() - tstart
#                         print(status + finish_stats(dtype, dim, elapsed))
#                         gc.collect()
#                         gc.collect()

# class over(object):
#     def __init__(self, *args):
#         self.values = args
#
#
#
#
# def make_params(**kwargs):
#     keys = list(kwargs.keys())
#     iterables = [kwargs[k].values if isinstance(kwargs[k], over) else (kwargs[k],) for k in keys]
#     all_values = list(product(*iterables))
#     param_dicts = [AttrDict({k: v for k, v in zip(keys, values)}) for values in all_values]
#     return [param_dicts]

#     # NOTE: subclasses should call prepare instead of setup
#     def setup(self, params):
#         for k, v in self.default_params.items():
#             params.setdefault(k, v)
#         self.prepare(params)
#
#
# def get_env_pytorch_examples():
#     pytorch_examples_home = os.environ.get('EXAMPLES_HOME')
#     if pytorch_examples_home is None:
#         print('EXAMPLES_HOME not found')
#         sys.exit()
#
#     return pytorch_examples_home
#
#
# def execution(cmd, log_path):
#     gc.collect()
#
#     # logging
#     log_file = open(log_path, "w+")
#     log_file.write(cmd)
#     log_file.write('\n')
#
#     exec_command = shlex.split(cmd)
#     proc = subprocess.Popen(exec_command, stdout=log_file, stderr=subprocess.STDOUT)
#     proc.wait()
#     return_code = proc.returncode
#     log_file.close()
#
#     log_file = open(log_path, 'r+')
#
#     if return_code == 0:
#         acc = parse_accuracy(log_file)
#     else:
#         acc = ('NA', 'NA')
#
#     return acc
#
#
# def parse_accuracy(log_file):
#     output_data = log_file.readlines()
#     _, _, prec1, _, prec2 = output_data[-2].split()
#     return (prec1, prec2)
#
#
# def config_runs(model, no_iter):
#     iters = [i for i in range(no_iter)]
#     if model == 'all':
#         model = model_names
#
#     return list(itertools.product(model, iters))
#
#
# def cmd_string(examples_home, model, data_path):
#     lr = 0.1
#     if model in ['alexnet', 'vgg11', 'vgg11_bn', 'vgg13_bn',
#                  'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']:
#         lr = 0.01
#
#     cmd = ' '.join(['python3', examples_home, '-a', model, '--lr', str(lr), data_path])
#     return cmd
