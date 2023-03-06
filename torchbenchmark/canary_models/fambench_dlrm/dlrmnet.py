import torch.nn as nn
import torch
import sys
import numpy as np
import itertools
from torch._ops import ops
from torch.nn.parameter import Parameter
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.scatter_gather import gather, scatter

# fambench imports
# projection
import project
# quotient-remainder trick
from tricks.qr_embedding_bag import QREmbeddingBag
# mixed-dimension trick
from tricks.md_embedding_bag import PrEmbeddingBag

class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        layers.training = self.requires_grad
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W)
            LL.weight.requires_grad = self.requires_grad
            LL.bias.data = torch.tensor(bt)
            LL.bias.requires_grad = self.requires_grad
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, weighted_pooling=None):
        # create_emb parameter description
        #
        # ln parameter:
        # ln is a list of all the tables' row counts. E.g. [10,5,16] would mean
        # table 0 has 10 rows, table 1 has 5 rows, and table 2 has 16 rows.
        #
        # m parameter (when m is a single value):
        # m is the length of all embedding vectors. All embedding vectors in all
        # embedding tables are created to be the same length. E.g. if ln were [3,2,5]
        # and m were 4, table 0 would be dimension 3 x 4, table 1 would be 2 x 4,
        # and table 2 would be 5 x 4.
        #
        # m parameter (when m is a list):
        # m is a list of all the tables' column counts. E.g. if m were [4,5,6] and
        # ln were [3,2,5], table 0 would be dimension 3 x 4, table 1 would be 2 x 5,
        # and table 2 would be 5 x 6.
        #
        # Key to remember:
        # embedding table i has shape: ln[i] rows, m columns, when m is a single value.
        # embedding table i has shape: ln[i] rows, m[i] columns, when m is a list.

        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            # torchbench: commment distributed
            # if ext_dist.my_size > 1:
            #     if i not in self.local_emb_indices:
            #         continue
            n = ln[i]

            # construct embedding operator
            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(
                    n,
                    m,
                    self.qr_collisions,
                    operation=self.qr_operation,
                    mode="sum",
                    sparse=True,
                )
            elif self.md_flag and n > self.md_threshold:
                base = max(m)
                _m = m[i] if n > self.md_threshold else base
                EE = PrEmbeddingBag(n, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=self.requires_grad)
            else:
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
                # initialize embeddings
                # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                # approach 1
                EE.weight.data = torch.tensor(W, requires_grad=self.requires_grad)
                # approach 2
                # EE.weight.data.copy_(torch.tensor(W))
                # approach 3
                # EE.weight = Parameter(torch.tensor(W),requires_grad=True)
            if weighted_pooling is None:
                v_W_l.append(None)
            else:
                v_W_l.append(torch.ones(n, dtype=torch.float32))
            emb_l.append(EE)
        return emb_l, v_W_l

    def __init__(
        self,
        args,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        proj_size=0,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
        weighted_pooling=None,
        loss_function="bce",
        learning_rate=0.1,
        use_gpu=False,
        use_fbgemm_gpu=False,
        fbgemm_gpu_codegen_pref="Split",
        inference_only=False,
        quantize_mlp_with_bit=False,
        quantize_emb_with_bit=False,
        use_torch2trt_for_mlp=False,
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):
            # save arguments
            self.ntables = len(ln_emb)
            self.m_spa = m_spa
            self.proj_size = proj_size
            self.use_gpu = use_gpu
            self.use_fbgemm_gpu = use_fbgemm_gpu
            self.fbgemm_gpu_codegen_pref = fbgemm_gpu_codegen_pref
            self.requires_grad = not inference_only
            self.ndevices_available = ndevices
            self.ndevices_in_use = ndevices
            self.output_d = 0
            self.add_new_weights_to_params = False
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params and not inference_only
            self.loss_threshold = loss_threshold
            self.loss_function = loss_function
            self.learning_rate = learning_rate
            if weighted_pooling is not None and weighted_pooling != "fixed":
                self.weighted_pooling = "learned"
            else:
                self.weighted_pooling = weighted_pooling
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold

            # torchbench: comment distributed
            # If running distributed, get local slice of embedding tables
            # if ext_dist.my_size > 1:
            #     n_emb = len(ln_emb)
            #     if n_emb < ext_dist.my_size:
            #         sys.exit(
            #             "only (%d) sparse features for (%d) devices, table partitions will fail"
            #             % (n_emb, ext_dist.my_size)
            #         )
            #     self.n_global_emb = n_emb
            #     self.n_local_emb, self.n_emb_per_rank = ext_dist.get_split_lengths(
            #         n_emb
            #     )
            #     self.local_emb_slice = ext_dist.get_my_slice(n_emb)
            #     self.local_emb_indices = list(range(n_emb))[self.local_emb_slice]

            # create operators
            self.emb_l, self.v_W_l = self.create_emb(m_spa, ln_emb, weighted_pooling)
            if self.weighted_pooling == "learned":
                self.v_W_l = nn.ParameterList(list(map(Parameter, self.v_W_l)))

            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

            if proj_size > 0:
                self.proj_l = project.create_proj(len(ln_emb) + 1, proj_size)

            # mlp quantization
            self.quantize_mlp_with_bit = quantize_mlp_with_bit
            self.use_torch2trt_for_mlp = use_torch2trt_for_mlp
            self.quantize_mlp_input_with_half_call = use_gpu and not args.use_torch2trt_for_mlp and args.quantize_mlp_with_bit == 16

            # embedding quantization
            self.quantize_emb = False
            self.emb_l_q = []
            self.quantize_bits = 32

            # fbgemm_gpu
            self.fbgemm_emb_l = []
            self.v_W_l_l = [self.v_W_l] if self.weighted_pooling else [None]

            self.interact_features_l = []

            # specify the loss function
            if self.loss_function == "mse":
                self.loss_fn = torch.nn.MSELoss(reduction="mean")
            elif self.loss_function == "bce":
                self.loss_fn = torch.nn.BCELoss(reduction="mean")
            elif self.loss_function == "wbce":
                self.loss_ws = torch.tensor(
                    np.fromstring(args.loss_weights, dtype=float, sep="-")
                )
                self.loss_fn = torch.nn.BCELoss(reduction="none")
            else:
                sys.exit(
                    "ERROR: --loss-function=" + self.loss_function + " is not supported"
                )

    def prepare_parallel_model(self, ndevices):
        device_ids = range(ndevices)
        # replicate mlp (data parallelism)
        self.bot_l_replicas = replicate(self.bot_l, device_ids)
        self.top_l_replicas = replicate(self.top_l, device_ids)

        # distribute embeddings (model parallelism)
        if self.weighted_pooling is not None:
            for k, w in enumerate(self.v_W_l):
                self.v_W_l[k] = Parameter(
                    w.to(torch.device("cuda:" + str(k % ndevices)))
                )
        if not self.use_fbgemm_gpu:
            for k, w in enumerate(self.emb_l):
                self.emb_l[k] = w.to(torch.device("cuda:" + str(k % ndevices)))
        else:
            from .fbgemm_embedding import fbgemm_gpu_emb_bag_wrapper
            self.fbgemm_emb_l, self.v_W_l_l = zip(
                *[
                    (
                        fbgemm_gpu_emb_bag_wrapper(
                            torch.device("cuda:" + str(k)),
                            self.emb_l[k::ndevices]
                            if self.emb_l
                            else self.emb_l_q[k::ndevices],
                            self.m_spa[k::ndevices]
                            if isinstance(self.m_spa, list)
                            else self.m_spa,
                            self.quantize_bits,
                            self.learning_rate,
                            self.fbgemm_gpu_codegen_pref,
                            self.requires_grad,
                        ),
                        self.v_W_l[k::ndevices] if self.weighted_pooling else None,
                    )
                    for k in range(ndevices)
                ]
            )
            self.add_new_weights_to_params = True
        self.interact_features_l = [self.nn_module_wrapper() for _ in range(ndevices)]

    # nn_module_wrapper is used to call functions concurrently across multi-gpus, using parallel_apply,
    # which requires an nn.Module subclass.
    class nn_module_wrapper(nn.Module):
        def __init__(self):
            super(DLRM_Net.nn_module_wrapper, self).__init__()
        def forward(self, E, x, ly):
            return E(x, ly)

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_emb(self, lS_o, lS_i):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        if self.use_fbgemm_gpu:
            # Deinterleave and reshape to 2d, so items are grouped by device
            # per row. Then parallel apply.
            ndevices = len(self.fbgemm_emb_l)
            lS_o_l = [lS_o[k::ndevices] for k in range(ndevices)]
            lS_i_l = [lS_i[k::ndevices] for k in range(ndevices)]
            ly = parallel_apply(
                self.fbgemm_emb_l, list(zip(lS_o_l, lS_i_l, self.v_W_l_l))
            )
            # Interleave and flatten to match non-fbgemm_gpu ly format.
            ly = [ly[i % ndevices][i // ndevices] for i in range(self.ntables)]
        else:
            ly = []
            for k, sparse_index_group_batch in enumerate(lS_i):
                sparse_offset_group_batch = lS_o[k]

                # embedding lookup
                # We are using EmbeddingBag, which implicitly uses sum operator.
                # The embeddings are represented as tall matrices, with sum
                # happening vertically across 0 axis, resulting in a row vector
                # E = emb_l[k]

                if self.v_W_l[k] is not None:
                    per_sample_weights = self.v_W_l[k].gather(
                        0, sparse_index_group_batch
                    )
                else:
                    per_sample_weights = None

                if self.quantize_emb:
                    if self.quantize_bits == 4:
                        E = ops.quantized.embedding_bag_4bit_rowwise_offsets
                    elif self.quantize_bits == 8:
                        E = ops.quantized.embedding_bag_byte_rowwise_offsets
                    QV = E(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )

                    ly.append(QV)
                else:
                    E = self.emb_l[k]
                    V = E(
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )

                    ly.append(V)

        # print(ly)
        return ly

    #  using quantizing functions from caffe2/aten/src/ATen/native/quantized/cpu
    def quantize_embedding(self, bits):

        n = len(self.emb_l)
        self.emb_l_q = [None] * n
        for k in range(n):
            if bits == 4:
                self.emb_l_q[k] = ops.quantized.embedding_bag_4bit_prepack(
                    self.emb_l[k].weight
                )
            elif bits == 8:
                self.emb_l_q[k] = ops.quantized.embedding_bag_byte_prepack(
                    self.emb_l[k].weight
                )
            elif bits == 16:
                self.emb_l_q[k] = self.emb_l[k].half().weight
            else:
                return
        self.emb_l = None
        self.quantize_emb = True
        self.quantize_bits = bits

    def interact_features(self, x, ly):

        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            if self.proj_size > 0:
                R = project.project(T, x, self.proj_l)
            else:
                Z = torch.bmm(T, torch.transpose(T, 1, 2))
                # append dense feature with the interactions (into a row vector)
                # approach 1: all
                # Zflat = Z.view((batch_size, -1))
                # approach 2: unique
                _, ni, nj = Z.shape
                # approach 1: tril_indices
                # offset = 0 if self.arch_interaction_itself else -1
                # li, lj = torch.tril_indices(ni, nj, offset=offset)
                # approach 2: custom
                offset = 1 if self.arch_interaction_itself else 0
                li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
                lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
                Zflat = Z[:, li, lj]
                # concatenate dense features and interactions
                R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_o, lS_i):
        # torchbench: only enable sequential forward
        return self.sequential_forward(dense_x, lS_o, lS_i)
        # if ext_dist.my_size > 1:
        #     # multi-node multi-device run
        #     return self.distributed_forward(dense_x, lS_o, lS_i)
        # elif self.ndevices_available <= 1:
        #     # single device run
        #     return self.sequential_forward(dense_x, lS_o, lS_i)
        # else:
        #     # single-node multi-device run
        #     return self.parallel_forward(dense_x, lS_o, lS_i)

    # torchbench: disable distributed forward
    # def distributed_forward(self, dense_x, lS_o, lS_i):
    #     batch_size = dense_x.size()[0]
    #     # WARNING: # of ranks must be <= batch size in distributed_forward call
    #     if batch_size < ext_dist.my_size:
    #         sys.exit(
    #             "ERROR: batch_size (%d) must be larger than number of ranks (%d)"
    #             % (batch_size, ext_dist.my_size)
    #         )
    #     if batch_size % ext_dist.my_size != 0:
    #         sys.exit(
    #             "ERROR: batch_size %d can not split across %d ranks evenly"
    #             % (batch_size, ext_dist.my_size)
    #         )

    #     dense_x = dense_x[ext_dist.get_my_slice(batch_size)]
    #     lS_o = lS_o[self.local_emb_slice]
    #     lS_i = lS_i[self.local_emb_slice]

    #     if (self.ntables != len(lS_o)) or (self.ntables != len(lS_i)):
    #         sys.exit(
    #             "ERROR: corrupted model input detected in distributed_forward call"
    #         )

    #     # embeddings
    #     with record_function("DLRM embedding forward"):
    #         ly = self.apply_emb(lS_o, lS_i)

    #     # WARNING: Note that at this point we have the result of the embedding lookup
    #     # for the entire batch on each rank. We would like to obtain partial results
    #     # corresponding to all embedding lookups, but part of the batch on each rank.
    #     # Therefore, matching the distribution of output of bottom mlp, so that both
    #     # could be used for subsequent interactions on each device.
    #     if self.ntables != len(ly):
    #         sys.exit("ERROR: corrupted intermediate result in distributed_forward call")

    #     a2a_req = ext_dist.alltoall(ly, self.n_emb_per_rank)

    #     with record_function("DLRM bottom mlp forward"):
    #         x = self.apply_mlp(dense_x, self.bot_l)

    #     ly = a2a_req.wait()
    #     ly = list(ly)

    #     # interactions
    #     with record_function("DLRM interaction forward"):
    #         z = self.interact_features(x, ly)

    #     # top mlp
    #     with record_function("DLRM top mlp forward"):
    #         # quantize top mlp's input to fp16 if PyTorch's built-in fp16 quantization is used.
    #         if self.quantize_mlp_input_with_half_call:
    #             z = z.half()
    #         p = self.apply_mlp(z, self.top_l)

    #     # clamp output if needed
    #     if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
    #         z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
    #     else:
    #         z = p

    #     return z

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i)
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        # print(z.detach().cpu().numpy())

        # quantize top mlp's input to fp16 if PyTorch's built-in fp16 quantization is used.
        if self.quantize_mlp_input_with_half_call:
            z = z.half()

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z

    def parallel_forward(self, dense_x, lS_o, lS_i):
        ### prepare model (overwrite) ###
        # WARNING: # of devices must be >= batch size in parallel_forward call
        batch_size = dense_x.size()[0]
        ndevices = min(self.ndevices_available, batch_size, self.ntables)
        device_ids = range(ndevices)
        # WARNING: must redistribute the model if mini-batch size changes(this is common
        # for last mini-batch, when # of elements in the dataset/batch size is not even
        if self.ndevices_in_use != ndevices:
            self.ndevices_in_use = ndevices
            self.prepare_parallel_model(ndevices)
        elif self.sync_dense_params:
            # When training, replicate the new/updated mlp weights each iteration.
            # For inference-only, this code should never run.
            self.bot_l_replicas = replicate(self.bot_l, device_ids)
            self.top_l_replicas = replicate(self.top_l, device_ids)

        ### prepare input (overwrite) ###
        # scatter dense features (data parallelism)
        # print(dense_x.device)
        dense_x = scatter(dense_x, device_ids, dim=0)
        # distribute sparse features (model parallelism)
        if (self.ntables != len(lS_o)) or (self.ntables != len(lS_i)):
            sys.exit("ERROR: corrupted model input detected in parallel_forward call")

        lS_o = [
            lS_o[k].to(torch.device("cuda:" + str(k % ndevices)))
            for k in range(self.ntables)
        ]
        lS_i = [
            lS_i[k].to(torch.device("cuda:" + str(k % ndevices)))
            for k in range(self.ntables)
        ]

        ### compute results in parallel ###
        # bottom mlp
        # WARNING: Note that the self.bot_l is a list of bottom mlp modules
        # that have been replicated across devices, while dense_x is a tuple of dense
        # inputs that has been scattered across devices on the first (batch) dimension.
        # The output is a list of tensors scattered across devices according to the
        # distribution of dense_x.
        x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)
        # debug prints
        # print(x)

        # embeddings
        ly = self.apply_emb(lS_o, lS_i)
        # debug prints
        # print(ly)

        # butterfly shuffle (implemented inefficiently for now)
        # WARNING: Note that at this point we have the result of the embedding lookup
        # for the entire batch on each device. We would like to obtain partial results
        # corresponding to all embedding lookups, but part of the batch on each device.
        # Therefore, matching the distribution of output of bottom mlp, so that both
        # could be used for subsequent interactions on each device.
        if self.ntables != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")

        t_list = [scatter(ly[k], device_ids, dim=0) for k in range(self.ntables)]

        # adjust the list to be ordered per device
        ly = list(map(lambda y: list(y), zip(*t_list)))
        # debug prints
        # print(ly)

        # interactions
        z = parallel_apply(self.interact_features_l, list(zip(itertools.repeat(self.interact_features),x,ly)))
        # debug prints
        # print(z)

        if self.quantize_mlp_input_with_half_call:
            z = [tens.half() for tens in z]

        # top mlp
        # WARNING: Note that the self.top_l is a list of top mlp modules that
        # have been replicated across devices, while z is a list of interaction results
        # that by construction are scattered across devices on the first (batch) dim.
        # The output is a list of tensors scattered across devices according to the
        # distribution of z.
        p = parallel_apply(self.top_l_replicas, z, None, device_ids)

        ### gather the distributed results ###
        p0 = gather(p, self.output_d, dim=0)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z0 = torch.clamp(
                p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z0 = p0

        return z0

    def print_weights(self):
        if self.use_fbgemm_gpu and len(self.fbgemm_emb_l):
            ntables_l = [
                len(e.fbgemm_gpu_emb_bag.embedding_specs) for e in self.fbgemm_emb_l
            ]
            for j in range(ntables_l[0] + 1):
                for k, e in enumerate(self.fbgemm_emb_l):
                    if j < ntables_l[k]:
                        print(
                            e.fbgemm_gpu_emb_bag.split_embedding_weights()[j]
                            .detach()
                            .cpu()
                            .numpy()
                        )
        elif self.quantize_bits != 32:
            for e in self.emb_l_q:
                print(e.data.detach().cpu().numpy())
        else:  # if self.emb_l:
            for param in self.emb_l.parameters():
                print(param.detach().cpu().numpy())
        if isinstance(self.v_W_l, nn.ParameterList):
            for param in self.v_W_l.parameters():
                print(param.detach().cpu().numpy())
        for param in self.bot_l.parameters():
            print(param.detach().cpu().numpy())
        for param in self.top_l.parameters():
            print(param.detach().cpu().numpy())