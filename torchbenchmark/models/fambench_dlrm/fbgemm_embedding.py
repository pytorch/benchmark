import torch.nn as nn
import torch
import os
import sys
import numpy as np
from torchbenchmark import REPO_PATH
# This file assumes fbgemm_gpu is installed
import fbgemm_gpu
from fbgemm_gpu import split_table_batched_embeddings_ops
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    CacheAlgorithm,
    PoolingMode,
    OptimType,
    SparseType,
    SplitTableBatchedEmbeddingBagsCodegen,
    IntNBitTableBatchedEmbeddingBagsCodegen,
)

class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass
DLRM_PATH = os.path.join(REPO_PATH, "submodules", "FAMBench", "benchmarks", "dlrm", "ootb")
with add_path(DLRM_PATH):
    # mixed-dimension trick
    from tricks.md_embedding_bag import PrEmbeddingBag

# quantize_fbgemm_gpu_embedding_bag is partially lifted from
# fbgemm_gpu/test/split_embedding_inference_converter.py, def _quantize_split_embs.
# Converts SplitTableBatchedEmbeddingBagsCodegen to IntNBitTableBatchedEmbeddingBagsCodegen
def quantize_fbgemm_gpu_embedding_bag(model, quantize_type, device):
    embedding_specs = []
    if device.type == "cpu":
        emb_location = split_table_batched_embeddings_ops.EmbeddingLocation.HOST
    else:
        emb_location = split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE

    for (E, D, _, _) in model.embedding_specs:
        weights_ty = quantize_type
        if D % weights_ty.align_size() != 0:
            assert D % 4 == 0
            weights_ty = (
                SparseType.FP16
            )  # fall back to FP16 if dimension couldn't be aligned with the required size
        embedding_specs.append(("", E, D, weights_ty, emb_location))

    q_model = (
        split_table_batched_embeddings_ops.IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            pooling_mode=model.pooling_mode,
            device=device,
        )
    )
    q_model.initialize_weights()
    for t, (_, _, _, weight_ty, _) in enumerate(embedding_specs):
        if weight_ty == SparseType.FP16:
            original_weight = model.split_embedding_weights()[t]
            q_weight = original_weight.half()
            weights = torch.tensor(q_weight.cpu().numpy().view(np.uint8))
            q_model.split_embedding_weights()[t][0].data.copy_(weights)

        elif weight_ty == SparseType.INT8:
            original_weight = model.split_embedding_weights()[t]
            q_weight = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                original_weight
            )
            weights = q_weight[:, :-8]
            scale_shift = torch.tensor(
                q_weight[:, -8:]
                .contiguous()
                .cpu()
                .numpy()
                .view(np.float32)
                .astype(np.float16)
                .view(np.uint8)
            )
            q_model.split_embedding_weights()[t][0].data.copy_(weights)
            q_model.split_embedding_weights()[t][1].data.copy_(scale_shift)

        elif weight_ty == SparseType.INT4 or weight_ty == SparseType.INT2:
            original_weight = model.split_embedding_weights()[t]
            q_weight = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                original_weight,
                bit_rate=quantize_type.bit_rate(),
            )
            weights = q_weight[:, :-4]
            scale_shift = torch.tensor(
                q_weight[:, -4:].contiguous().cpu().numpy().view(np.uint8)
            )
            q_model.split_embedding_weights()[t][0].data.copy_(weights)
            q_model.split_embedding_weights()[t][1].data.copy_(scale_shift)
    return q_model

def create_fbgemm_gpu_emb_bag(
    device,
    emb_l,
    m_spa,
    quantize_bits,
    learning_rate,
    codegen_preference=None,
    requires_grad=True,
):
    if isinstance(emb_l[0], PrEmbeddingBag):
        emb_l = [e.embs for e in emb_l]
    if isinstance(emb_l[0], nn.EmbeddingBag):
        emb_l = [e.weight for e in emb_l]
    Es = [e.shape[0] for e in emb_l]

    if isinstance(m_spa, list):
        Ds = m_spa
    else:
        Ds = [m_spa for _ in emb_l]

    if device.type == "cpu":
        emb_location = split_table_batched_embeddings_ops.EmbeddingLocation.HOST
        compute_device = split_table_batched_embeddings_ops.ComputeDevice.CPU
    else:
        emb_location = split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE
        compute_device = split_table_batched_embeddings_ops.ComputeDevice.CUDA
    pooling_mode = PoolingMode.SUM
    cache_algorithm = CacheAlgorithm.LRU

    sparse_type_dict = {
        4: SparseType.INT4,
        8: SparseType.INT8,
        16: SparseType.FP16,
        32: SparseType.FP32,
    }
    codegen_type_dict = {
        4: "IntN",
        8: "Split" if codegen_preference != "IntN" else "IntN",
        16: "Split" if codegen_preference != "IntN" else "IntN",
        32: "Split",
    }

    codegen_type = codegen_type_dict[quantize_bits]
    quantize_type = sparse_type_dict[quantize_bits]
    if codegen_type == "IntN":
        # Create non-quantized model and then call quantize_fbgemm_gpu_embedding_bag
        fbgemm_gpu_emb_bag = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    E,  # num of rows in the table
                    D,  # num of columns in the table
                    split_table_batched_embeddings_ops.EmbeddingLocation.HOST,
                    split_table_batched_embeddings_ops.ComputeDevice.CPU,
                )
                for (E, D) in zip(Es, Ds)
            ],
            weights_precision=SparseType.FP32,
            optimizer=OptimType.EXACT_SGD,
            learning_rate=learning_rate,
            cache_algorithm=cache_algorithm,
            pooling_mode=pooling_mode,
        ).to(device)
        if quantize_type == quantize_type.FP16:
            weights = fbgemm_gpu_emb_bag.split_embedding_weights()
            for i, emb in enumerate(weights):
                emb.data.copy_(emb_l[i])

        elif quantize_type == quantize_type.INT8:
            # copy quantized values upsampled/recasted to FP32
            for i in range(len(Es)):
                fbgemm_gpu_emb_bag.split_embedding_weights()[i].data.copy_(
                    torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(emb_l[i])
                )
        elif quantize_type == quantize_type.INT4:
            # copy quantized values upsampled/recasted to FP32
            for i in range(len(Es)):
                fbgemm_gpu_emb_bag.split_embedding_weights()[i].data.copy_(
                    torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(
                        emb_l[i],
                        bit_rate=quantize_type.bit_rate(),
                    )
                )
        fbgemm_gpu_emb_bag = quantize_fbgemm_gpu_embedding_bag(
            fbgemm_gpu_emb_bag, quantize_type, device
        )
    else:
        fbgemm_gpu_emb_bag = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    E,  # num of rows in the table
                    D,  # num of columns in the table
                    emb_location,
                    compute_device,
                )
                for (E, D) in zip(Es, Ds)
            ],
            weights_precision=quantize_type,
            optimizer=OptimType.EXACT_SGD,
            learning_rate=learning_rate,
            cache_algorithm=cache_algorithm,
            pooling_mode=pooling_mode,
        ).to(device)

        weights = fbgemm_gpu_emb_bag.split_embedding_weights()
        for i, emb in enumerate(weights):
            emb.data.copy_(emb_l[i])

    if not requires_grad:
        torch.no_grad()
        torch.set_grad_enabled(False)

    return fbgemm_gpu_emb_bag

# The purpose of this wrapper is to encapsulate the format conversions to/from fbgemm_gpu
# so parallel_apply() executes the format-in -> fbgemm_gpu op -> format-out instructions
# for each respective GPU in parallel.
class fbgemm_gpu_emb_bag_wrapper(nn.Module):
    def __init__(
        self,
        device,
        emb_l,
        m_spa,
        quantize_bits,
        learning_rate,
        codegen_preference,
        requires_grad,
    ):
        super(fbgemm_gpu_emb_bag_wrapper, self).__init__()
        self.fbgemm_gpu_emb_bag = create_fbgemm_gpu_emb_bag(
            device,
            emb_l,
            m_spa,
            quantize_bits,
            learning_rate,
            codegen_preference,
            requires_grad,
        )
        self.device = device
        self.m_spa = m_spa
        # create cumsum array for mixed dimension support
        if isinstance(m_spa, list):
            self.m_spa_cumsum = np.cumsum([0] + m_spa)
        if not requires_grad:
            torch.no_grad()
            torch.set_grad_enabled(False)

    def forward(self, lS_o, lS_i, v_W_l=None):

        # convert offsets to fbgemm format
        lengths_list = list(map(len, lS_i))
        indices_lengths_cumsum = np.cumsum([0] + lengths_list)
        if isinstance(lS_o, list):
            lS_o = torch.stack(lS_o)
        lS_o = lS_o.to(self.device)
        lS_o += torch.from_numpy(indices_lengths_cumsum[:-1, np.newaxis]).to(
            self.device
        )
        numel = torch.tensor([indices_lengths_cumsum[-1]], dtype=torch.long).to(
            self.device
        )
        lS_o = torch.cat((lS_o.flatten(), numel))

        # create per_sample_weights
        if v_W_l:
            per_sample_weights = torch.cat(
                [a.gather(0, b) for a, b in zip(v_W_l, lS_i)]
            )
        else:
            per_sample_weights = None

        # convert indices to fbgemm_gpu format
        if isinstance(lS_i, torch.Tensor):
            lS_i = [lS_i]
        lS_i = torch.cat(lS_i, dim=0).to(self.device)

        if isinstance(self.fbgemm_gpu_emb_bag, IntNBitTableBatchedEmbeddingBagsCodegen):
            lS_o = lS_o.int()
            lS_i = lS_i.int()

        # gpu embedding bag op
        ly = self.fbgemm_gpu_emb_bag(lS_i, lS_o, per_sample_weights)

        # convert the results to the next layer's input format.
        if isinstance(self.m_spa, list):
            # handle mixed dimensions case.
            ly = [
                ly[:, s:e]
                for (s, e) in zip(self.m_spa_cumsum[:-1], self.m_spa_cumsum[1:])
            ]
        else:
            # handle case in which all tables share the same column dimension.
            cols = self.m_spa
            ntables = len(self.fbgemm_gpu_emb_bag.embedding_specs)
            ly = ly.reshape(-1, ntables, cols).swapaxes(0, 1)
            ly = list(ly)
        return ly