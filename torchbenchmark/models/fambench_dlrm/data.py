import os
import torch
import sys
import numpy as np
from torchbenchmark import REPO_PATH
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
    import dlrm_data_pytorch as dp
    # mixed-dimension trick
    from tricks.md_embedding_bag import md_solver

def prep_data(args):
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    if args.data_generation == "dataset":
        train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
        table_feature_map = {idx: idx for idx in range(len(train_data.counts))}
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

        ln_emb = train_data.counts
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(
                list(
                    map(
                        lambda x: x if x < args.max_ind_range else args.max_ind_range,
                        ln_emb,
                    )
                )
            )
        else:
            ln_emb = np.array(ln_emb)
        m_den = train_data.m_den
        ln_bot[0] = m_den
    else:
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        train_data, train_ld, test_data, test_ld = dp.make_random_data_and_loader(
            args, ln_emb, m_den, cache_size=args.precache_ml_data
        )
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)
    nbatches_in_use = nbatches_test if args.inference_only else nbatches
    assert nbatches_in_use > args.warmup_steps, (f"Change --warmup-steps={args.warmup_steps} to be lower than {nbatches_in_use}.")

    args.ln_emb = ln_emb.tolist()

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    ln_emb = np.asarray(ln_emb)
    num_fea = ln_emb.size + 1  # num sparse + num dense features

    if args.use_fbgemm_gpu:
        assert m_spa % 4 == 0, (
            f"{m_spa} % 4 is not 0, but fbgemm_gpu requires the embedding dim "
            + "(--arch-sparse-feature-size number) to be evenly divisible by 4."
        )

    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_project_size > 0:
            num_int = num_fea * args.arch_project_size + m_den_out
        else:
            if args.arch_interaction_itself:
                num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
            else:
                num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )
    if args.qr_flag:
        if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
            sys.exit(
                "ERROR: 2 arch-sparse-feature-size "
                + str(2 * m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
                + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
            )
        if args.qr_operation != "concat" and m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    else:
        if m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    # assign mixed dimensions if applicable
    if args.md_flag:
        m_spa = md_solver(
            torch.tensor(ln_emb),
            args.md_temperature,  # alpha
            d0=m_spa,
            round_dim=args.md_round_dims,
        ).tolist()
        if args.use_fbgemm_gpu:
            for m in m_spa:
                assert m % 4 == 0, (
                    "Found an incompatible embedding dim in m_spa. "
                    + f"{m} % 4 is not 0, but fbgemm_gpu requires the "
                    + "embedding dim to be evenly divisible by 4."
                )
    return ln_bot, ln_emb, ln_top, m_spa