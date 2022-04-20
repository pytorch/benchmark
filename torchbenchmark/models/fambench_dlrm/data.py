import numpy as np

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