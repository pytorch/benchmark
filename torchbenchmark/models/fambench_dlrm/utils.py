import torch

# The following function is a wrapper to avoid checking this multiple times in th
# loop below.
def unpack_batch(b, device):
    # Experiment with unweighted samples
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()).to(device), None

def dlrm_wrap(dlrm, X, lS_o, lS_i, use_gpu, device, ndevices=1):
    if dlrm.quantize_mlp_input_with_half_call:
        X = X.half()
    if use_gpu:
        # lS_i can be either a list of tensors or a stacked tensor.
        # Handle each case below:
        if ndevices == 1:
            lS_i = (
                [S_i.to(device) for S_i in lS_i]
                if isinstance(lS_i, list)
                else lS_i.to(device)
            )
            lS_o = (
                [S_o.to(device) for S_o in lS_o]
                if isinstance(lS_o, list)
                else lS_o.to(device)
            )
    return dlrm(X.to(device), lS_o, lS_i)


def loss_fn_wrap(dlrm, args, Z, T, use_gpu, device):
    if args.loss_function == "mse" or args.loss_function == "bce":
        return dlrm.loss_fn(Z, T.to(device))
    elif args.loss_function == "wbce":
        loss_ws_ = dlrm.loss_ws[T.data.view(-1).long()].view_as(T).to(device)
        loss_fn_ = dlrm.loss_fn(Z, T.to(device))
        loss_sc_ = loss_ws_ * loss_fn_
        return loss_sc_.mean()


def prefetch(dl, device):
    out = []
    for inputBatch in dl:
        X, lS_o, lS_i, T = inputBatch
        lS_i = (
                [S_i.to(device) for S_i in lS_i]
                if isinstance(lS_i, list)
                else lS_i.to(device)
            )
        lS_o = (
            [S_o.to(device) for S_o in lS_o]
            if isinstance(lS_o, list)
            else lS_o.to(device)
        )
        out.append(tuple([X.to(device), lS_o, lS_i, T]))
    return out